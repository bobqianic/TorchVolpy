from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import math
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from torch_volpy.movie import Movie


_SIGNAL_FILTER_METHODS = {"sos32"}


@dataclass
class SpikePursuitResult:
    roi_id: int
    t: torch.Tensor
    ts: torch.Tensor
    t_rec: torch.Tensor
    t_sub: torch.Tensor
    spikes: torch.Tensor
    num_spikes: List[int]
    low_spikes: bool
    templates: torch.Tensor
    snr: float
    thresh: float
    weights: torch.Tensor
    locality: bool
    context_coord: torch.Tensor
    mean_im: torch.Tensor
    F0: torch.Tensor
    dFF: torch.Tensor
    rawROI: Dict[str, torch.Tensor]


@dataclass
class SpikePursuitPreparedROI:
    roi_id: int
    t0: torch.Tensor
    weights0: torch.Tensor
    bw: torch.Tensor
    bw_flat: torch.Tensor
    notbw_flat: torch.Tensor
    F0: torch.Tensor
    mean_im: torch.Tensor
    context_coord: torch.Tensor
    Ub: torch.Tensor
    solve_bg: Callable[[torch.Tensor], torch.Tensor]
    recon: torch.Tensor
    pred_pixels: torch.Tensor
    ridge_alpha: float
    solve_recon: Optional[Callable[[torch.Tensor], torch.Tensor]]


@dataclass(frozen=True)
class _ContextPatchSpec:
    roi_id: int
    x0: int
    x1: int
    y0: int
    y1: int


class Spikepursuit:
    """
    PyTorch port of the original volspike/spikepursuit-style algorithm.

    Assumptions
    -----------
    - movie is a Movie instance backed by HDF5 with shape (T, Y, X) or (T, Y, X, C)
    - roi_mask is either a 2D label tensor where:
        * background is 0
        * each ROI has a unique non-zero integer/label value
      or a 3D Mask-RCNN-style instance stack with shape (N, Y, X)
    - all numeric work is performed with PyTorch tensors

    Notes
    -----
    This is algorithmically faithful to the original implementation, but a few
    scipy/sklearn operations are replaced with native torch equivalents:
    - spike denoising uses a Torch-native VolPy-compatible Butterworth filtfilt
    - peak finding is implemented with tensor comparisons
    - PCA/background extraction uses torch.pca_lowrank
    - ridge regression follows VolPy/scikit-learn LSQR semantics for parity
    - morphology uses conv2d-based binary dilation
    """

    def __init__(
        self,
        movie,
        roi_mask: torch.Tensor,
        *,
        channel: Optional[int] = None,
        fr: float = 400.0,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
        cache_movie: bool = False,
        cache_movie_device: Union[str, torch.device] = "cpu",
        template_size: float = 0.02,
        context_size: int = 35,
        censor_size: int = 12,
        visualize_roi: bool = False,
        flip_signal: bool = True,
        hp_freq_pb: float = 10.0,
        nPC_bg: int = 8,
        ridge_bg: float = 0.01,
        hp_freq: float = 1.0,
        clip: int = 100,
        threshold_method: str = "adaptive_threshold",
        min_spikes: int = 5,
        pnorm: float = 0.5,
        threshold: float = 2.0,
        sigmas: Sequence[float] = (1.0, 1.5, 2.0),
        n_iter: int = 2,
        weight_update: str = "ridge",
        do_plot: bool = False,
        do_cross_val: bool = False,
        sub_freq: float = 20.0,
        batch_filter_workspace_bytes: Optional[int] = None,
        prefetch_next_batch_patch: bool = False,
        keep_cuda_cache_between_batches: bool = False,
        cuda_batch_safety_margin: float = 1.0,
        signal_filter_method: str = "sos32",
        eps: float = 1e-8,
    ) -> None:
        self.movie = movie
        self.channel = channel
        self.fr = float(fr)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype
        self.eps = float(eps)
        self._movie_cache: Optional[torch.Tensor] = None
        self._patch_read_override: Optional[Tuple[torch.Tensor, int, int]] = None
        self._prepared_patch_override: Optional[Tuple[torch.Tensor, int, int]] = None
        self._active_batch_patch_bytes: Optional[int] = None
        self.prefetch_next_batch_patch = bool(prefetch_next_batch_patch)
        self.keep_cuda_cache_between_batches = bool(keep_cuda_cache_between_batches)
        self.cuda_batch_safety_margin = min(1.0, max(0.1, float(cuda_batch_safety_margin)))
        self._disk_cache: Dict[Tuple[int, str, torch.dtype], torch.Tensor] = {}
        self._gaussian_kernel2d_cache: Dict[Tuple[int, float, str, torch.dtype], torch.Tensor] = {}
        self._butter_sos_coefficients_cache: Dict[
            Tuple[int, float, str, str, torch.dtype],
            torch.Tensor,
        ] = {}
        self.batch_filter_workspace_bytes = (
            None
            if batch_filter_workspace_bytes is None or int(batch_filter_workspace_bytes) <= 0
            else int(batch_filter_workspace_bytes)
        )

        self.args = {
            "template_size": float(template_size),
            "context_size": int(context_size),
            "censor_size": int(censor_size),
            "visualize_ROI": bool(visualize_roi),
            "flip_signal": bool(flip_signal),
            "hp_freq_pb": float(hp_freq_pb),
            "nPC_bg": int(nPC_bg),
            "ridge_bg": float(ridge_bg),
            "hp_freq": float(hp_freq),
            "clip": int(clip),
            "threshold_method": str(threshold_method),
            "min_spikes": int(min_spikes),
            "pnorm": float(pnorm),
            "threshold": float(threshold),
            "sigmas": tuple(float(s) for s in sigmas),
            "n_iter": int(n_iter),
            "weight_update": str(weight_update),
            "do_plot": bool(do_plot),
            "do_cross_val": bool(do_cross_val),
            "sub_freq": float(sub_freq),
            "signal_filter_method": str(signal_filter_method).lower(),
        }
        if self.args["signal_filter_method"] not in _SIGNAL_FILTER_METHODS:
            allowed = "', '".join(sorted(_SIGNAL_FILTER_METHODS))
            raise ValueError(f"signal_filter_method must be one of: '{allowed}'.")

        if roi_mask.ndim not in (2, 3):
            raise ValueError("roi_mask must be a 2D label tensor or a 3D instance stack.")

        self.roi_mask = roi_mask.to(self.device)
        self._roi_mask_is_stack = self.roi_mask.ndim == 3
        self.roi_ids = self._get_roi_ids(self.roi_mask)
        self._roi_id_set = frozenset(int(v) for v in self.roi_ids.detach().cpu().tolist())

        movie_shape = self.movie.shape
        if len(movie_shape) == 3:
            _, y, x = movie_shape
        elif len(movie_shape) == 4:
            _, y, x, _ = movie_shape
            if self.channel is None:
                raise ValueError("Movie has shape (T, Y, X, C). Please provide channel=...")
        else:
            raise ValueError(f"Unsupported movie shape: {movie_shape}")

        self._frame_shape = (int(y), int(x))
        roi_frame_shape = tuple(int(v) for v in self.roi_mask.shape[-2:])
        if self._frame_shape != roi_frame_shape:
            raise ValueError(
                f"ROI mask shape {tuple(self.roi_mask.shape)} does not match movie frame shape {self._frame_shape}."
            )

        if self.args["do_cross_val"]:
            raise NotImplementedError("do_cross_val=True is not implemented in this PyTorch port.")

        self._filter_fft_cache: Dict[
            Tuple[str, str, int, int, int, int, int, int],
            Tuple[torch.Tensor, torch.Tensor],
        ] = {}

        if cache_movie:
            self.preload_movie(cache_movie_device)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def fit(
        self,
        roi_ids: Optional[Iterable[int]] = None,
        *,
        batch_patch_bytes: Optional[int] = None,
        max_rois_per_batch: Optional[int] = None,
    ) -> Dict[int, SpikePursuitResult]:
        out: Dict[int, SpikePursuitResult] = {}
        for result in self.iter_fit(
            roi_ids,
            batch_patch_bytes=batch_patch_bytes,
            max_rois_per_batch=max_rois_per_batch,
        ):
            out[int(result.roi_id)] = result
        return out

    def iter_fit(
        self,
        roi_ids: Optional[Iterable[int]] = None,
        *,
        batch_patch_bytes: Optional[int] = None,
        max_rois_per_batch: Optional[int] = None,
    ) -> Iterator[SpikePursuitResult]:
        """
        Yield ROI fits while batching movie reads by spatial union patch.

        Each batch reads a shared movie patch once, high-pass filters the union
        patch once, then fits the ROIs whose context windows are contained in
        that patch. This keeps peak memory bounded by ``batch_patch_bytes``
        instead of loading/filtering all ROI patches independently.
        """
        for prepared in self.iter_prepare(
            roi_ids,
            batch_patch_bytes=batch_patch_bytes,
            max_rois_per_batch=max_rois_per_batch,
        ):
            yield self.fit_prepared_roi(prepared)

    def iter_prepare(
        self,
        roi_ids: Optional[Iterable[int]] = None,
        *,
        batch_patch_bytes: Optional[int] = None,
        max_rois_per_batch: Optional[int] = None,
    ) -> Iterator[SpikePursuitPreparedROI]:
        """
        Yield prepared ROI state while batching movie reads by spatial union patch.

        The prepared state contains the expensive movie/background/predictor
        terms and can be passed to ``fit_prepared_roi`` for multiple
        threshold-parameter sweeps.
        """
        ids = [int(roi_id) for roi_id in (self.roi_ids.tolist() if roi_ids is None else roi_ids)]
        if not ids:
            return

        specs = [self._context_patch_spec(roi_id) for roi_id in ids]
        max_bytes = self._normalize_batch_patch_bytes(batch_patch_bytes)
        max_rois = None if max_rois_per_batch is None else int(max_rois_per_batch)
        if max_rois is not None and max_rois <= 0:
            max_rois = None

        previous_active_budget = self._active_batch_patch_bytes
        self._active_batch_patch_bytes = max_bytes
        prefetch_executor = self._create_patch_prefetch_executor()
        prefetch_future: Optional[Future[torch.Tensor]] = None
        prefetch_bounds: Optional[Tuple[int, int, int, int]] = None
        try:
            batches = self._context_patch_batches(
                specs,
                max_patch_bytes=max_bytes,
                max_rois_per_batch=max_rois,
            )
            for batch_index, batch in enumerate(batches):
                x0, x1, y0, y1 = self._context_batch_bounds(batch)
                next_batch = batches[batch_index + 1] if batch_index + 1 < len(batches) else None
                patch = None
                data_hp = None
                try:
                    patch = self._read_batch_patch(
                        x0,
                        x1,
                        y0,
                        y1,
                        prefetch_future=prefetch_future,
                        prefetch_bounds=prefetch_bounds,
                    )
                    prefetch_future = None
                    prefetch_bounds = None
                    if prefetch_executor is not None and next_batch is not None:
                        prefetch_bounds = self._context_batch_bounds(next_batch)
                        prefetch_future = self._submit_movie_patch_prefetch(prefetch_executor, *prefetch_bounds)
                    data_hp = self._prepare_context_batch_highpass(patch)
                except torch.cuda.OutOfMemoryError:
                    if self.device.type != "cuda":
                        raise
                    if prefetch_bounds == (x0, x1, y0, y1):
                        prefetch_future = None
                        prefetch_bounds = None
                    del data_hp
                    del patch
                    self._flush_cuda_batch_cache()
                    patch = self._read_movie_patch_from_source(x0, x1, y0, y1)
                    data_hp = self._prepare_context_batch_highpass(patch)

                previous_override = self._patch_read_override
                previous_prepared_override = self._prepared_patch_override
                self._patch_read_override = (patch, x0, y0)
                self._prepared_patch_override = (data_hp, x0, y0)
                oom_during_fit = False
                try:
                    for spec in batch:
                        yield self.prepare_roi(spec.roi_id)
                except torch.cuda.OutOfMemoryError:
                    if self.device.type != "cuda":
                        raise
                    oom_during_fit = True
                    raise
                finally:
                    self._patch_read_override = previous_override
                    self._prepared_patch_override = previous_prepared_override
                    del data_hp
                    del patch

                    if oom_during_fit or self._should_flush_cuda_batch_cache(next_batch, max_bytes=max_bytes):
                        self._flush_cuda_batch_cache()
        finally:
            if prefetch_future is not None:
                prefetch_future.cancel()
            if prefetch_executor is not None:
                prefetch_executor.shutdown(wait=False, cancel_futures=True)
            self._active_batch_patch_bytes = previous_active_budget

    def preload_movie(self, device: Union[str, torch.device] = "cpu") -> "Spikepursuit":
        """
        Cache the full movie on CPU or GPU so ROI extraction does not reread
        full-frame HDF5 chunks for every spatial crop.
        """
        cache_device = torch.device(device)
        if len(self.movie.shape) == 3:
            data = self.movie.read(
                slice(None),
                as_tensor=True,
                dtype=None,
                device=cache_device,
            )
        else:
            data = self.movie.read(
                (slice(None), slice(None), slice(None), self.channel),
                as_tensor=True,
                dtype=None,
                device=cache_device,
            )
        self._movie_cache = data.to(dtype=self.dtype)
        return self

    @torch.inference_mode()
    def fit_roi(
        self,
        roi_id: int,
        weights_init: Optional[torch.Tensor] = None,
    ) -> SpikePursuitResult:
        prepared = self.prepare_roi(roi_id, weights_init=weights_init)
        return self.fit_prepared_roi(prepared)

    @torch.inference_mode()
    def prepare_roi(
        self,
        roi_id: int,
        weights_init: Optional[torch.Tensor] = None,
    ) -> SpikePursuitPreparedROI:
        if int(roi_id) not in self._roi_id_set:
            raise KeyError(f"ROI id {roi_id} not found in roi_mask.")

        bw_full = self._roi_binary_mask(int(roi_id))
        if not torch.any(bw_full):
            raise ValueError(f"ROI id {roi_id} is empty.")

        prepared = self._extract_prepared_context_patch_data(bw_full)
        if prepared is None:
            patch, bw, notbw, context_coord = self._extract_context_patch(bw_full)
            if self.args["flip_signal"]:
                patch.neg_()

            T, h, w = patch.shape
            mean_im = patch.mean(dim=0)
            raw_roi_mean = patch[:, bw].mean(dim=1)

            data_hp = self._prepare_context_batch_highpass(patch, flip_signal=False).reshape(T, -1)
            del patch
        else:
            data_hp, raw_roi_mean, mean_im, bw, notbw, context_coord = prepared
            T, h, w = data_hp.shape
            data_hp = data_hp.reshape(T, -1)

        bw_flat = bw.reshape(-1)
        notbw_flat = notbw.reshape(-1)
        F0 = torch.abs(raw_roi_mean - data_hp[:, bw_flat].mean(dim=1)).clamp_min(self.eps)
        del raw_roi_mean

        if weights_init is None:
            t0 = data_hp[:, bw_flat].mean(dim=1)
            weights0 = bw.to(self.dtype)
        else:
            if weights_init.shape != bw.shape:
                raise ValueError(
                    f"weights_init must match cropped ROI patch shape {tuple(bw.shape)}, got {tuple(weights_init.shape)}"
                )
            t0 = data_hp @ weights_init.reshape(-1).to(device=self.device, dtype=self.dtype)
            weights0 = weights_init.to(self.dtype)

        t0 = t0 - t0.mean()

        data_svd = data_hp[:, notbw_flat]
        if data_svd.shape[1] < self.args["nPC_bg"] + 1:
            raise ValueError(
                f"Too few background pixels ({data_svd.shape[1]}) for nPC_bg={self.args['nPC_bg']}. "
                f"Decrease context_size and/or censor_size."
            )

        Ub = self._background_components(data_svd, self.args["nPC_bg"])
        del data_svd
        alpha_bg = self.args["nPC_bg"] * self.args["ridge_bg"]
        solve_bg = self._ridge_solver(Ub, alpha=alpha_bg)
        beta_bg = solve_bg(t0)
        t0 = (t0 - Ub @ beta_bg).to(torch.float32)
        del beta_bg

        pred = self._build_predictor(data_hp.reshape(T, h, w), sigma=1.5, kernel_size=7)
        pred_pixels = pred[:, 1:]
        ridge_alpha = float((pred_pixels.square().sum() * 1e-2).item())

        sigma_idx = min(1, max(0, len(self.args["sigmas"]) - 1))
        sigma = self.args["sigmas"][sigma_idx]
        if abs(float(sigma) - 1.5) <= 1e-12:
            recon = pred
        else:
            recon = self._build_predictor(data_hp.reshape(T, h, w), sigma=sigma, kernel_size=None)
        solve_recon = (
            self._ridge_solver_fit_intercept(recon, alpha=ridge_alpha)
            if self.args["weight_update"].lower() == "ridge"
            else None
        )
        del data_hp

        return SpikePursuitPreparedROI(
            roi_id=int(roi_id),
            t0=t0,
            weights0=weights0,
            bw=bw,
            bw_flat=bw_flat,
            notbw_flat=notbw_flat,
            F0=F0,
            mean_im=mean_im,
            context_coord=context_coord,
            Ub=Ub,
            solve_bg=solve_bg,
            recon=recon,
            pred_pixels=pred_pixels,
            ridge_alpha=float(ridge_alpha),
            solve_recon=solve_recon,
        )

    @torch.inference_mode()
    def fit_prepared_roi(self, prepared: SpikePursuitPreparedROI) -> SpikePursuitResult:
        t0 = prepared.t0
        weights0 = prepared.weights0
        bw = prepared.bw
        bw_flat = prepared.bw_flat
        notbw_flat = prepared.notbw_flat
        h, w = bw.shape
        F0 = prepared.F0
        mean_im = prepared.mean_im
        context_coord = prepared.context_coord
        Ub = prepared.Ub
        solve_bg = prepared.solve_bg
        recon = prepared.recon
        pred_pixels = prepared.pred_pixels
        ridge_alpha = prepared.ridge_alpha
        solve_recon = prepared.solve_recon
        window_length = int(self.fr * self.args["template_size"])

        ts, spikes, t_rec, templates, low_spikes, thresh = self.denoise_spikes(
            t0,
            window_length=window_length,
            fr=self.fr,
            hp_freq=self.args["hp_freq"],
            clip=self.args["clip"],
            threshold_method=self.args["threshold_method"],
            min_spikes=self.args["min_spikes"],
            pnorm=self.args["pnorm"],
            threshold=self.args["threshold"],
            do_plot=self.args["do_plot"] and self.args["n_iter"] <= 1,
        )

        raw_t = t0.clone()
        if spikes.numel() > 0:
            denom = raw_t[spikes].mean()
            if torch.abs(denom) > self.eps:
                raw_t = raw_t * (t0[spikes].mean() / denom)

        rawROI = {
            "t": raw_t,
            "ts": ts.clone(),
            "spikes": spikes.clone(),
            "weights": weights0.clone(),
            "templates": templates.clone(),
        }

        num_spikes = [int(spikes.numel())]
        weights = None
        t = t0.clone()

        for iteration in range(self.args["n_iter"]):
            do_plot_iter = bool(self.args["do_plot"] and iteration == self.args["n_iter"] - 1)
            tr = t_rec.clone().to(self.dtype)

            if self.args["weight_update"].lower() == "nmf":
                C = torch.stack([tr, torch.ones_like(tr)], dim=0)
                CCt = C @ C.t() + self.eps * torch.eye(2, device=self.device, dtype=self.dtype)
                CY = C @ recon[:, 1:]
                A = torch.clamp(torch.linalg.solve(CCt, CY), min=0.0)
                for _ in range(5):
                    for m in range(2):
                        A[m] = A[m] + (CY[m] - CCt[m] @ A) / CCt[m, m].clamp_min(self.eps)
                        if m == 0:
                            A[m] = torch.clamp(A[m], min=0.0)
                weights = torch.cat([
                    torch.zeros(1, device=self.device, dtype=self.dtype),
                    A[0],
                ], dim=0)
            elif self.args["weight_update"].lower() == "ridge":
                if solve_recon is None:
                    raise RuntimeError("Ridge solver was not initialized.")
                weights = solve_recon(tr)
            else:
                raise ValueError("weight_update must be 'ridge' or 'NMF'.")

            t = (recon @ weights).to(torch.float32)
            t = t - t.mean()

            b = solve_bg(t)
            t = (t - Ub @ b).to(torch.float32)

            if spikes.numel() > 0:
                denom = t[spikes].mean()
                if torch.abs(denom) > self.eps:
                    shrink = t0[spikes].mean() / denom
                    weights = weights * shrink.to(weights.dtype)
                    t = t * shrink.to(t.dtype)

            ts, spikes, t_rec, templates, low_spikes, thresh = self.denoise_spikes(
                t,
                window_length=window_length,
                fr=self.fr,
                hp_freq=self.args["hp_freq"],
                clip=self.args["clip"],
                threshold_method=self.args["threshold_method"],
                min_spikes=self.args["min_spikes"],
                pnorm=self.args["pnorm"],
                threshold=self.args["threshold"],
                do_plot=do_plot_iter,
            )
            num_spikes.append(int(spikes.numel()))

        if weights is None:
            weights = self._ridge_fit_intercept(recon, t_rec.to(self.dtype), alpha=ridge_alpha)

        if spikes.numel() > 0:
            t = t - self._numpy_median(t)

        snr = self._compute_snr(t, spikes)

        matrix = pred_pixels.t() @ t_rec
        sigmax = torch.sqrt(pred_pixels.square().sum(dim=0)).clamp_min(self.eps)
        sigmay = torch.sqrt((t_rec ** 2).sum()).clamp_min(self.eps)
        imcorr = matrix / (sigmax * sigmay)

        max_corr_roi = imcorr[bw_flat].max() if torch.any(bw_flat) else torch.tensor(float("-inf"), device=self.device)
        locality = not bool(torch.any(imcorr[notbw_flat] > max_corr_roi).item())

        weights_patch = weights[1:].reshape(h, w)
        weights_fov = torch.zeros(self._frame_shape, device=self.device, dtype=self.dtype)
        x0, x1 = int(context_coord[0, 0]), int(context_coord[0, 1]) + 1
        y0, y1 = int(context_coord[1, 0]), int(context_coord[1, 1]) + 1
        weights_fov[x0:x1, y0:y1] = weights_patch

        t_sub = self.signal_filter(t - t_rec, self.args["sub_freq"], self.fr, order=5, mode="low", dim=0)

        dff = t / F0
        rawROI["dFF"] = rawROI["t"] / F0

        return SpikePursuitResult(
            roi_id=int(prepared.roi_id),
            t=t,
            ts=ts,
            t_rec=t_rec,
            t_sub=t_sub,
            spikes=spikes,
            num_spikes=num_spikes,
            low_spikes=bool(low_spikes),
            templates=templates,
            snr=float(snr),
            thresh=float(thresh),
            weights=weights_fov,
            locality=bool(locality),
            context_coord=context_coord,
            mean_im=mean_im,
            F0=F0,
            dFF=dff,
            rawROI=rawROI,
        )

    # ------------------------------------------------------------------
    # movie + ROI handling
    # ------------------------------------------------------------------

    def _get_roi_ids(self, mask: torch.Tensor) -> torch.Tensor:
        if mask.ndim == 3:
            flat = mask.reshape(mask.shape[0], -1)
            nonempty = torch.any(flat != 0, dim=1)
            ids = torch.nonzero(nonempty, as_tuple=False).flatten() + 1
            return ids.to(torch.int64)

        ids = torch.unique(mask)
        ids = ids[ids != 0]
        return ids.to(torch.int64)

    def _roi_binary_mask(self, roi_id: int) -> torch.Tensor:
        roi_id = int(roi_id)
        if self._roi_mask_is_stack:
            plane_index = roi_id - 1
            if plane_index < 0 or plane_index >= int(self.roi_mask.shape[0]):
                raise KeyError(f"ROI id {roi_id} not found in roi_mask.")
            return self.roi_mask[plane_index] != 0
        return self.roi_mask == roi_id

    def _extract_context_patch(
        self, bw_full: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        spec = self._context_patch_spec_for_mask(bw_full, roi_id=0)
        x0, x1, y0, y1 = spec.x0, spec.x1, spec.y0, spec.y1

        bw = bw_full[x0:x1, y0:y1].to(torch.bool)
        disk_fp = self._disk(self.args["censor_size"], device=self.device, dtype=self.dtype)
        notbw = ~self._binary_dilate(bw.to(self.dtype), disk_fp)

        patch = self._read_movie_patch(x0, x1, y0, y1)
        context_coord = torch.tensor([[x0, x1 - 1], [y0, y1 - 1]], device=self.device, dtype=torch.long)
        return patch, bw, notbw, context_coord

    def _prepare_context_batch_highpass(
        self,
        patch: torch.Tensor,
        *,
        flip_signal: Optional[bool] = None,
    ) -> torch.Tensor:
        T, H, W = patch.shape
        pixels = H * W
        chunk_pixels = self._batch_highpass_chunk_pixels(T, pixels, budget_bytes=self._active_batch_patch_bytes)
        flat = patch.reshape(T, pixels)
        do_flip = self.args["flip_signal"] if flip_signal is None else bool(flip_signal)

        if chunk_pixels >= pixels:
            data = flat.to(device=self.device, dtype=self.dtype, non_blocking=True).clone()
            if do_flip:
                data.neg_()
            data.sub_(data.mean(dim=0, keepdim=True))
            data.sub_(data.mean(dim=0, keepdim=True))
            data_hp = self.signal_filter(
                data,
                self.args["hp_freq_pb"],
                self.fr,
                order=3,
                mode="high",
                dim=0,
            )
            return data_hp.reshape(T, H, W).contiguous()

        data_hp = torch.empty((T, pixels), device=self.device, dtype=self.dtype)
        for start in range(0, pixels, chunk_pixels):
            stop = min(start + chunk_pixels, pixels)
            data = flat[:, start:stop].to(device=self.device, dtype=self.dtype, non_blocking=True).clone()
            if do_flip:
                data.neg_()
            data.sub_(data.mean(dim=0, keepdim=True))
            data.sub_(data.mean(dim=0, keepdim=True))
            filtered = self.signal_filter(
                data,
                self.args["hp_freq_pb"],
                self.fr,
                order=3,
                mode="high",
                dim=0,
            )
            data_hp[:, start:stop] = filtered
            del filtered
            del data
        return data_hp.reshape(T, H, W).contiguous()

    def _extract_prepared_context_patch_data(
        self,
        bw_full: torch.Tensor,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        if self._prepared_patch_override is None:
            return None

        data_hp, base_x, base_y = self._prepared_patch_override
        spec = self._context_patch_spec_for_mask(bw_full, roi_id=0)
        x0, x1, y0, y1 = spec.x0, spec.x1, spec.y0, spec.y1
        rel_x0 = int(x0) - int(base_x)
        rel_x1 = int(x1) - int(base_x)
        rel_y0 = int(y0) - int(base_y)
        rel_y1 = int(y1) - int(base_y)

        if (
            rel_x0 < 0
            or rel_y0 < 0
            or rel_x1 > data_hp.shape[1]
            or rel_y1 > data_hp.shape[2]
        ):
            return None

        bw = bw_full[x0:x1, y0:y1].to(torch.bool)
        disk_fp = self._disk(self.args["censor_size"], device=self.device, dtype=self.dtype)
        notbw = ~self._binary_dilate(bw.to(self.dtype), disk_fp)

        patch_view = self._read_movie_patch_view(x0, x1, y0, y1)
        if patch_view is None:
            patch = self._read_movie_patch(x0, x1, y0, y1)
            if self.args["flip_signal"]:
                patch.neg_()
            mean_im = patch.mean(dim=0)
            raw_roi_mean = patch[:, bw].mean(dim=1)
            del patch
        elif self.args["flip_signal"]:
            mean_im = -patch_view.mean(dim=0)
            raw_roi_mean = -patch_view[:, bw].mean(dim=1)
        else:
            mean_im = patch_view.mean(dim=0)
            raw_roi_mean = patch_view[:, bw].mean(dim=1)

        context_coord = torch.tensor([[x0, x1 - 1], [y0, y1 - 1]], device=self.device, dtype=torch.long)
        return (
            data_hp[:, rel_x0:rel_x1, rel_y0:rel_y1],
            raw_roi_mean,
            mean_im,
            bw,
            notbw,
            context_coord,
        )

    def _context_patch_spec(self, roi_id: int) -> _ContextPatchSpec:
        if int(roi_id) not in self._roi_id_set:
            raise KeyError(f"ROI id {roi_id} not found in roi_mask.")

        bw_full = self._roi_binary_mask(int(roi_id))
        if not torch.any(bw_full):
            raise ValueError(f"ROI id {roi_id} is empty.")
        return self._context_patch_spec_for_mask(bw_full, int(roi_id))

    def _context_patch_spec_for_mask(self, bw_full: torch.Tensor, roi_id: int) -> _ContextPatchSpec:
        xinds, yinds = torch.where(bw_full)
        if xinds.numel() == 0:
            raise ValueError(f"ROI id {roi_id} is empty.")

        context_size = int(self.args["context_size"])
        before = (context_size - 1) // 2
        after = context_size // 2
        height, width = bw_full.shape
        x0 = max(0, int(xinds.min().item()) - before)
        x1 = min(int(height), int(xinds.max().item()) + after + 1)
        y0 = max(0, int(yinds.min().item()) - before)
        y1 = min(int(width), int(yinds.max().item()) + after + 1)

        return _ContextPatchSpec(int(roi_id), x0, x1, y0, y1)

    def _context_patch_batches(
        self,
        specs: Sequence[_ContextPatchSpec],
        *,
        max_patch_bytes: int,
        max_rois_per_batch: Optional[int],
    ) -> List[List[_ContextPatchSpec]]:
        if not specs:
            return []

        batches: List[List[_ContextPatchSpec]] = []
        current: List[_ContextPatchSpec] = []
        current_box: Optional[Tuple[int, int, int, int]] = None

        for spec in sorted(specs, key=lambda item: (item.x0, item.y0, item.x1, item.y1, item.roi_id)):
            if current_box is None:
                current = [spec]
                current_box = (spec.x0, spec.x1, spec.y0, spec.y1)
                continue

            cx0, cx1, cy0, cy1 = current_box
            nx0 = min(cx0, spec.x0)
            nx1 = max(cx1, spec.x1)
            ny0 = min(cy0, spec.y0)
            ny1 = max(cy1, spec.y1)
            candidate = [*current, spec]
            too_many_rois = max_rois_per_batch is not None and len(current) >= max_rois_per_batch
            too_large_patch = self._estimate_context_batch_bytes(candidate) > max_patch_bytes

            if current and (too_many_rois or too_large_patch):
                batches.append(current)
                current = [spec]
                current_box = (spec.x0, spec.x1, spec.y0, spec.y1)
            else:
                current.append(spec)
                current_box = (nx0, nx1, ny0, ny1)

        if current:
            batches.append(current)
        return batches

    @staticmethod
    def _context_batch_bounds(batch: Sequence[_ContextPatchSpec]) -> Tuple[int, int, int, int]:
        x0 = min(spec.x0 for spec in batch)
        x1 = max(spec.x1 for spec in batch)
        y0 = min(spec.y0 for spec in batch)
        y1 = max(spec.y1 for spec in batch)
        return x0, x1, y0, y1

    def _normalize_batch_patch_bytes(self, batch_patch_bytes: Optional[int]) -> int:
        if batch_patch_bytes is not None and int(batch_patch_bytes) > 0:
            return int(batch_patch_bytes)
        return self._default_batch_patch_bytes()

    def _default_batch_patch_bytes(self) -> int:
        default = 4096 * 1024 * 1024
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return default
        try:
            with torch.cuda.device(self.device):
                free_bytes, _ = torch.cuda.mem_get_info()
            return max(64 * 1024 * 1024, min(default, int(free_bytes * 0.90)))
        except Exception:
            return default

    def _estimate_patch_bytes(self, x0: int, x1: int, y0: int, y1: int) -> int:
        return self._estimate_patch_tensor_bytes(x0, x1, y0, y1)

    def _estimate_context_batch_bytes(self, specs: Sequence[_ContextPatchSpec]) -> int:
        if not specs:
            return 0

        x0 = min(spec.x0 for spec in specs)
        x1 = max(spec.x1 for spec in specs)
        y0 = min(spec.y0 for spec in specs)
        y1 = max(spec.y1 for spec in specs)
        union_bytes = self._estimate_patch_tensor_bytes(x0, x1, y0, y1)
        prepare_peak = self._estimate_context_prepare_peak_bytes(x0, x1, y0, y1)

        max_roi_bytes = max(
            self._estimate_patch_tensor_bytes(spec.x0, spec.x1, spec.y0, spec.y1)
            for spec in specs
        )
        persistent_batch = union_bytes * 2
        active_roi_fit = self._estimate_roi_fit_peak_bytes(max_roi_bytes)
        return max(prepare_peak, persistent_batch + active_roi_fit)

    def _estimate_roi_fit_peak_bytes(self, roi_patch_bytes: int) -> int:
        roi_patch_bytes = max(0, int(roi_patch_bytes))
        if roi_patch_bytes <= 0:
            return 0

        # fit_roi keeps the prepared high-pass patch while building blurred
        # predictors, SVD/background matrices, correlation images, and filtered
        # traces. These temporaries scale with T * context pixels, so leave a
        # conservative multiplier in the batch planner instead of only counting
        # the raw ROI patch.
        return roi_patch_bytes * 8

    def _estimate_context_prepare_peak_bytes(self, x0: int, x1: int, y0: int, y1: int) -> int:
        frames = int(self.movie.shape[0])
        height = max(0, int(x1) - int(x0))
        width = max(0, int(y1) - int(y0))
        pixels = height * width
        element_size = torch.empty((), dtype=self.dtype).element_size()
        base = frames * pixels * element_size
        if base <= 0:
            return 0

        chunk_pixels = self._batch_highpass_chunk_pixels(frames, pixels, budget_bytes=self._active_batch_patch_bytes)
        chunk_base = frames * chunk_pixels * element_size
        mean_vector = chunk_pixels * element_size
        persistent = base if chunk_pixels >= pixels else base * 2
        center_peak = persistent + chunk_base + mean_vector
        filter_peak = (
            persistent
            + chunk_base
            + self._estimate_signal_filter_workspace_bytes(frames, chunk_pixels, order=3)
            + chunk_base
        )
        final_peak = persistent + chunk_base * 2 + self._estimate_signal_filter_retained_bytes(frames, chunk_pixels, order=3)
        return max(center_peak, filter_peak, final_peak)

    def _batch_highpass_chunk_pixels(
        self,
        frames: int,
        pixels: int,
        *,
        budget_bytes: Optional[int] = None,
    ) -> int:
        pixels = max(1, int(pixels))
        frames = max(0, int(frames))
        if frames <= 0 or pixels <= 1 or float(self.args["hp_freq_pb"]) <= 0.0:
            return pixels

        target_bytes = self._batch_filter_workspace_target_bytes()
        if target_bytes is None:
            return pixels

        if budget_bytes is None:
            budget_bytes = self._active_batch_patch_bytes
        retained_budget = self._retained_highpass_bytes(frames, pixels)
        if budget_bytes is not None:
            remaining = int(budget_bytes) - retained_budget
            target_bytes = min(target_bytes, max(1, remaining))

        if self._estimate_signal_filter_workspace_bytes(frames, pixels, order=3) <= target_bytes:
            return pixels

        lo = 1
        hi = pixels
        best = 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if self._estimate_signal_filter_workspace_bytes(frames, mid, order=3) <= target_bytes:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return max(1, best)

    def _batch_filter_workspace_target_bytes(self) -> Optional[int]:
        if self.batch_filter_workspace_bytes is not None:
            return int(self.batch_filter_workspace_bytes)
        if self.device.type != "cuda":
            return None
        target = 512 * 1024 * 1024
        if self._active_batch_patch_bytes is not None:
            target = min(target, max(64 * 1024 * 1024, int(self._active_batch_patch_bytes * 0.125)))
        return target

    def _retained_highpass_bytes(self, frames: int, pixels: int) -> int:
        frames = max(0, int(frames))
        pixels = max(0, int(pixels))
        element_size = torch.empty((), dtype=self.dtype).element_size()
        # patch + output data_hp are live across the batched ROI loop.
        return frames * pixels * element_size * 2

    def _estimate_signal_filter_workspace_bytes(self, frames: int, rows: int, order: int) -> int:
        frames = max(0, int(frames))
        rows = max(0, int(rows))
        if frames <= 0 or rows <= 0 or float(self.args["hp_freq_pb"]) <= 0.0:
            return 0

        element_size = self._signal_filter_work_element_size()
        ntaps = int(order) + 1
        edge = 3 * (ntaps - 1)
        length = frames + 2 * edge
        padded_buffer = rows * length * element_size

        if self.device.type != "cuda":
            return padded_buffer * 4

        n_conv = 1 << int(math.ceil(math.log2(max(1, 2 * length - 1))))
        complex_size = element_size * 2
        bytes_per_row = (
            length * element_size * 3
            + (n_conv // 2 + 1) * complex_size * 2
            + n_conv * element_size
        )
        chunk_rows = self._fft_lfilter_chunk_rows(
            rows,
            length,
            n_conv,
            self.dtype,
            self.device,
            target_bytes=self._batch_filter_workspace_target_bytes(),
        )
        chunk_workspace = int(chunk_rows) * int(bytes_per_row)
        response_cache = (n_conv // 2 + 1) * complex_size + (int(order) * length * element_size)
        # filtfilt keeps the padded input and the first-pass result while the
        # second pass builds another lfilter output. The multiplier is kept
        # conservative because CUDA FFT calls can retain temporary work buffers.
        return padded_buffer * 6 + chunk_workspace + response_cache

    def _estimate_signal_filter_retained_bytes(self, frames: int, rows: int, order: int) -> int:
        frames = max(0, int(frames))
        rows = max(0, int(rows))
        if frames <= 0 or rows <= 0 or float(self.args["hp_freq_pb"]) <= 0.0:
            return 0

        element_size = self._signal_filter_work_element_size()
        ntaps = int(order) + 1
        edge = 3 * (ntaps - 1)
        length = frames + 2 * edge
        return rows * length * element_size

    def _signal_filter_work_element_size(self) -> int:
        return torch.empty((), dtype=torch.float32).element_size()

    def _release_filter_fft_cache(self) -> None:
        self._filter_fft_cache.clear()

    def _should_flush_cuda_batch_cache(
        self,
        next_batch: Optional[Sequence[_ContextPatchSpec]],
        *,
        max_bytes: int,
    ) -> bool:
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return False
        if not self.keep_cuda_cache_between_batches:
            return True
        if next_batch is None:
            return False

        try:
            next_estimate = self._estimate_context_batch_bytes(next_batch)
            if next_estimate > int(max_bytes):
                return True

            with torch.cuda.device(self.device):
                free_bytes, _ = torch.cuda.mem_get_info()
            allocated = int(torch.cuda.memory_allocated(self.device))
            reserved = int(torch.cuda.memory_reserved(self.device))
        except Exception:
            return False

        cached = max(0, reserved - allocated)
        reusable_capacity = int(free_bytes) + cached
        capped_next_estimate = int(next_estimate / self.cuda_batch_safety_margin)
        cap_headroom = max(0, int(max_bytes) - allocated)

        return capped_next_estimate > reusable_capacity or capped_next_estimate > cap_headroom

    def _flush_cuda_batch_cache(self) -> None:
        self._release_filter_fft_cache()
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return
        try:
            torch.cuda.synchronize(self.device)
        except RuntimeError:
            pass
        torch.cuda.empty_cache()

    def _estimate_patch_tensor_bytes(self, x0: int, x1: int, y0: int, y1: int) -> int:
        frames = int(self.movie.shape[0])
        height = max(0, int(x1) - int(x0))
        width = max(0, int(y1) - int(y0))
        element_size = torch.empty((), dtype=self.dtype).element_size()
        return frames * height * width * element_size

    def _read_movie_patch(self, x0: int, x1: int, y0: int, y1: int) -> torch.Tensor:
        patch_view = self._read_movie_patch_view(x0, x1, y0, y1)
        if patch_view is not None:
            return patch_view.clone()

        return self._read_movie_patch_from_source(x0, x1, y0, y1)

    def _create_patch_prefetch_executor(self) -> Optional[ThreadPoolExecutor]:
        if not self.prefetch_next_batch_patch:
            return None
        if self._movie_cache is not None:
            return None
        if not isinstance(self.movie, Movie):
            return None
        if self.movie.transform is not None:
            return None
        return ThreadPoolExecutor(max_workers=1, thread_name_prefix="spikepursuit_patch_prefetch")

    def _submit_movie_patch_prefetch(
        self,
        executor: ThreadPoolExecutor,
        x0: int,
        x1: int,
        y0: int,
        y1: int,
    ) -> Future[torch.Tensor]:
        return executor.submit(self._read_movie_patch_cpu_from_fresh_source, x0, x1, y0, y1)

    def _read_batch_patch(
        self,
        x0: int,
        x1: int,
        y0: int,
        y1: int,
        *,
        prefetch_future: Optional[Future[torch.Tensor]],
        prefetch_bounds: Optional[Tuple[int, int, int, int]],
    ) -> torch.Tensor:
        if prefetch_future is not None and prefetch_bounds == (x0, x1, y0, y1):
            try:
                cpu_patch = prefetch_future.result()
            except Exception:
                return self._read_movie_patch_from_source(x0, x1, y0, y1)
            return self._move_cpu_patch_to_device(cpu_patch)
        return self._read_movie_patch_from_source(x0, x1, y0, y1)

    def _move_cpu_patch_to_device(self, patch: torch.Tensor) -> torch.Tensor:
        if patch.device == self.device and patch.dtype == self.dtype:
            return patch
        return patch.to(device=self.device, dtype=self.dtype, non_blocking=True)

    def _read_movie_patch_view(self, x0: int, x1: int, y0: int, y1: int) -> Optional[torch.Tensor]:
        if self._patch_read_override is not None:
            patch, base_x, base_y = self._patch_read_override
            rel_x0 = int(x0) - int(base_x)
            rel_x1 = int(x1) - int(base_x)
            rel_y0 = int(y0) - int(base_y)
            rel_y1 = int(y1) - int(base_y)
            if (
                rel_x0 >= 0
                and rel_y0 >= 0
                and rel_x1 <= patch.shape[1]
                and rel_y1 <= patch.shape[2]
            ):
                return patch[:, rel_x0:rel_x1, rel_y0:rel_y1].to(
                    device=self.device,
                    dtype=self.dtype,
                    non_blocking=True,
                )

        return None

    def _read_movie_patch_cpu_from_fresh_source(self, x0: int, x1: int, y0: int, y1: int) -> torch.Tensor:
        if not isinstance(self.movie, Movie):
            raise RuntimeError("CPU patch prefetch is only supported for HDF5 Movie inputs.")

        with Movie(
            self.movie.h5_path,
            dataset=self.movie.dataset,
            mode=self.movie.mode,
            rdcc_nbytes=self.movie.rdcc_nbytes,
        ) as reader:
            if len(self.movie.shape) == 3:
                patch = reader.read(
                    (slice(None), slice(x0, x1), slice(y0, y1)),
                    as_tensor=True,
                    dtype=None,
                    device=None,
                )
            else:
                patch = reader.read(
                    (slice(None), slice(x0, x1), slice(y0, y1), self.channel),
                    as_tensor=True,
                    dtype=None,
                    device=None,
                )
        return patch

    def _read_movie_patch_from_source(self, x0: int, x1: int, y0: int, y1: int) -> torch.Tensor:
        if self._movie_cache is not None:
            patch = self._movie_cache[:, x0:x1, y0:y1]
            if patch.device == self.device and patch.dtype == self.dtype:
                return patch.clone()
            return patch.to(device=self.device, dtype=self.dtype, non_blocking=True)

        if len(self.movie.shape) == 3:
            patch = self.movie.read(
                (slice(None), slice(x0, x1), slice(y0, y1)),
                as_tensor=True,
                dtype=None,
                device=self.device,
            )
        else:
            patch = self.movie.read(
                (slice(None), slice(x0, x1), slice(y0, y1), self.channel),
                as_tensor=True,
                dtype=None,
                device=self.device,
            )
        return patch.to(self.dtype)

    # ------------------------------------------------------------------
    # core signal processing
    # ------------------------------------------------------------------

    def denoise_spikes(
        self,
        data: torch.Tensor,
        *,
        window_length: int,
        fr: float,
        hp_freq: float = 1.0,
        clip: int = 100,
        threshold_method: str = "adaptive_threshold",
        min_spikes: int = 5,
        pnorm: float = 0.5,
        threshold: float = 2.0,
        do_plot: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool, float]:
        del do_plot  # plotting omitted in this pure-PyTorch class

        data = self._volpy_signal_filter_1d(data, hp_freq, fr, order=5, mode="high")
        data = data - self._numpy_median(data)

        locs_all = self.find_peaks(data)
        pks = data[locs_all] if locs_all.numel() > 0 else torch.empty(0, device=self.device, dtype=data.dtype)

        if threshold_method == "adaptive_threshold":
            # Original VolPy hardcodes the first adaptive-threshold pass to
            # pnorm=0.25, while the second pass below uses the configured pnorm.
            thresh, _, _, low_spikes = self.adaptive_thresh(
                pks, clip=clip, pnorm=0.25, min_spikes=min_spikes
            )
            locs = self.find_peaks(data, min_height=thresh)
        elif threshold_method == "simple":
            thresh, low_spikes = self.simple_thresh(
                data, pks, clip=clip, threshold=threshold, min_spikes=min_spikes
            )
            locs = self.find_peaks(data, min_height=thresh)
        else:
            raise ValueError("threshold_method must be 'adaptive_threshold' or 'simple'.")

        window = torch.arange(-window_length, window_length + 1, device=self.device, dtype=torch.long)
        valid = (locs > -window[0]) & (locs < (len(data) - window[-1]))
        locs = locs[valid]

        if locs.numel() == 0:
            templates = torch.zeros(window.numel(), device=self.device, dtype=data.dtype)
            datafilt = data.clone()
            spikes = torch.empty(0, device=self.device, dtype=torch.long)
            t_rec = torch.zeros_like(data)
            return datafilt, spikes, t_rec, templates, True, float(thresh)

        ptd = data[locs[:, None] + window[None, :]]
        pta = torch.quantile(ptd, 0.5, dim=0)
        pta = pta - torch.min(pta)
        templates = pta

        datafilt = self.whitened_matched_filter(data, locs, window)
        datafilt = datafilt - self._numpy_median(datafilt)

        pks2_idx = self.find_peaks(datafilt)
        pks2 = datafilt[pks2_idx] if pks2_idx.numel() > 0 else torch.empty(0, device=self.device, dtype=data.dtype)

        if threshold_method == "adaptive_threshold":
            thresh2, _, _, low_spikes = self.adaptive_thresh(
                pks2, clip=0, pnorm=pnorm, min_spikes=min_spikes
            )
            spikes = self.find_peaks(datafilt, min_height=thresh2)
        else:
            thresh2, low_spikes = self.simple_thresh(
                datafilt, pks2, clip=0, threshold=threshold, min_spikes=min_spikes
            )
            spikes = self.find_peaks(datafilt, min_height=thresh2)

        t_rec = torch.zeros_like(datafilt)
        if spikes.numel() > 0:
            t_rec[spikes] = 1.0
            t_rec = self._same_conv1d(t_rec, pta)
            denom = datafilt[spikes].mean()
            if torch.abs(denom) <= self.eps:
                factor = torch.tensor(1.0, device=self.device, dtype=datafilt.dtype)
            else:
                factor = data[spikes].mean() / denom
            datafilt = datafilt * factor
            thresh2 = float(thresh2) * float(factor.item())
        else:
            t_rec.zero_()

        return datafilt, spikes, t_rec, templates, bool(low_spikes), float(thresh2)

    def adaptive_thresh(
        self,
        pks: torch.Tensor,
        clip: int,
        pnorm: float = 0.5,
        min_spikes: int = 5,
    ) -> Tuple[float, float, float, bool]:
        if pks.numel() == 0:
            return float("inf"), 1.0, 0.0, True
        if pks.numel() == 1:
            return float(pks.item()), 0.0, 1.0, True

        pks = pks.flatten().to(device=self.device, dtype=torch.float64)
        pmin, pmax = pks.min(), pks.max()
        spread = torch.stack([pmin, pmax])
        span = (spread[1] - spread[0]).clamp_min(self.eps)
        spread = spread + span * torch.tensor([-0.05, 0.05], device=self.device, dtype=pks.dtype)
        pts = torch.linspace(spread[0], spread[1], 2001, device=self.device, dtype=pks.dtype)

        n = pks.numel()
        bw = (pks.std(unbiased=True) * (n ** (-1.0 / 5.0))).clamp_min(self.eps)

        diff = (pts[:, None] - pks[None, :]) / bw
        norm_const = bw * torch.sqrt(torch.tensor(2.0 * torch.pi, device=self.device, dtype=pks.dtype))
        f = torch.exp(-0.5 * diff.square()).mean(dim=1) / norm_const
        xi = pts

        med = torch.quantile(pks, 0.5)
        center_idx = torch.where(xi > med)[0]
        center = int(center_idx[0].item()) if center_idx.numel() > 0 else len(xi) // 2

        fmodel = torch.cat([f[: center + 1], torch.flip(f[:center], dims=[0])], dim=0)
        if len(fmodel) < len(f):
            pad = torch.full((len(f) - len(fmodel),), torch.min(fmodel), device=self.device, dtype=f.dtype)
            fmodel = torch.cat([fmodel, pad], dim=0)
        else:
            fmodel = fmodel[: len(f)]

        csf = torch.cumsum(f, dim=0) / f.sum().clamp_min(self.eps)
        csmodel = torch.cumsum(fmodel, dim=0) / torch.maximum(f.sum(), fmodel.sum()).clamp_min(self.eps)

        cross = torch.where((csf[:-1] > csmodel[:-1] + torch.finfo(csf.dtype).eps) & (csf[1:] < csmodel[1:]))[0]
        lastpt = int(cross[0].item()) if cross.numel() > 0 else center

        fmodel[: lastpt + 1] = f[: lastpt + 1]
        fmodel[lastpt:] = torch.minimum(fmodel[lastpt:], f[lastpt:])

        csf = torch.cumsum(f, dim=0)
        csmodel = torch.cumsum(fmodel, dim=0)
        csf2 = csf[-1] - csf
        csmodel2 = csmodel[-1] - csmodel
        obj = csf2.pow(pnorm) - csmodel2.pow(pnorm)
        maxind = int(torch.argmax(obj).item())
        thresh = float(xi[maxind].item())

        low_spikes = False
        n_above = int((pks > thresh).sum().item())

        if n_above < min_spikes:
            low_spikes = True
            q = max(0.0, min(1.0, 1.0 - min_spikes / max(1, len(pks))))
            thresh = float(torch.quantile(pks, q).item())
        elif clip > 0 and n_above > clip:
            q = max(0.0, min(1.0, 1.0 - clip / len(pks)))
            thresh = float(torch.quantile(pks, q).item())

        ix = int(torch.argmin(torch.abs(xi - thresh)).item())
        denom = csf2[ix].clamp_min(self.eps)
        false_pos_rate = float((csmodel2[ix] / denom).item())
        det_denom = torch.max(csf2 - csmodel2).clamp_min(self.eps)
        detection_rate = float(((csf2[ix] - csmodel2[ix]) / det_denom).item())

        return thresh, false_pos_rate, detection_rate, low_spikes

    def simple_thresh(
        self,
        data: torch.Tensor,
        pks: torch.Tensor,
        clip: int,
        threshold: float = 2.0,
        min_spikes: int = 5,
    ) -> Tuple[float, bool]:
        neg = -data[data < 0]
        if neg.numel() == 0:
            std = torch.tensor(1.0, device=self.device, dtype=data.dtype)
        else:
            std = torch.sqrt((neg.square().sum() / neg.numel()).clamp_min(self.eps))
        thresh = float((threshold * std).item())

        locs = self.find_peaks(data, min_height=thresh)
        low_spikes = False

        if locs.numel() < min_spikes and pks.numel() > 0:
            q = max(0.0, min(1.0, 1.0 - min_spikes / max(1, len(pks))))
            thresh = float(torch.quantile(pks, q).item())
            low_spikes = True
        elif clip > 0 and locs.numel() > clip and pks.numel() > 0:
            q = max(0.0, min(1.0, 1.0 - clip / len(pks)))
            thresh = float(torch.quantile(pks, q).item())

        return thresh, low_spikes

    def whitened_matched_filter(
        self,
        data: torch.Tensor,
        locs: torch.Tensor,
        window: torch.Tensor,
    ) -> torch.Tensor:
        n = int(data.numel())
        if n < 4 or locs.numel() == 0:
            return data.clone()

        win_len = int(window.numel())
        censor = torch.zeros(n, device=self.device, dtype=self.dtype)
        censor[locs] = 1.0
        censor = self._same_conv1d(censor, torch.ones(win_len, device=self.device, dtype=self.dtype))
        censor = censor < 0.5
        noise = data[censor]

        if noise.numel() < 8:
            noise = data

        nfft = 1 << int(math.ceil(math.log2(n)))
        pxx = self._welch_psd(noise, nfft=nfft)
        full_pxx = torch.cat([pxx, torch.flip(pxx[1:-1], dims=[0])], dim=0)
        scaling = 1.0 / torch.sqrt(full_pxx.clamp_min(self.eps))

        padded = F.pad(data.flatten().to(self.dtype), (0, nfft - n))
        spec = torch.fft.fft(padded)
        data_scaled = torch.fft.ifft(spec * scaling).real

        valid = (locs > -window[0]) & (locs < (n - window[-1]))
        locs = locs[valid]
        if locs.numel() == 0:
            return data_scaled[:n]

        ptd_scaled = data_scaled[locs[:, None] + window[None, :]]
        pta_scaled = ptd_scaled.mean(dim=0)
        datafilt = self._same_conv1d(data_scaled, torch.flip(pta_scaled, dims=[0]))
        return datafilt[:n]

    def _volpy_signal_filter_1d(
        self,
        sg: torch.Tensor,
        freq: float,
        fr: float,
        order: int = 3,
        mode: str = "high",
    ) -> torch.Tensor:
        return self._volpy_signal_filter_nd(
            sg.flatten(),
            freq,
            fr,
            order=order,
            mode=mode,
            dim=0,
        )

    def _volpy_signal_filter_nd(
        self,
        sg: torch.Tensor,
        freq: float,
        fr: float,
        order: int = 3,
        mode: str = "high",
        dim: int = -1,
        method_override: Optional[str] = None,
    ) -> torch.Tensor:
        if freq <= 0:
            return sg
        if sg.numel() == 0:
            return sg

        nyq = fr / 2.0
        norm_freq = float(freq) / nyq
        if not 0.0 < norm_freq < 1.0:
            raise ValueError(f"Butterworth cutoff must be between 0 and Nyquist, got freq={freq}, fr={fr}")

        method = str(method_override or self.args.get("signal_filter_method", "sos32")).lower()
        if method != "sos32":
            raise ValueError("signal_filter_method must be 'sos32'.")

        sos = self._butter_sos_coefficients(
            int(order),
            norm_freq,
            mode,
            device=sg.device,
            dtype=torch.float32,
        )
        x = torch.movedim(sg.to(torch.float32), dim, -1)
        y = self._sosfiltfilt_fft_nd(sos, x, padlen=3 * int(order))
        return torch.movedim(y, -1, dim).to(torch.float32)

    def _butter_sos_coefficients(
        self,
        order: int,
        norm_freq: float,
        mode: str,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if mode not in {"high", "low"}:
            raise ValueError("mode must be 'high' or 'low'.")
        if order <= 0:
            raise ValueError("Butterworth order must be positive.")

        key = (int(order), float(norm_freq), str(mode), str(device), dtype)
        cached = self._butter_sos_coefficients_cache.get(key)
        if cached is not None:
            return cached

        compute_device = device if device.type in {"cpu", "cuda"} else torch.device("cpu")
        z, p, k = self._buttap(int(order), device=compute_device)
        warped = 4.0 * torch.tan(torch.tensor(math.pi * float(norm_freq) / 2.0, device=compute_device, dtype=torch.float64))
        if mode == "low":
            z, p, k = self._lp2lp_zpk(z, p, k, wo=warped)
        else:
            z, p, k = self._lp2hp_zpk(z, p, k, wo=warped)

        z, p, k = self._bilinear_zpk(z, p, k, fs=torch.tensor(2.0, device=compute_device, dtype=torch.float64))
        sos = self._zpk2sos_butter(z, p, k, mode=mode, order=int(order))
        cached = sos.to(device=device, dtype=dtype)
        self._butter_sos_coefficients_cache[key] = cached
        return cached

    def _zpk2sos_butter(
        self,
        z: torch.Tensor,
        p: torch.Tensor,
        k: torch.Tensor,
        *,
        mode: str,
        order: int,
    ) -> torch.Tensor:
        del z  # Butterworth low/high-pass zeros are fixed by the bilinear transform.
        zero = torch.tensor(1.0 if mode == "high" else -1.0, device=p.device, dtype=torch.float64)
        real_mask = torch.abs(torch.imag(p)) < 1e-10
        real_poles = torch.real(p[real_mask])
        complex_poles = p[torch.imag(p) > 1e-10]
        if complex_poles.numel() > 0:
            order_idx = torch.argsort(torch.abs(complex_poles))
            complex_poles = complex_poles[order_idx]

        sections: List[torch.Tensor] = []

        def section_from_roots(z_roots: torch.Tensor, p_roots: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
            b = torch.real(gain.to(torch.complex128) * self._poly_from_roots(z_roots.to(torch.complex128))).to(torch.float64)
            a = torch.real(self._poly_from_roots(p_roots.to(torch.complex128))).to(torch.float64)
            if b.numel() < 3:
                b = F.pad(b, (0, 3 - int(b.numel())))
            if a.numel() < 3:
                a = F.pad(a, (0, 3 - int(a.numel())))
            return torch.cat([b[:3], a[:3]], dim=0)

        section_gain = k.to(torch.float64)
        if int(order) % 2 == 1:
            if real_poles.numel() == 0:
                raise RuntimeError("Odd-order Butterworth filter did not produce a real pole.")
            z_roots = zero[None]
            p_roots = real_poles[:1].to(torch.complex128)
            sections.append(section_from_roots(z_roots, p_roots, section_gain))
            section_gain = torch.tensor(1.0, device=p.device, dtype=torch.float64)

        z_pair = torch.stack([zero, zero], dim=0)
        for pole in complex_poles:
            p_pair = torch.stack([pole, torch.conj(pole)], dim=0)
            sections.append(section_from_roots(z_pair, p_pair, section_gain))
            section_gain = torch.tensor(1.0, device=p.device, dtype=torch.float64)

        if not sections:
            raise RuntimeError("Butterworth SOS conversion produced no sections.")
        return torch.stack(sections, dim=0)

    def _buttap(self, order: int, *, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if abs(int(order)) != int(order):
            raise ValueError("Filter order must be a nonnegative integer")
        z = torch.empty(0, device=device, dtype=torch.complex128)
        if int(order) == 0:
            p = torch.empty(0, device=device, dtype=torch.complex128)
        else:
            m = torch.arange(-int(order) + 1, int(order), 2, device=device, dtype=torch.float64)
            p = -torch.exp(1j * torch.pi * m / (2.0 * float(order)))
        k = torch.tensor(1.0, device=device, dtype=torch.float64)
        return z, p, k

    def _lp2lp_zpk(
        self,
        z: torch.Tensor,
        p: torch.Tensor,
        k: torch.Tensor,
        *,
        wo: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        degree = int(p.numel() - z.numel())
        z_lp = wo * z
        p_lp = wo * p
        k_lp = k * wo.pow(degree)
        return z_lp, p_lp, k_lp

    def _lp2hp_zpk(
        self,
        z: torch.Tensor,
        p: torch.Tensor,
        k: torch.Tensor,
        *,
        wo: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        degree = int(p.numel() - z.numel())
        z_hp = wo / z if z.numel() > 0 else torch.empty(0, device=z.device, dtype=z.dtype)
        p_hp = wo / p
        if degree > 0:
            z_hp = torch.cat([z_hp, torch.zeros(degree, device=z.device, dtype=z.dtype)], dim=0)
        if z.numel() == 0:
            z_prod = torch.tensor(1.0, device=z.device, dtype=torch.float64)
        else:
            z_prod = torch.prod(-z)
        p_prod = torch.prod(-p)
        k_hp = k * torch.real(z_prod / p_prod)
        return z_hp, p_hp, k_hp

    def _bilinear_zpk(
        self,
        z: torch.Tensor,
        p: torch.Tensor,
        k: torch.Tensor,
        *,
        fs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        degree = int(p.numel() - z.numel())
        fs2 = 2.0 * fs
        z_z = (fs2 + z) / (fs2 - z) if z.numel() > 0 else torch.empty(0, device=z.device, dtype=z.dtype)
        p_z = (fs2 + p) / (fs2 - p)
        if degree > 0:
            z_z = torch.cat([z_z, -torch.ones(degree, device=z.device, dtype=z.dtype)], dim=0)
        if z.numel() == 0:
            z_prod = torch.tensor(1.0, device=z.device, dtype=torch.complex128)
        else:
            z_prod = torch.prod(fs2 - z)
        p_prod = torch.prod(fs2 - p)
        k_z = k * torch.real(z_prod / p_prod)
        return z_z, p_z, k_z

    def _poly_from_roots(self, roots: torch.Tensor) -> torch.Tensor:
        roots = torch.atleast_1d(roots)
        if roots.numel() == 0:
            return torch.ones(1, device=roots.device, dtype=roots.dtype)

        coeffs = torch.ones(1, device=roots.device, dtype=roots.dtype)
        for root in roots:
            coeffs = F.pad(coeffs, (0, 1)) - root * F.pad(coeffs, (1, 0))
        return coeffs

    def _sosfiltfilt_fft_nd(self, sos: torch.Tensor, x: torch.Tensor, *, padlen: int) -> torch.Tensor:
        edge = int(padlen)
        if x.shape[-1] <= edge:
            raise ValueError(
                f"The length of the input vector x must be greater than padlen, which is {edge}."
            )

        left_ext = 2.0 * x[..., :1] - torch.flip(x[..., 1 : edge + 1], dims=[-1])
        right_ext = 2.0 * x[..., -1:] - torch.flip(x[..., -edge - 1 : -1], dims=[-1])
        ext = torch.cat([left_ext, x, right_ext], dim=-1)

        zi = self._sosfilt_zi(sos).to(device=x.device, dtype=x.dtype)
        y = self._sosfilt_fft_nd(sos, ext, zi * ext[..., :1].unsqueeze(-2))
        y = self._sosfilt_fft_nd(sos, torch.flip(y, dims=[-1]), zi * y[..., -1:].unsqueeze(-2))
        y = torch.flip(y, dims=[-1])
        return y[..., edge:-edge]

    def _sosfilt_fft_nd(
        self,
        sos: torch.Tensor,
        x: torch.Tensor,
        zi: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        y = x
        for section_index in range(int(sos.shape[0])):
            section = sos[section_index]
            section_zi = None if zi is None else zi[..., section_index, :]
            y = self._lfilter_fft_nd(section[:3], section[3:], y, section_zi)
        return y

    def _sosfilt_zi(self, sos: torch.Tensor) -> torch.Tensor:
        zi: List[torch.Tensor] = []
        scale = torch.ones((), device=sos.device, dtype=sos.dtype)
        for section_index in range(int(sos.shape[0])):
            section = sos[section_index]
            b = section[:3]
            a = section[3:]
            zi.append(self._lfilter_zi(b, a) * scale)
            scale = scale * b.sum() / a.sum()
        return torch.stack(zi, dim=0)

    def _lfilter_fft_nd(
        self,
        b: torch.Tensor,
        a: torch.Tensor,
        x: torch.Tensor,
        zi: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        length = int(x.shape[-1])
        n_conv = 1 << int(math.ceil(math.log2(max(1, 2 * length - 1))))
        h_fft, init_response = self._lfilter_fft_response_tensors(
            b, a, length=length, n_conv=n_conv, device=x.device, dtype=x.dtype
        )

        flat = x.contiguous().reshape(-1, length)
        out = torch.empty_like(flat)

        zi_flat = None
        if zi is not None and init_response.numel() > 0:
            order = int(init_response.shape[0])
            z = zi.to(device=x.device, dtype=x.dtype)
            target_shape = (*x.shape[:-1], order)
            if z.shape != target_shape:
                z = torch.broadcast_to(z, target_shape)
            zi_flat = z.contiguous().reshape(-1, order)

        chunk_rows = self._fft_lfilter_chunk_rows(
            flat.shape[0],
            length,
            n_conv,
            x.dtype,
            x.device,
            target_bytes=self._batch_filter_workspace_target_bytes(),
        )
        for start in range(0, flat.shape[0], chunk_rows):
            stop = min(start + chunk_rows, flat.shape[0])
            block = flat[start:stop]
            spec = torch.fft.rfft(block, n=n_conv, dim=-1)
            filtered = torch.fft.irfft(spec * h_fft, n=n_conv, dim=-1)[..., :length]
            if zi_flat is not None:
                filtered = filtered + zi_flat[start:stop] @ init_response
            out[start:stop] = filtered

        return out.reshape_as(x)

    def _lfilter_fft_response_tensors(
        self,
        b: torch.Tensor,
        a: torch.Tensor,
        *,
        length: int,
        n_conv: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (
            str(device),
            str(dtype),
            int(length),
            int(n_conv),
            int(b.data_ptr()),
            int(a.data_ptr()),
            int(b.numel()),
            int(a.numel()),
        )
        cached = self._filter_fft_cache.get(key)
        if cached is not None:
            return cached

        impulse, init_response = self._lfilter_responses_fast(
            b.to(device=device, dtype=torch.float64),
            a.to(device=device, dtype=torch.float64),
            int(length),
        )
        h_fft = torch.fft.rfft(impulse.to(dtype=dtype), n=n_conv)
        init_t = init_response.to(dtype=dtype)
        cached = (h_fft, init_t)
        self._filter_fft_cache[key] = cached
        return cached

    def _lfilter_responses_fast(
        self,
        b: torch.Tensor,
        a: torch.Tensor,
        length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n = max(int(a.numel()), int(b.numel()))
        if n <= 3:
            return self._lfilter_responses_closed_form(b, a, length)
        return self._lfilter_responses_torch(b, a, length)

    def _lfilter_responses_closed_form(
        self,
        b: torch.Tensor,
        a: torch.Tensor,
        length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n = max(int(a.numel()), int(b.numel()))
        if a.numel() < n:
            a = F.pad(a, (0, n - int(a.numel())))
        if b.numel() < n:
            b = F.pad(b, (0, n - int(b.numel())))
        if not torch.isclose(a[0], torch.ones((), device=a.device, dtype=a.dtype)):
            b = b / a[0]
            a = a / a[0]

        order = n - 1
        if order == 0:
            impulse = torch.zeros(length, device=b.device, dtype=b.dtype)
            impulse[0] = b[0]
            return impulse, torch.empty((0, length), device=b.device, dtype=b.dtype)

        system = torch.zeros((order, order), device=b.device, dtype=torch.complex128)
        system[0, :] = (-a[1:]).to(torch.complex128)
        if order > 1:
            system[1:, :-1] = torch.eye(order - 1, device=b.device, dtype=torch.complex128)
        state_input = (b[1:] - a[1:] * b[0]).to(torch.complex128)

        evals, evecs = torch.linalg.eig(system)
        inv_evecs = torch.linalg.inv(evecs)
        basis = inv_evecs[:, 0]
        powers = evals[None, :] ** torch.arange(length, device=b.device, dtype=torch.float64)[:, None]
        powers[0, :] = torch.ones_like(evals)
        weighted_powers = powers * basis[None, :]
        init_response = torch.real(weighted_powers @ evecs.transpose(0, 1)).transpose(0, 1).to(b.dtype)

        impulse = torch.empty(length, device=b.device, dtype=b.dtype)
        impulse[0] = b[0]
        if length > 1:
            impulse[1:] = torch.real((powers[:-1] * (state_input @ evecs * basis)[None, :]).sum(dim=1)).to(b.dtype)
        return impulse, init_response

    def _lfilter_responses_torch(
        self,
        b: torch.Tensor,
        a: torch.Tensor,
        length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n = max(int(a.numel()), int(b.numel()))
        if a.numel() < n:
            a = F.pad(a, (0, n - int(a.numel())))
        if b.numel() < n:
            b = F.pad(b, (0, n - int(b.numel())))
        if not torch.isclose(a[0], torch.ones((), device=a.device, dtype=a.dtype)):
            b = b / a[0]
            a = a / a[0]

        order = n - 1
        if order == 0:
            impulse = torch.zeros(length, device=b.device, dtype=b.dtype)
            impulse[0] = b[0]
            return impulse, torch.empty((0, length), device=b.device, dtype=b.dtype)

        batch = order + 1
        x = torch.zeros((batch, length), device=b.device, dtype=b.dtype)
        x[0, 0] = 1.0
        z = torch.zeros((batch, order), device=b.device, dtype=b.dtype)
        if order > 0:
            z[1:, :] = torch.eye(order, device=b.device, dtype=b.dtype)

        y = torch.empty((batch, length), device=b.device, dtype=b.dtype)
        for idx in range(length):
            sample = x[:, idx]
            out = b[0] * sample + z[:, 0]
            y[:, idx] = out
            if order > 1:
                new_z = torch.empty_like(z)
                new_z[:, :-1] = z[:, 1:] + b[1:-1] * sample[:, None] - a[1:-1] * out[:, None]
                new_z[:, -1] = b[-1] * sample - a[-1] * out
                z = new_z
            else:
                z[:, 0] = b[-1] * sample - a[-1] * out

        return y[0], y[1:]

    def _fft_lfilter_chunk_rows(
        self,
        rows: int,
        length: int,
        n_conv: int,
        dtype: torch.dtype,
        device: torch.device,
        *,
        target_bytes: Optional[int] = None,
    ) -> int:
        if rows <= 0:
            return 1
        elem_size = torch.empty((), dtype=dtype).element_size()
        complex_size = elem_size * 2
        bytes_per_row = (
            length * elem_size * 3
            + (n_conv // 2 + 1) * complex_size * 2
            + n_conv * elem_size
        )
        if target_bytes is not None and int(target_bytes) > 0:
            target_bytes = int(target_bytes)
        elif device.type == "cuda":
            try:
                free_bytes, _ = torch.cuda.mem_get_info(device)
                target_bytes = min(int(free_bytes * 0.20), 512 * 1024 * 1024)
            except RuntimeError:
                target_bytes = 256 * 1024 * 1024
        else:
            target_bytes = 256 * 1024 * 1024
        min_rows = 1 if target_bytes < bytes_per_row * 16 else 16
        target_rows = max(min_rows, max(1, target_bytes // max(1, bytes_per_row)))
        return max(1, min(rows, target_rows))

    def _lfilter_zi(self, b: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        device = a.device
        dtype = a.dtype
        n = max(int(a.numel()), int(b.numel()))
        if a.numel() < n:
            a = F.pad(a, (0, n - int(a.numel())))
        if b.numel() < n:
            b = F.pad(b, (0, n - int(b.numel())))
        if not torch.isclose(a[0], torch.ones((), device=device, dtype=dtype)):
            b = b / a[0]
            a = a / a[0]

        order = n - 1
        if order == 0:
            return torch.empty(0, device=device, dtype=dtype)

        companion = torch.zeros((order, order), device=device, dtype=dtype)
        companion[0, :] = -a[1:]
        if order > 1:
            companion[1:, :-1] = torch.eye(order - 1, device=device, dtype=dtype)
        system = torch.eye(order, device=device, dtype=dtype) - companion.transpose(0, 1)
        rhs = b[1:] - a[1:] * b[0]
        return torch.linalg.solve(system, rhs)

    def signal_filter(
        self,
        sg: torch.Tensor,
        freq: float,
        fr: float,
        order: int = 3,
        mode: str = "high",
        dim: int = 0,
    ) -> torch.Tensor:
        return self._volpy_signal_filter_nd(sg, freq, fr, order=order, mode=mode, dim=dim)

    def find_peaks(self, x: torch.Tensor, min_height: Optional[float] = None) -> torch.Tensor:
        x = x.flatten()
        if x.numel() < 3:
            return torch.empty(0, device=self.device, dtype=torch.long)

        left = x[1:-1] > x[:-2]
        right = x[1:-1] >= x[2:]
        peaks = left & right
        idx = torch.where(peaks)[0] + 1

        if min_height is not None:
            idx = idx[x[idx] >= min_height]

        return idx.to(torch.long)

    # ------------------------------------------------------------------
    # linear algebra + kernels
    # ------------------------------------------------------------------

    def _background_components(self, x: torch.Tensor, k: int) -> torch.Tensor:
        q = min(k, min(x.shape) - 1)
        if q <= 0:
            raise ValueError("Cannot compute background PCs: matrix too small.")
        x = x.to(self.dtype)
        frames, pixels = x.shape

        if frames >= pixels:
            gram = x.t() @ x
            evals, evecs = torch.linalg.eigh(gram)
            order = torch.argsort(evals, descending=True)[:q]
            vals = evals[order].clamp_min(self.eps)
            vecs = evecs[:, order]
            u = x @ vecs
            u = u / torch.sqrt(vals)[None, :]
        else:
            gram = x @ x.t()
            evals, evecs = torch.linalg.eigh(gram)
            order = torch.argsort(evals, descending=True)[:q]
            u = evecs[:, order]

        u, _ = torch.linalg.qr(u, mode="reduced")
        return u[:, :q]

    def _ridge_fit(self, X: torch.Tensor, y: torch.Tensor, alpha: float) -> torch.Tensor:
        X = X.to(self.dtype)
        y = y.to(self.dtype)
        XtX = X.t() @ X
        n = XtX.shape[0]
        reg = float(alpha) * torch.eye(n, device=self.device, dtype=self.dtype)
        Xty = X.t() @ y
        return torch.linalg.solve(XtX + reg, Xty)

    def _ridge_fit_intercept(self, X: torch.Tensor, y: torch.Tensor, alpha: float) -> torch.Tensor:
        return self._ridge_solver_fit_intercept(X, alpha=alpha)(y)

    def _ridge_solver(self, X: torch.Tensor, alpha: float) -> Callable[[torch.Tensor], torch.Tensor]:
        Xf = X.to(torch.float64)
        sqrt_alpha = float(math.sqrt(float(alpha)))
        device = X.device
        dtype = X.dtype

        def solve(y: torch.Tensor) -> torch.Tensor:
            coef = self._lsqr_dense(
                Xf,
                y.to(torch.float64),
                damp=sqrt_alpha,
                atol=1e-4,
                btol=1e-4,
            )
            return coef.to(device=device, dtype=dtype)

        return solve

    def _ridge_solver_fit_intercept(self, X: torch.Tensor, alpha: float) -> Callable[[torch.Tensor], torch.Tensor]:
        Xf = X.to(torch.float64)
        x_mean = Xf.mean(dim=0)
        X_centered = Xf - x_mean
        sqrt_alpha = float(math.sqrt(float(alpha)))
        device = X.device
        dtype = X.dtype

        def solve(y: torch.Tensor) -> torch.Tensor:
            yf = y.to(torch.float64)
            y_mean = yf.mean()
            coef = self._lsqr_dense(
                X_centered,
                yf - y_mean,
                damp=sqrt_alpha,
                atol=1e-4,
                btol=1e-4,
            )
            weights = coef.clone()
            weights[0] = y_mean - x_mean @ coef
            return weights.to(device=device, dtype=dtype)

        return solve

    def _lsqr_dense(
        self,
        A: torch.Tensor,
        b: torch.Tensor,
        *,
        damp: float = 0.0,
        atol: float = 1e-6,
        btol: float = 1e-6,
        conlim: float = 1e8,
        iter_lim: Optional[int] = None,
        x0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        A = A.to(torch.float64)
        b = torch.atleast_1d(b.to(torch.float64))
        if b.ndim > 1:
            b = b.squeeze()

        m, n = A.shape
        if iter_lim is None:
            iter_lim = 2 * n

        eps = torch.finfo(torch.float64).eps
        dampsq = float(damp) ** 2
        ctol = 0.0 if conlim <= 0 else 1.0 / float(conlim)

        def sym_ortho(a_val: float, b_val: float) -> Tuple[float, float, float]:
            if b_val == 0:
                return float(math.copysign(1.0, a_val)) if a_val != 0 else 0.0, 0.0, abs(a_val)
            if a_val == 0:
                return 0.0, float(math.copysign(1.0, b_val)), abs(b_val)
            if abs(b_val) > abs(a_val):
                tau = a_val / b_val
                s = math.copysign(1.0, b_val) / math.sqrt(1.0 + tau * tau)
                c = s * tau
                r = b_val / s
            else:
                tau = b_val / a_val
                c = math.copysign(1.0, a_val) / math.sqrt(1.0 + tau * tau)
                s = c * tau
                r = a_val / c
            return c, s, r

        u = b.clone()
        bnorm = float(torch.linalg.vector_norm(b).item())
        if x0 is None:
            x = torch.zeros(n, device=A.device, dtype=torch.float64)
            beta = bnorm
        else:
            x = x0.to(device=A.device, dtype=torch.float64).clone()
            u = u - A @ x
            beta = float(torch.linalg.vector_norm(u).item())

        if beta > 0:
            u = u / beta
            v = A.transpose(0, 1) @ u
            alfa = float(torch.linalg.vector_norm(v).item())
        else:
            v = x.clone()
            alfa = 0.0

        if alfa > 0:
            v = v / alfa
        w = v.clone()

        rhobar = alfa
        phibar = beta
        rnorm = beta
        r1norm = rnorm
        r2norm = rnorm
        anorm = 0.0
        acond = 0.0
        ddnorm = 0.0
        res2 = 0.0
        xnorm = 0.0
        xxnorm = 0.0
        z = 0.0
        cs2 = -1.0
        sn2 = 0.0
        arnorm = alfa * beta
        istop = 0
        if arnorm == 0:
            return x

        for itn in range(1, int(iter_lim) + 1):
            u = A @ v - alfa * u
            beta = float(torch.linalg.vector_norm(u).item())
            if beta > 0:
                u = u / beta
                anorm = math.sqrt(anorm * anorm + alfa * alfa + beta * beta + dampsq)
                v = A.transpose(0, 1) @ u - beta * v
                alfa = float(torch.linalg.vector_norm(v).item())
                if alfa > 0:
                    v = v / alfa

            if damp > 0:
                rhobar1 = math.sqrt(rhobar * rhobar + dampsq)
                cs1 = rhobar / rhobar1
                sn1 = float(damp) / rhobar1
                psi = sn1 * phibar
                phibar = cs1 * phibar
            else:
                rhobar1 = rhobar
                psi = 0.0

            cs, sn, rho = sym_ortho(rhobar1, beta)
            theta = sn * alfa
            rhobar = -cs * alfa
            phi = cs * phibar
            phibar = sn * phibar
            tau = sn * phi

            t1 = phi / rho
            t2 = -theta / rho
            dk = w / rho
            x = x + t1 * w
            w = v + t2 * w
            ddnorm = ddnorm + float(torch.linalg.vector_norm(dk).item()) ** 2

            delta = sn2 * rho
            gambar = -cs2 * rho
            rhs = phi - delta * z
            zbar = rhs / gambar
            xnorm = math.sqrt(xxnorm + zbar * zbar)
            gamma = math.sqrt(gambar * gambar + theta * theta)
            cs2 = gambar / gamma
            sn2 = theta / gamma
            z = rhs / gamma
            xxnorm = xxnorm + z * z

            acond = anorm * math.sqrt(ddnorm)
            res1 = phibar * phibar
            res2 = res2 + psi * psi
            rnorm = math.sqrt(res1 + res2)
            arnorm = alfa * abs(tau)

            if damp > 0:
                r1sq = rnorm * rnorm - dampsq * xxnorm
                r1norm = math.sqrt(abs(r1sq))
                if r1sq < 0:
                    r1norm = -r1norm
            else:
                r1norm = rnorm
            r2norm = rnorm

            test1 = rnorm / (bnorm + eps)
            test2 = arnorm / (anorm * rnorm + eps)
            test3 = 1.0 / (acond + eps)
            t1_test = test1 / (1.0 + anorm * xnorm / (bnorm + eps))
            rtol = btol + atol * anorm * xnorm / (bnorm + eps)

            if itn >= iter_lim:
                istop = 7
            if 1.0 + test3 <= 1.0:
                istop = 6
            if 1.0 + test2 <= 1.0:
                istop = 5
            if 1.0 + t1_test <= 1.0:
                istop = 4
            if test3 <= ctol:
                istop = 3
            if test2 <= atol:
                istop = 2
            if test1 <= rtol:
                istop = 1
            if istop != 0:
                break

        return x

    def _build_predictor(
        self,
        movie_patch: torch.Tensor,
        sigma: float,
        kernel_size: Optional[int] = None,
    ) -> torch.Tensor:
        blurred = self._gaussian_blur_2d(movie_patch, sigma=sigma, kernel_size=kernel_size)
        T = movie_patch.shape[0]
        flat = blurred.reshape(T, -1)
        ones = torch.ones((T, 1), device=flat.device, dtype=flat.dtype)
        return torch.cat([ones, flat], dim=1)

    def _gaussian_blur_2d(
        self,
        x: torch.Tensor,
        sigma: float,
        kernel_size: Optional[int] = None,
    ) -> torch.Tensor:
        if sigma <= 0:
            return x
        if kernel_size is None:
            kernel_size = int(2 * torch.ceil(torch.tensor(2 * sigma)).item() + 1)
        kernel = self._gaussian_kernel2d(kernel_size, sigma, device=x.device, dtype=x.dtype)
        pad = kernel_size // 2
        padded = F.pad(x[:, None, :, :], (pad, pad, pad, pad), mode="replicate")
        y = F.conv2d(padded, kernel[None, None, :, :], padding=0)
        return y[:, 0]

    def _gaussian_kernel2d(
        self,
        kernel_size: int,
        sigma: float,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if kernel_size % 2 == 0:
            kernel_size += 1
        key = (int(kernel_size), float(sigma), str(device), dtype)
        cached = self._gaussian_kernel2d_cache.get(key)
        if cached is not None:
            return cached

        ax = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2.0
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx.square() + yy.square()) / (2 * sigma * sigma))
        kernel = kernel / kernel.sum().clamp_min(self.eps)
        self._gaussian_kernel2d_cache[key] = kernel
        return kernel

    def _disk(
        self,
        radius: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        radius = int(max(0, radius))
        key = (-(radius + 1), str(device), dtype)
        cached = self._disk_cache.get(key)
        if cached is not None:
            return cached

        size = 2 * radius + 1
        ax = torch.arange(size, device=device, dtype=dtype) - radius
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        footprint = ((xx.square() + yy.square()) <= (radius * radius)).to(dtype)
        self._disk_cache[key] = footprint
        return footprint

    def _binary_dilate(self, mask: torch.Tensor, footprint: torch.Tensor) -> torch.Tensor:
        fp = footprint
        if fp.ndim != 2:
            raise ValueError("footprint must be 2D.")
        if fp.shape[0] % 2 == 0:
            fp = torch.cat([fp, torch.zeros(1, fp.shape[1], device=fp.device, dtype=fp.dtype)], dim=0)
        if fp.shape[1] % 2 == 0:
            fp = torch.cat([fp, torch.zeros(fp.shape[0], 1, device=fp.device, dtype=fp.dtype)], dim=1)

        pad_y = fp.shape[0] // 2
        pad_x = fp.shape[1] // 2
        x = mask.to(self.dtype)[None, None]
        y = F.conv2d(x, fp[None, None], padding=(pad_y, pad_x))
        return y[0, 0] > 0

    def _same_conv1d(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        x = x.flatten().to(self.dtype)
        kernel = kernel.flatten().to(self.dtype)
        pad = kernel.numel() // 2
        y = F.conv1d(x[None, None], torch.flip(kernel, dims=[0])[None, None], padding=pad)
        return y[0, 0, : x.numel()]

    def _numpy_median(self, x: torch.Tensor) -> torch.Tensor:
        return torch.quantile(x.flatten(), 0.5)

    def _welch_psd(self, x: torch.Tensor, nfft: int) -> torch.Tensor:
        x = x.flatten().to(self.dtype)
        n = x.numel()
        if n < 8:
            spec = torch.fft.rfft(x, n=nfft)
            return (spec.abs() ** 2).clamp_min(self.eps)

        nperseg = min(1000, n)
        if nperseg < 8:
            nperseg = n
        step = max(1, nperseg // 2)

        if n < nperseg:
            pad = nperseg - n
            xpad = F.pad(x, (0, pad))
            segs = xpad[None, :]
        else:
            segs = x.unfold(0, nperseg, step)

        win = torch.hamming_window(nperseg, periodic=True, device=self.device, dtype=self.dtype)
        segs = segs * win
        spec = torch.fft.rfft(segs, n=nfft, dim=-1)
        pxx = (spec.abs() ** 2).mean(dim=0)
        scale = 1.0 / ((2.0 * math.pi) * win.square().sum().clamp_min(self.eps))
        pxx = pxx * scale
        if nfft % 2 == 0:
            pxx[1:-1] *= 2.0
        else:
            pxx[1:] *= 2.0
        return pxx.clamp_min(self.eps)

    def _compute_snr(self, t: torch.Tensor, spikes: torch.Tensor) -> float:
        if spikes.numel() == 0:
            return 0.0
        t0 = t - self._numpy_median(t)
        sgn = t0[spikes].mean()
        neg = -t0[t0 < 0]
        if neg.numel() == 0:
            return 0.0
        noise = torch.sqrt((neg.square().sum() / neg.numel()).clamp_min(self.eps))
        return float((sgn / noise.clamp_min(self.eps)).item())
