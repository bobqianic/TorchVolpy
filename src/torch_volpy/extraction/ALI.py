import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class ALIResult:
    """Container for intermediate and final outputs from the ALI pipeline."""
    data: torch.Tensor
    df: torch.Tensor
    f0: torch.Tensor
    spk: torch.Tensor
    df_ap: torch.Tensor
    df_ap_denoised: torch.Tensor
    ucomps: torch.Tensor
    vcomps: torch.Tensor
    spk_fine: torch.Tensor
    brightness: torch.Tensor
    roi: torch.Tensor
    cnt: torch.Tensor
    cen: Tuple[torch.Tensor, torch.Tensor]
    alimap: torch.Tensor
    peaks: torch.Tensor
    clust_cen: torch.Tensor
    clust_idx: torch.Tensor
    sort_idx: torch.Tensor
    footprint: torch.Tensor
    fp: torch.Tensor
    support: torch.Tensor
    traces_ls: torch.Tensor
    traces: torch.Tensor


class ALI:
    """
    Activity Localization Imaging (ALI) pipeline in pure PyTorch.

    Device policy:
        - If `device` is passed at init, the whole pipeline runs there.
        - Otherwise, tensors run on the input tensor's device.
        - Non-tensor inputs default to CPU.
        - Outputs stay on the working device.

    Input:
        data: array-like or torch.Tensor of shape [H, W, T]

    Output dict keys:
        data, df, f0, spk, df_ap, df_ap_denoised, ucomps, vcomps, spk_fine,
        brightness, roi, cnt, cen, alimap, peaks, clust_cen,
        clust_idx, sort_idx, footprint, fp, support, traces_ls, traces
    """

    def __init__(
        self,
        fs: int = 2000,
        hp_window_ms: float = 10.0,
        nsvd: int = 25,
        factor: int = 4,
        coarse_sigma: float = 1.8,
        coarse_gaussian_radius: int = 2,
        coarse_threshold_std: float = 5.0,
        min_component_size: int = 4,
        fine_npix: int = 15,
        fine_radius: float = 4.0,
        cluster_threshold: float = 2.0,
        peak_kernel_size: int = 3,
        assign_radius: float = 1.5,
        alimap_sigma: float = 0.7,
        alimap_gaussian_radius: int = 2,
        footprint_radius: float = 10.0,
        solve_eps: float = 1e-6,
        device: Optional[str] = None,
        cc_max_iter: int = 2048,
        verbose: bool = False,
    ) -> None:
        self.fs = fs
        self.hp_window_ms = hp_window_ms
        self.nsvd = nsvd
        self.factor = factor

        self.coarse_sigma = coarse_sigma
        self.coarse_gaussian_radius = coarse_gaussian_radius
        self.coarse_threshold_std = coarse_threshold_std
        self.min_component_size = min_component_size

        self.fine_npix = fine_npix
        self.fine_radius = fine_radius

        self.cluster_threshold = cluster_threshold
        self.peak_kernel_size = peak_kernel_size
        self.assign_radius = assign_radius

        self.alimap_sigma = alimap_sigma
        self.alimap_gaussian_radius = alimap_gaussian_radius
        self.footprint_radius = footprint_radius
        self.solve_eps = solve_eps

        self.device_override = None if device is None else torch.device(device)
        self.cc_max_iter = cc_max_iter
        self.verbose = verbose

        if self.device_override is not None:
            if self.device_override.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError(f"device={self.device_override} requested, but CUDA is not available.")

    def __call__(self, data: Any) -> ALIResult:
        """Run the ALI pipeline on an input movie crop."""
        return self.forward(data)

    def to(self, device: str) -> "ALI":
        """Set the default torch device used by subsequent ALI runs."""
        self.device_override = torch.device(device)
        if self.device_override.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"device={self.device_override} requested, but CUDA is not available.")
        return self

    def _resolve_device(self, x: Any) -> torch.device:
        if self.device_override is not None:
            return self.device_override
        if torch.is_tensor(x):
            return x.device
        return torch.device("cpu")

    def _as_tensor(self, x: Any, device: torch.device) -> torch.Tensor:
        if torch.is_tensor(x):
            return x.to(device=device, dtype=torch.float32)
        return torch.as_tensor(x, dtype=torch.float32, device=device)

    @staticmethod
    def _ensure_odd(k: int) -> int:
        return k if (k % 2 == 1) else k + 1

    def _hp_window_samples(self, T: int) -> int:
        win = int(round(self.fs * self.hp_window_ms / 1000.0))
        win = max(1, self._ensure_odd(win))
        if win > T:
            win = T if T % 2 == 1 else max(1, T - 1)
        return max(1, win)

    @staticmethod
    def _gaussian_kernel2d(
        sigma: float,
        radius: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        ax = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel

    def _gaussian_blur_frames(
        self,
        stack: torch.Tensor,
        sigma: float,
        radius: int,
    ) -> torch.Tensor:
        """
        stack: [H, W, T]
        returns: [H, W, T]
        """
        kernel = self._gaussian_kernel2d(
            sigma=sigma,
            radius=radius,
            device=stack.device,
            dtype=stack.dtype,
        ).view(1, 1, 2 * radius + 1, 2 * radius + 1)

        x = stack.permute(2, 0, 1).unsqueeze(1)  # [T, 1, H, W]
        x = F.conv2d(x, kernel, padding=radius)
        return x.squeeze(1).permute(1, 2, 0).contiguous()

    def _gaussian_blur_image(
        self,
        image: torch.Tensor,
        sigma: float,
        radius: int,
    ) -> torch.Tensor:
        """
        image: [H, W]
        returns: [H, W]
        """
        kernel = self._gaussian_kernel2d(
            sigma=sigma,
            radius=radius,
            device=image.device,
            dtype=image.dtype,
        ).view(1, 1, 2 * radius + 1, 2 * radius + 1)

        x = image.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        x = F.conv2d(x, kernel, padding=radius)
        return x[0, 0]

    def hp_filter(self, stack: torch.Tensor) -> torch.Tensor:
        """
        Median high-pass filter along time for each pixel trace.

        stack: [H, W, T]
        returns: [H, W, T]
        """
        H, W, T = stack.shape
        win = self._hp_window_samples(T)

        if win == 1 or T == 1:
            baseline = stack.median(dim=2, keepdim=True).values
            return stack - baseline

        x = stack.reshape(H * W, 1, T)  # [Npix,1,T]
        pad = win // 2

        x_pad = F.pad(x, (pad, pad), mode="reflect")
        windows = x_pad.unfold(dimension=-1, size=win, step=1)  # [Npix,1,T,win]
        baseline = windows.median(dim=-1).values  # [Npix,1,T]

        df = (x - baseline).reshape(H, W, T)
        return df

    def _connected_components_3d_torch(self, bw: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        26-connected components for [H, W, T] bool tensor using iterative max-pooling.
        Works on CPU and CUDA.

        Returns:
            labeled: [H, W, T] long tensor
            num_features: int
        """
        if bw.ndim != 3 or bw.dtype != torch.bool:
            raise ValueError("bw must be a bool tensor of shape [H, W, T]")

        if not bool(bw.any().item()):
            return torch.zeros_like(bw, dtype=torch.long), 0

        labels = torch.arange(
            1,
            bw.numel() + 1,
            device=bw.device,
            dtype=torch.float64,
        ).reshape_as(bw)

        mask = bw.to(torch.float64)
        x = (labels * mask).unsqueeze(0).unsqueeze(0)  # [1,1,H,W,T]
        mask5 = mask.unsqueeze(0).unsqueeze(0)

        for _ in range(self.cc_max_iter):
            x_next = F.max_pool3d(x, kernel_size=3, stride=1, padding=1)
            x_next = x_next * mask5
            if torch.equal(x_next, x):
                break
            x = x_next
        else:
            raise RuntimeError(
                f"3D connected-components did not converge in {self.cc_max_iter} iterations. "
                "Increase cc_max_iter for large or elongated components."
            )

        labeled = x[0, 0].to(torch.long)
        num_features = int(torch.unique(labeled[labeled > 0]).numel())
        return labeled, num_features

    def spk_coarse(self, df_pos: torch.Tensor) -> torch.Tensor:
        """
        Detect coarse spike locations from positive-going events.

        df_pos: [H, W, T]
        returns:
            spk: [N, 3] int64, columns are [row, col, frame]
        """
        df_lp = self._gaussian_blur_frames(
            df_pos,
            sigma=self.coarse_sigma,
            radius=self.coarse_gaussian_radius,
        )

        sd = df_lp.std(dim=2)
        bw = df_lp > (self.coarse_threshold_std * sd.unsqueeze(-1))

        labeled, num_features = self._connected_components_3d_torch(bw)

        if num_features == 0:
            return torch.empty((0, 3), dtype=torch.long, device=df_pos.device)

        seg_ids, counts = torch.unique(
            labeled[labeled > 0],
            sorted=True,
            return_counts=True,
        )
        keep_ids = seg_ids[counts > self.min_component_size]

        if keep_ids.numel() == 0:
            return torch.empty((0, 3), dtype=torch.long, device=df_pos.device)

        spk_list = []
        for seg_id in keep_ids:
            coords = torch.nonzero(labeled == seg_id, as_tuple=False)  # [K, 3]
            vals = df_lp[coords[:, 0], coords[:, 1], coords[:, 2]]
            max_idx = torch.argmax(vals)
            spk_list.append(coords[max_idx])

        return torch.stack(spk_list, dim=0).to(dtype=torch.long)

    def denoising(
        self,
        df_ap: torch.Tensor,
        nsvd: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Truncated SVD denoising.

        df_ap: [H, W, N]
        returns:
            df_denoised: [H, W, N]
            ucomps: [H, W, K]
            vcomps: [N, K]
        """
        H, W, N = df_ap.shape
        x = df_ap.reshape(H * W, N)

        k = min(nsvd or self.nsvd, min(x.shape))
        if k < 1:
            raise ValueError("nsvd must be >= 1")

        U, S, Vh = torch.linalg.svd(x, full_matrices=False)
        U_k = U[:, :k]
        S_k = S[:k]
        V_k = Vh[:k, :].T  # [N, K]

        rec = (U_k * S_k.unsqueeze(0)) @ V_k.T  # [H*W, N]

        df_denoised = rec.reshape(H, W, N)
        ucomps = U_k.reshape(H, W, k)
        vcomps = V_k
        return df_denoised, ucomps, vcomps

    def select_connected(
        self,
        im: torch.Tensor,
        startpix: torch.Tensor,
        N: int,
        radius: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return up to N brightest connected pixels within radius of startpix.

        Inputs:
            im: [H, W]
            startpix: [2] long tensor [row, col]

        Returns:
            pixel_list: [M] linear indices
            roimap: [H, W] bool
            indexIJ: [M, 2] long, rows then cols
        """
        H, W = im.shape
        sr = int(startpix[0].item())
        sc = int(startpix[1].item())

        roimap = torch.zeros((H, W), dtype=torch.bool, device=im.device)
        queued = {(sr, sc)}
        queue = [(sr, sc)]
        selected = []

        def pix_val(rc: Tuple[int, int]) -> float:
            return float(im[rc[0], rc[1]].item())

        while len(selected) < N and len(queue) > 0:
            queue.sort(key=pix_val, reverse=True)
            pr, pc = queue.pop(0)

            if bool(roimap[pr, pc].item()):
                continue

            roimap[pr, pc] = True
            selected.append((pr, pc))

            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = pr + dr, pc + dc
                if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in queued:
                    d = math.sqrt((nr - sr) ** 2 + (nc - sc) ** 2)
                    if d <= radius:
                        queue.append((nr, nc))
                        queued.add((nr, nc))

        if len(selected) == 0:
            indexIJ = startpix.view(1, 2)
        else:
            indexIJ = torch.tensor(selected, dtype=torch.long, device=im.device)

        pixel_list = indexIJ[:, 0] * W + indexIJ[:, 1]
        return pixel_list, roimap, indexIJ

    def spk_fine(
        self,
        df_pos: torch.Tensor,
        npix: Optional[int] = None,
        radius: Optional[float] = None,
        initloc: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sub-pixel spike localization.

        df_pos: [H, W, N]
        initloc: [N, 2] integer coarse locations in 0-based coords

        returns:
            sloc: [N, 2] float
            bns: [N]
            roi: [H, W, N] bool
        """
        H, W, N = df_pos.shape
        npix = npix or self.fine_npix
        radius = radius or self.fine_radius

        sloc = []
        bns = []
        roi = []

        for i in range(N):
            im = df_pos[:, :, i]

            if initloc is None:
                flat_idx = torch.argmax(im)
                I = flat_idx // W
                J = flat_idx % W
                start = torch.stack([I, J]).long()
            else:
                start = initloc[i].long()

            pixel_list, roimap, indexIJ = self.select_connected(
                im=im,
                startpix=start,
                N=npix,
                radius=radius,
            )

            vals = im.reshape(-1)[pixel_list]
            w = vals.square()
            wsum = w.sum()

            if float(wsum.item()) <= 0.0:
                loc = indexIJ.float().mean(dim=0)
            else:
                loc = (indexIJ.float() * w.unsqueeze(1)).sum(dim=0) / wsum

            sloc.append(loc)
            bns.append(vals.mean())
            roi.append(roimap)

        sloc = torch.stack(sloc, dim=0)   # [N,2]
        bns = torch.stack(bns, dim=0)     # [N]
        roi = torch.stack(roi, dim=2)     # [H,W,N]
        return sloc, bns, roi

    def density_map(
        self,
        sloc: torch.Tensor,
        sz: Tuple[int, int],
        factor: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Build a high-resolution spike density map using 0-based coordinates.

        sloc: [N, 2] with row/col in 0-based pixel coordinates
        sz: (H, W)

        returns:
            cnt: [H*factor, W*factor]
            cen: (cen_row, cen_col)
        """
        H, W = sz
        factor = factor or self.factor

        device = sloc.device
        dtype = sloc.dtype

        edges_r = torch.linspace(-0.5, H - 0.5, H * factor + 1, device=device, dtype=dtype)
        edges_c = torch.linspace(-0.5, W - 0.5, W * factor + 1, device=device, dtype=dtype)

        sloc_r = sloc[:, 0].contiguous()
        sloc_c = sloc[:, 1].contiguous()
        r_idx = torch.bucketize(sloc_r, edges_r) - 1
        c_idx = torch.bucketize(sloc_c, edges_c) - 1

        valid = (
            (r_idx >= 0) & (r_idx < H * factor) &
            (c_idx >= 0) & (c_idx < W * factor)
        )

        cnt = torch.zeros((H * factor, W * factor), device=device, dtype=dtype)
        ones = torch.ones_like(r_idx[valid], dtype=dtype)
        cnt.index_put_((r_idx[valid], c_idx[valid]), ones, accumulate=True)

        cen_r = 0.5 * (edges_r[:-1] + edges_r[1:])
        cen_c = 0.5 * (edges_c[:-1] + edges_c[1:])
        return cnt, (cen_r, cen_c)

    def detect_peaks(
        self,
        alimap: torch.Tensor,
        threshold_abs: Optional[float] = None,
        kernel_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Local-max peak detection in torch.

        alimap: [H, W]
        returns:
            peaks: [N, 2] int64 [row_idx, col_idx]
        """
        threshold_abs = threshold_abs if threshold_abs is not None else self.cluster_threshold
        kernel_size = kernel_size if kernel_size is not None else self.peak_kernel_size
        kernel_size = self._ensure_odd(kernel_size)
        pad = kernel_size // 2

        x = alimap.unsqueeze(0).unsqueeze(0)
        pooled = F.max_pool2d(x, kernel_size=kernel_size, stride=1, padding=pad)
        is_peak = (x == pooled) & (x >= threshold_abs)
        peaks = torch.nonzero(is_peak[0, 0], as_tuple=False)

        if peaks.numel() == 0:
            return torch.empty((0, 2), dtype=torch.long, device=alimap.device)

        vals = alimap[peaks[:, 0], peaks[:, 1]]
        order = torch.argsort(vals, descending=True)
        return peaks[order]

    def assign_cluster(
        self,
        sloc: torch.Tensor,
        clust_cen: torch.Tensor,
        radius: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Assign spikes to nearest cluster center.

        sloc: [N, 2]
        clust_cen: [2, C]
        returns:
            clust_idx: [N], values in {0,1,...,C}; 0 = unassigned
        """
        radius = radius if radius is not None else self.assign_radius

        N = sloc.shape[0]
        if clust_cen.numel() == 0:
            return torch.zeros(N, dtype=torch.long, device=sloc.device)

        centers = clust_cen.T  # [C, 2]
        dist = torch.cdist(sloc.float(), centers.float())  # [N, C]
        min_dist, idx = torch.min(dist, dim=1)

        clust_idx = torch.zeros(N, dtype=torch.long, device=sloc.device)
        mask = min_dist < radius
        clust_idx[mask] = idx[mask] + 1
        return clust_idx

    def fp_support(
        self,
        footprint: torch.Tensor,
        clust_cen: torch.Tensor,
        r: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Limit footprint support to an r-pixel disk around the cluster center.

        footprint: [H, W, C]
        clust_cen: [2, C]
        """
        r = r if r is not None else self.footprint_radius
        H, W, C = footprint.shape

        if C == 0:
            support = torch.zeros_like(footprint, dtype=torch.bool)
            return footprint, support

        yy, xx = torch.meshgrid(
            torch.arange(H, device=footprint.device, dtype=footprint.dtype),
            torch.arange(W, device=footprint.device, dtype=footprint.dtype),
            indexing="ij",
        )
        yy = yy.unsqueeze(-1)
        xx = xx.unsqueeze(-1)

        cy = clust_cen[0].view(1, 1, C)
        cx = clust_cen[1].view(1, 1, C)

        dist = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        support = dist <= r

        fp = footprint.clone()
        fp[~support] = 0
        return fp, support

    def t_decompose(
        self,
        df: torch.Tensor,
        fp: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract temporal traces from df movie given footprints.

        df: [P, T]
        fp: [P, C]

        returns:
            trace_ls: [T, C]
            trace_new: [T, C]
        """
        C = fp.shape[1]
        if C == 0:
            T = df.shape[1]
            empty = torch.empty((T, 0), dtype=df.dtype, device=df.device)
            return empty, empty

        cc = fp.T @ fp
        proj = fp.T @ df

        eye = torch.eye(C, dtype=cc.dtype, device=cc.device)
        cc_reg = cc + self.solve_eps * eye

        trace_ls = torch.linalg.solve(cc_reg, proj)  # [C,T]

        cn = torch.diag(cc_reg)
        trace_new = trace_ls.clone()
        trace_nonneg = torch.clamp(trace_new, min=0)

        for i in range(C):
            trace_new[i, :] = (
                trace_nonneg[i, :]
                + (proj[i, :] - cc_reg[i, :] @ trace_nonneg) / cn[i]
            )

        trace_new = trace_new - trace_new.median(dim=1, keepdim=True).values

        return trace_ls.T.contiguous(), trace_new.T.contiguous()

    def compute_footprints(
        self,
        df: torch.Tensor,
        spike_frames: torch.Tensor,
        clust_idx: torch.Tensor,
        nclust: int,
    ) -> torch.Tensor:
        """
        Average df frames for spikes assigned to each cluster.

        df: [H, W, T]
        spike_frames: [N]
        clust_idx: [N]
        returns:
            footprint: [H, W, C]
        """
        H, W, _ = df.shape
        footprint = torch.zeros((H, W, nclust), dtype=df.dtype, device=df.device)

        for i in range(nclust):
            mask = clust_idx == (i + 1)
            if bool(mask.any().item()):
                frames = spike_frames[mask].long()
                footprint[:, :, i] = df.index_select(2, frames).mean(dim=2)

        return footprint

    @torch.inference_mode()
    def forward(self, data: Any) -> ALIResult:
        """Compute ALI spike localization, clusters, footprints, and traces."""
        work_device = self._resolve_device(data)
        data = self._as_tensor(data, device=work_device)

        if data.ndim != 3:
            raise ValueError("Expected input of shape [H, W, T]")

        H, W, T = data.shape
        f0 = data.mean(dim=2)

        # 1) high-pass filter
        df = self.hp_filter(data)

        # 2) coarse spike detect on sign-flipped movie
        spk = self.spk_coarse(-df)

        if spk.numel() == 0:
            empty_long = torch.empty((0,), dtype=torch.long, device=data.device)
            empty_map = torch.zeros((H * self.factor, W * self.factor), dtype=data.dtype, device=data.device)

            if spk.numel() == 0:
                empty_long = torch.empty((0,), dtype=torch.long, device=data.device)
                empty_map = torch.zeros((H * self.factor, W * self.factor), dtype=data.dtype, device=data.device)

                return ALIResult(
                    data=data,
                    df=df,
                    f0=f0,
                    spk=torch.empty((0, 3), dtype=torch.long, device=data.device),
                    df_ap=torch.empty((H, W, 0), dtype=data.dtype, device=data.device),
                    df_ap_denoised=torch.empty((H, W, 0), dtype=data.dtype, device=data.device),
                    ucomps=torch.empty((H, W, 0), dtype=data.dtype, device=data.device),
                    vcomps=torch.empty((0, 0), dtype=data.dtype, device=data.device),
                    spk_fine=torch.empty((0, 3), dtype=data.dtype, device=data.device),
                    brightness=torch.empty((0,), dtype=data.dtype, device=data.device),
                    roi=torch.empty((H, W, 0), dtype=torch.bool, device=data.device),
                    cnt=empty_map,
                    cen=(
                        torch.linspace(
                            -0.5 + 0.5 / self.factor,
                            H - 0.5 - 0.5 / self.factor,
                            H * self.factor,
                            device=data.device,
                            dtype=data.dtype,
                        ),
                        torch.linspace(
                            -0.5 + 0.5 / self.factor,
                            W - 0.5 - 0.5 / self.factor,
                            W * self.factor,
                            device=data.device,
                            dtype=data.dtype,
                        ),
                    ),
                    alimap=empty_map,
                    peaks=torch.empty((0, 2), dtype=torch.long, device=data.device),
                    clust_cen=torch.empty((2, 0), dtype=data.dtype, device=data.device),
                    clust_idx=empty_long,
                    sort_idx=empty_long,
                    footprint=torch.empty((H, W, 0), dtype=data.dtype, device=data.device),
                    fp=torch.empty((H, W, 0), dtype=data.dtype, device=data.device),
                    support=torch.empty((H, W, 0), dtype=torch.bool, device=data.device),
                    traces_ls=torch.empty((T, 0), dtype=data.dtype, device=data.device),
                    traces=torch.empty((T, 0), dtype=data.dtype, device=data.device),
                )

        # 3) denoise candidate spike frames
        spike_frames = spk[:, 2].long()
        df_ap = df.index_select(2, spike_frames)
        df_ap_denoised, ucomps, vcomps = self.denoising(df_ap, self.nsvd)

        # 4) sub-pixel spike localization
        fine_xy, brightness, roi = self.spk_fine(
            df_pos=-df_ap_denoised,
            npix=self.fine_npix,
            radius=self.fine_radius,
            initloc=spk[:, :2],
        )
        spk_fine = spk.to(dtype=data.dtype).clone()
        spk_fine[:, :2] = fine_xy

        # 5) ALI map
        cnt, cen = self.density_map(spk_fine[:, :2], (H, W), self.factor)
        alimap = self._gaussian_blur_image(
            cnt,
            sigma=self.alimap_sigma,
            radius=self.alimap_gaussian_radius,
        )

        # 6) detect ALI clusters
        peaks = self.detect_peaks(
            alimap,
            threshold_abs=self.cluster_threshold,
            kernel_size=self.peak_kernel_size,
        )

        if peaks.numel() == 0:
            clust_cen = torch.empty((2, 0), dtype=data.dtype, device=data.device)
            clust_idx = torch.zeros(spk.shape[0], dtype=torch.long, device=data.device)
            sort_idx = torch.empty((0,), dtype=torch.long, device=data.device)
            footprint = torch.empty((H, W, 0), dtype=data.dtype, device=data.device)
            fp = footprint.clone()
            support = torch.empty((H, W, 0), dtype=torch.bool, device=data.device)
            traces_ls = torch.empty((T, 0), dtype=data.dtype, device=data.device)
            traces = traces_ls.clone()
        else:
            clust_cen = torch.stack(
                [cen[0][peaks[:, 0]], cen[1][peaks[:, 1]]],
                dim=0,
            )  # [2, C]

            clust_idx = self.assign_cluster(
                spk_fine[:, :2],
                clust_cen,
                radius=self.assign_radius,
            )

            sort_idx = torch.argsort(clust_cen[1], descending=True)
            nclust = clust_cen.shape[1]

            # 7) footprints
            footprint = self.compute_footprints(
                df=df,
                spike_frames=spike_frames,
                clust_idx=clust_idx,
                nclust=nclust,
            )

            # 8) support-limited footprints
            fp, support = self.fp_support(
                footprint=footprint,
                clust_cen=clust_cen,
                r=self.footprint_radius,
            )

            # 9) temporal traces
            traces_ls, traces = self.t_decompose(
                df.reshape(H * W, T),
                fp.reshape(H * W, nclust),
            )
            traces = torch.clamp(traces, min=0)

        return ALIResult(
            data=data,
            df=df,
            f0=f0,
            spk=spk,
            df_ap=df_ap,
            df_ap_denoised=df_ap_denoised,
            ucomps=ucomps,
            vcomps=vcomps,
            spk_fine=spk_fine,
            brightness=brightness,
            roi=roi,
            cnt=cnt,
            cen=cen,
            alimap=alimap,
            peaks=peaks,
            clust_cen=clust_cen,
            clust_idx=clust_idx,
            sort_idx=sort_idx,
            footprint=footprint,
            fp=fp,
            support=support,
            traces_ls=traces_ls,
            traces=traces,
        )
