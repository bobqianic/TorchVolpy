from __future__ import annotations

from typing import Literal, Optional, Tuple, Union

import numpy as np
import torch

from ..filter import Filter


class Translation:
    """
    Estimate per-frame translational shifts against a template using
    FFT-based cross-correlation in PyTorch.

    Returns shifts as [dy, dx], meaning:
        corrected_frame = shift(frame, dy, dx)

    Assumptions
    -----------
    - movie axis 0 is time
    - frame shape is either:
        (Y, X)
      or
        (Y, X, C)   # channel-last
    - correlation is done on a 2D plane:
        * (Y, X) directly
        * (Y, X, C) reduced to mean over channels

    Notes
    -----
    - Integer-pixel shifts are estimated from the correlation peak.
    - If upsample_factor > 1, local DFT upsampling or quadratic peak
      refinement returns subpixel shifts quantized to 1 / upsample_factor pixels.
    - It reads the movie in chunks, so it does not load the full movie.
    - The default normalized cross-power mode is phase correlation:
          corr = ifft( FFT(template) * conj(FFT(frame)) / abs(...) )
      normalization="none" uses unnormalized FFT cross-correlation, closer
      to CaImAn/VolPy rigid registration.
    """

    def __init__(
        self,
        movie,
        template: torch.Tensor,
        max_shifts: Union[int, Tuple[int, int]],
        frames_per_chunk: int = 256,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
        use_hann: bool = True,
        center: bool = True,
        upsample_factor: int = 1,
        subpixel_method: Literal["none", "quadratic", "dft"] = "quadratic",
        normalization: Literal["phase", "none"] = "phase",
        eps: float = 1e-8,
        high_pass_filter_size: Optional[int] = None,
        add_to_movie: float = 0.0,
    ) -> None:
        self.movie = movie
        self.frames_per_chunk = int(frames_per_chunk)
        self.device = torch.device(device)
        self.dtype = dtype
        self.use_hann = bool(use_hann)
        self.center = bool(center)
        self.upsample_factor = int(upsample_factor)
        self.subpixel_method = subpixel_method
        self.normalization = normalization
        self.eps = float(eps)
        self.add_to_movie = float(add_to_movie)

        self.high_pass_filter_size = high_pass_filter_size

        if self.frames_per_chunk <= 0:
            raise ValueError("frames_per_chunk must be > 0")
        if self.upsample_factor <= 0:
            raise ValueError("upsample_factor must be > 0")
        if self.subpixel_method not in ("none", "quadratic", "dft"):
            raise ValueError("subpixel_method must be 'none', 'quadratic', or 'dft'")
        if self.normalization not in ("phase", "none"):
            raise ValueError("normalization must be 'phase' or 'none'")

        if self.high_pass_filter_size is not None:
            if int(self.high_pass_filter_size) <= 0:
                raise ValueError("high_pass_filter_size must be > 0")
            if int(self.high_pass_filter_size) % 2 == 0:
                raise ValueError("high_pass_filter_size must be odd")
            self.high_pass_filter_size = int(self.high_pass_filter_size)

        self.max_shifts = self._normalize_max_shifts(max_shifts)

        self.frame_shape = tuple(movie.frame_shape)
        if len(self.frame_shape) not in (2, 3):
            raise ValueError(
                f"Only frame shapes (Y, X) or (Y, X, C) are supported, got {self.frame_shape}"
            )

        self.template = self._prepare_image(template)
        self.height, self.width = int(self.template.shape[-2]), int(self.template.shape[-1])

        max_y, max_x = self.max_shifts
        if max_y >= self.height or max_x >= self.width:
            raise ValueError(
                f"max_shifts={self.max_shifts} is too large for spatial size {(self.height, self.width)}"
            )

        self.window = self._make_window(self.height, self.width) if self.use_hann else None

        self.template_fft = self._precompute_template_fft(self.template)
        self.search_y_idx, self.search_x_idx = self._build_search_indices(
            self.height,
            self.width,
            self.max_shifts,
            self.device,
        )

    @staticmethod
    def _normalize_max_shifts(
        max_shifts: Union[int, Tuple[int, int]]
    ) -> Tuple[int, int]:
        if isinstance(max_shifts, int):
            if max_shifts < 0:
                raise ValueError("max_shifts must be >= 0")
            return int(max_shifts), int(max_shifts)

        if len(max_shifts) != 2:
            raise ValueError("max_shifts must be an int or a tuple (max_y, max_x)")

        max_y, max_x = int(max_shifts[0]), int(max_shifts[1])
        if max_y < 0 or max_x < 0:
            raise ValueError("max_shifts values must be >= 0")
        return max_y, max_x

    def _prepare_image(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert template/frame data to a 2D correlation plane on self.device/self.dtype.
        """
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)

        x = x.to(device=self.device, dtype=self.dtype)

        if x.ndim == len(self.frame_shape):
            if len(self.frame_shape) == 2:
                return x
            else:
                return x.mean(dim=-1)

        raise ValueError(
            f"Template shape {tuple(x.shape)} is incompatible with movie frame shape {self.frame_shape}"
        )

    def _prepare_batch(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Convert frame batch to shape (B, Y, X) on self.device/self.dtype.
        """
        if not isinstance(frames, torch.Tensor):
            frames = torch.as_tensor(frames)

        frames = frames.to(device=self.device, dtype=self.dtype)

        expected_ndim = len(self.frame_shape) + 1
        if frames.ndim != expected_ndim:
            raise ValueError(
                f"Expected batch ndim={expected_ndim}, got {frames.ndim} for shape {tuple(frames.shape)}"
            )

        if len(self.frame_shape) == 2:
            return frames

        return frames.mean(dim=-1)

    @staticmethod
    def _make_window(height: int, width: int) -> torch.Tensor:
        wy = torch.hann_window(height, periodic=False)
        wx = torch.hann_window(width, periodic=False)
        return torch.outer(wy, wx)

    def _precompute_template_fft(self, template_2d: torch.Tensor) -> torch.Tensor:
        x = template_2d
        if self.add_to_movie != 0.0:
            x = x + self.add_to_movie

        if self.center:
            x = x - x.mean()

        if self.window is not None:
            x = x * self.window.to(device=x.device, dtype=x.dtype)

        return torch.fft.fftn(x, dim=(-2, -1))

    @staticmethod
    def _build_search_indices(
        height: int,
        width: int,
        max_shifts: Tuple[int, int],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        max_y, max_x = max_shifts

        if max_y == 0:
            y_idx = torch.tensor([0], device=device, dtype=torch.long)
        else:
            y_idx = torch.cat(
                [
                    torch.arange(0, max_y + 1, device=device),
                    torch.arange(height - max_y, height, device=device),
                ]
            ).long()

        if max_x == 0:
            x_idx = torch.tensor([0], device=device, dtype=torch.long)
        else:
            x_idx = torch.cat(
                [
                    torch.arange(0, max_x + 1, device=device),
                    torch.arange(width - max_x, width, device=device),
                ]
            ).long()

        return y_idx, x_idx

    @staticmethod
    def _to_signed_shift(idx: torch.Tensor, size: int) -> torch.Tensor:
        """
        Convert FFT-wrap index to signed shift.
        Example:
            0 -> 0
            1 -> +1
            size-1 -> -1
        """
        half = size // 2
        return torch.where(idx <= half, idx, idx - size)

    def _phase_correlation(
        self,
        frames_2d: torch.Tensor,
        return_cross_power: bool = False,
    ):
        """
        frames_2d: (B, Y, X)
        returns corr: (B, Y, X)
        """
        x = frames_2d

        if self.high_pass_filter_size is not None:
            x = Filter.hp_gaussian(
                size=self.high_pass_filter_size,
                movie=x,
                sigma=None,
                padding_mode="reflect",
                dtype=self.dtype,
                device=self.device,
            )

        if self.add_to_movie != 0.0:
            x = x + self.add_to_movie

        if self.center:
            x = x - x.mean(dim=(-2, -1), keepdim=True)

        if self.window is not None:
            x = x * self.window.to(device=x.device, dtype=x.dtype).unsqueeze(0)

        frame_fft = torch.fft.fftn(x, dim=(-2, -1))

        # Use template * conj(frame) so the peak is the correction shift to apply.
        cross_power = self.template_fft.unsqueeze(0) * torch.conj(frame_fft)

        if self.normalization == "phase":
            denom = torch.clamp(torch.abs(cross_power), min=self.eps)
            cross_power = cross_power / denom

        corr = torch.fft.ifftn(
            cross_power,
            dim=(-2, -1),
        ).real

        if return_cross_power:
            return corr, cross_power
        return corr

    def _upsampled_dft_batched(
        self,
        data: torch.Tensor,
        upsampled_region_size: int,
        upsample_factor: int,
        axis_offsets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Batched matrix-multiply DFT in a small region around each frame's peak.

        This is the torch/GPU analogue of the Guizar-Sicairos local DFT
        refinement used by scikit-image/CaImAn, without zero-padding the full
        image by upsample_factor.
        """
        b, height, width = data.shape
        real_dtype = torch.float64 if data.dtype == torch.complex128 else torch.float32
        device = data.device

        size_y = int(upsampled_region_size)
        size_x = int(upsampled_region_size)
        factor = float(upsample_factor)

        y_region = (
            torch.arange(size_y, device=device, dtype=real_dtype).unsqueeze(0)
            - axis_offsets[:, 0].to(device=device, dtype=real_dtype).unsqueeze(1)
        )
        x_region = (
            torch.arange(size_x, device=device, dtype=real_dtype).unsqueeze(0)
            - axis_offsets[:, 1].to(device=device, dtype=real_dtype).unsqueeze(1)
        )

        y_freq = torch.fft.ifftshift(
            torch.arange(height, device=device, dtype=real_dtype)
        ) - torch.floor(torch.tensor(height // 2, device=device, dtype=real_dtype))
        x_freq = torch.fft.ifftshift(
            torch.arange(width, device=device, dtype=real_dtype)
        ) - torch.floor(torch.tensor(width // 2, device=device, dtype=real_dtype))

        row_phase = (
            -2j
            * torch.pi
            / (float(height) * factor)
            * y_region[:, :, None]
            * y_freq[None, None, :]
        )
        col_phase = (
            -2j
            * torch.pi
            / (float(width) * factor)
            * x_freq[None, :, None]
            * x_region[:, None, :]
        )

        row_kernel = torch.exp(row_phase).to(dtype=data.dtype)
        col_kernel = torch.exp(col_phase).to(dtype=data.dtype)

        tmp = torch.einsum("brh,bhw->brw", row_kernel, data)
        return torch.einsum("brw,bwc->brc", tmp, col_kernel)

    def _refine_subpixel_dft(
        self,
        cross_power: torch.Tensor,
        dy: torch.Tensor,
        dx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.upsample_factor <= 1 or self.subpixel_method != "dft":
            return dy, dx

        factor = int(self.upsample_factor)
        region_size = int(np.ceil(factor * 1.5))
        dftshift = float(np.fix(region_size / 2.0))

        shifts = torch.stack((dy.to(self.dtype), dx.to(self.dtype)), dim=1)
        shifts = torch.round(shifts * float(factor)) / float(factor)
        offsets = dftshift - shifts * float(factor)

        # Same convention as skimage/CaImAn: DFT of image_product.conj(), then
        # conjugate back. Here cross_power is FFT(template) * conj(FFT(frame)),
        # so the refined shifts are still correction shifts to apply directly.
        upsampled = self._upsampled_dft_batched(
            cross_power.conj(),
            region_size,
            factor,
            offsets,
        ).conj()

        flat = torch.abs(upsampled).reshape(upsampled.shape[0], -1)
        peak_idx = torch.argmax(flat, dim=1)
        local_y = torch.div(peak_idx, region_size, rounding_mode="floor")
        local_x = peak_idx % region_size

        maxima_y = local_y.to(self.dtype) - dftshift
        maxima_x = local_x.to(self.dtype) - dftshift

        dy = shifts[:, 0] + maxima_y / float(factor)
        dx = shifts[:, 1] + maxima_x / float(factor)

        max_y, max_x = self.max_shifts
        dy = torch.clamp(dy, -float(max_y), float(max_y))
        dx = torch.clamp(dx, -float(max_x), float(max_x))
        return dy, dx

    def _refine_subpixel_quadratic(
        self,
        corr: torch.Tensor,
        peak_y: torch.Tensor,
        peak_x: torch.Tensor,
        dy: torch.Tensor,
        dx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Refine integer FFT-correlation peaks with a separable quadratic fit.

        This is a fast batched approximation to local DFT upsampling. It runs on
        the same device as corr and avoids per-frame CPU/OpenCV work.
        """
        if self.upsample_factor <= 1 or self.subpixel_method == "none":
            return dy, dx

        b = corr.shape[0]
        batch = torch.arange(b, device=corr.device)

        ym = (peak_y - 1) % self.height
        yp = (peak_y + 1) % self.height
        xm = (peak_x - 1) % self.width
        xp = (peak_x + 1) % self.width

        center = corr[batch, peak_y, peak_x]
        y_prev = corr[batch, ym, peak_x]
        y_next = corr[batch, yp, peak_x]
        x_prev = corr[batch, peak_y, xm]
        x_next = corr[batch, peak_y, xp]

        denom_y = y_prev - 2 * center + y_next
        denom_x = x_prev - 2 * center + x_next

        zero = torch.zeros_like(center)
        delta_y = torch.where(
            torch.abs(denom_y) > self.eps,
            0.5 * (y_prev - y_next) / denom_y,
            zero,
        )
        delta_x = torch.where(
            torch.abs(denom_x) > self.eps,
            0.5 * (x_prev - x_next) / denom_x,
            zero,
        )

        # Quadratic interpolation is only locally valid around the winning bin.
        delta_y = torch.clamp(delta_y, -0.5, 0.5)
        delta_x = torch.clamp(delta_x, -0.5, 0.5)

        dy = dy.to(self.dtype) + delta_y
        dx = dx.to(self.dtype) + delta_x

        if self.upsample_factor > 1:
            factor = float(self.upsample_factor)
            dy = torch.round(dy * factor) / factor
            dx = torch.round(dx * factor) / factor

        max_y, max_x = self.max_shifts
        dy = torch.clamp(dy, -float(max_y), float(max_y))
        dx = torch.clamp(dx, -float(max_x), float(max_x))
        return dy, dx

    def _find_shifts_from_corr(
        self,
        corr: torch.Tensor,
        cross_power: Optional[torch.Tensor] = None,
        return_scores: bool = False,
    ):
        """
        corr: (B, Y, X)
        """
        region = corr.index_select(-2, self.search_y_idx).index_select(-1, self.search_x_idx)
        b = region.shape[0]

        flat = region.reshape(b, -1)
        peak_idx = torch.argmax(flat, dim=1)

        region_w = region.shape[-1]
        local_y = torch.div(peak_idx, region_w, rounding_mode="floor")
        local_x = peak_idx % region_w

        peak_y = self.search_y_idx[local_y]
        peak_x = self.search_x_idx[local_x]

        dy = self._to_signed_shift(peak_y, self.height)
        dx = self._to_signed_shift(peak_x, self.width)

        if (
            self.upsample_factor > 1
            and self.subpixel_method == "dft"
            and cross_power is not None
        ):
            dy, dx = self._refine_subpixel_dft(cross_power, dy, dx)
            shifts = torch.stack([dy, dx], dim=1).to(self.dtype)
        elif self.upsample_factor > 1 and self.subpixel_method != "none":
            dy, dx = self._refine_subpixel_quadratic(corr, peak_y, peak_x, dy, dx)
            shifts = torch.stack([dy, dx], dim=1).to(self.dtype)
        else:
            shifts = torch.stack([dy, dx], dim=1).to(torch.int64)

        if not return_scores:
            return shifts

        scores = flat.gather(1, peak_idx[:, None]).squeeze(1)
        return shifts, scores

    @torch.inference_mode()
    def estimate_range(
        self,
        start: int = 0,
        stop: Optional[int] = None,
        return_scores: bool = False,
    ):
        """
        Estimate shifts for movie frames in [start, stop).

        Returns
        -------
        shifts : torch.Tensor, shape (N, 2), dtype int64
            Each row is [dy, dx] to apply to the frame.
        scores : torch.Tensor, shape (N,), optional
            Peak correlation score for each frame.
        """
        if stop is None:
            stop = self.movie.num_frames

        start = int(start)
        stop = int(stop)

        if start < 0 or stop < start or stop > self.movie.num_frames:
            raise ValueError(f"Invalid range start={start}, stop={stop}")

        all_shifts = []
        all_scores = []

        for chunk_start in range(start, stop, self.frames_per_chunk):
            chunk_stop = min(chunk_start + self.frames_per_chunk, stop)

            frames = self.movie.read_frames(
                chunk_start,
                chunk_stop,
                as_tensor=True,
                dtype=np.float32,
            )

            frames_2d = self._prepare_batch(frames)
            need_cross_power = self.upsample_factor > 1 and self.subpixel_method == "dft"
            if need_cross_power:
                corr, cross_power = self._phase_correlation(
                    frames_2d,
                    return_cross_power=True,
                )
            else:
                corr = self._phase_correlation(frames_2d)
                cross_power = None

            if return_scores:
                shifts, scores = self._find_shifts_from_corr(
                    corr,
                    cross_power=cross_power,
                    return_scores=True,
                )
                all_scores.append(scores.cpu())
            else:
                shifts = self._find_shifts_from_corr(
                    corr,
                    cross_power=cross_power,
                    return_scores=False,
                )

            all_shifts.append(shifts.cpu())

        shifts = torch.cat(all_shifts, dim=0) if all_shifts else torch.empty((0, 2), dtype=torch.int64)

        if not return_scores:
            return shifts

        scores = torch.cat(all_scores, dim=0) if all_scores else torch.empty((0,), dtype=self.dtype)
        return shifts, scores

    @torch.inference_mode()
    def estimate_frame(
        self,
        index: int,
        return_score: bool = False,
    ):
        """
        Estimate shift for a single frame.
        """
        frame = self.movie.read(index, as_tensor=True, dtype=np.float32)

        if frame.ndim == len(self.frame_shape):
            frame = frame.unsqueeze(0)

        frames_2d = self._prepare_batch(frame)
        need_cross_power = self.upsample_factor > 1 and self.subpixel_method == "dft"
        if need_cross_power:
            corr, cross_power = self._phase_correlation(
                frames_2d,
                return_cross_power=True,
            )
        else:
            corr = self._phase_correlation(frames_2d)
            cross_power = None

        if return_score:
            shifts, scores = self._find_shifts_from_corr(
                corr,
                cross_power=cross_power,
                return_scores=True,
            )
            return shifts[0].cpu(), scores[0].cpu()

        shifts = self._find_shifts_from_corr(
            corr,
            cross_power=cross_power,
            return_scores=False,
        )
        return shifts[0].cpu()

    def __call__(
        self,
        start: int = 0,
        stop: Optional[int] = None,
        return_scores: bool = False,
    ):
        return self.estimate_range(start=start, stop=stop, return_scores=return_scores)
