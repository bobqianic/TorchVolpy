from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from .Template import Template
from .Translation import Translation
from ..movie import Movie


class MotionCorrect:
    """
    Build a template, estimate per-frame translational shifts, apply them,
    and save the corrected movie to disk.

    Workflow
    --------
    1) Build template with Template
    2) Estimate shifts with Translation
    3) Apply rigid shifts to each frame
    4) Save corrected movie to HDF5

    Notes
    -----
    - Translation is assumed to return correction shifts [dy, dx]
      that should be applied directly to the frame.
    - Subpixel correction uses batched torch grid_sample, so it runs on GPU
      when device is a CUDA device.
    - Frames are processed in chunks; the whole movie is never loaded at once.
    """

    def __init__(
        self,
        movie,
        max_shifts: Union[int, Tuple[int, int]],
        frames_per_chunk: int = 256,
        device: Union[str, torch.device] = "cpu",
        template=None,
        shifts: Optional[Union[np.ndarray, torch.Tensor, Sequence[Sequence[int]]]] = None,
        template_kwargs: Optional[Dict[str, Any]] = None,
        translation_kwargs: Optional[Dict[str, Any]] = None,
        template_strategy: Literal["binmedian", "caiman_rigid"] = "binmedian",
        high_pass_filter_size: Optional[int] = None,
        upsample_factor: int = 10,
        interpolation: Literal["integer", "nearest", "bilinear", "bicubic", "opencv_cubic"] = "bicubic",
        padding_mode: Literal["zeros", "border", "reflection"] = "border",
        border_nan: Union[bool, Literal["copy", "min", "nan"]] = False,
        copy_border_strips: bool = False,
        align_corners: bool = True,
        clip_interpolated: bool = True,
        add_to_movie: Optional[float] = None,
        gsig_filt: Optional[Tuple[float, float]] = None,
    ) -> None:
        self.movie = movie
        self.max_shifts = self._normalize_max_shifts(max_shifts)
        self.frames_per_chunk = int(frames_per_chunk)
        self.device = torch.device(device)

        if self.frames_per_chunk <= 0:
            raise ValueError("frames_per_chunk must be > 0")

        self.frame_shape = tuple(movie.frame_shape)
        if len(self.frame_shape) not in (2, 3):
            raise ValueError(
                f"Only frame shapes (Y, X) or (Y, X, C) are supported, got {self.frame_shape}"
            )

        if high_pass_filter_size is not None:
            high_pass_filter_size = int(high_pass_filter_size)
            if high_pass_filter_size <= 0:
                raise ValueError("high_pass_filter_size must be > 0")
            if high_pass_filter_size % 2 == 0:
                raise ValueError("high_pass_filter_size must be odd")

        self.high_pass_filter_size = high_pass_filter_size
        self.upsample_factor = int(upsample_factor)
        if self.upsample_factor <= 0:
            raise ValueError("upsample_factor must be > 0")

        if interpolation not in ("integer", "nearest", "bilinear", "bicubic", "opencv_cubic"):
            raise ValueError(
                "interpolation must be 'integer', 'nearest', 'bilinear', 'bicubic', or 'opencv_cubic'"
            )
        if padding_mode not in ("zeros", "border", "reflection"):
            raise ValueError("padding_mode must be 'zeros', 'border', or 'reflection'")
        if template_strategy not in ("binmedian", "caiman_rigid"):
            raise ValueError("template_strategy must be 'binmedian' or 'caiman_rigid'")
        if border_nan not in (False, True, "copy", "min", "nan"):
            raise ValueError("border_nan must be False, True, 'copy', 'min', or 'nan'")

        self.gsig_filt = self._normalize_gsig_filt(gsig_filt)
        self.template_strategy = template_strategy
        self.interpolation = interpolation
        self.padding_mode = padding_mode
        self.border_nan = border_nan
        self.copy_border_strips = bool(copy_border_strips)
        self.align_corners = bool(align_corners)
        self.clip_interpolated = bool(clip_interpolated)
        self._base_grid_cache = {}
        self.add_to_movie = None if add_to_movie is None else float(add_to_movie)

        self.num_frames = int(movie.num_frames)

        self.template_kwargs = dict(template_kwargs or {})
        self.translation_kwargs = dict(translation_kwargs or {})

        self.template_builder = None
        self.translation_estimator = None

        self.template = template
        self.shifts = (
            None if shifts is None else self._normalize_shifts(shifts, self.num_frames)
        )
        self.scores = None

    @staticmethod
    def _normalize_max_shifts(max_shifts: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
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

    @staticmethod
    def _normalize_gsig_filt(gsig_filt: Optional[Sequence[float]]) -> Optional[Tuple[float, float]]:
        if gsig_filt is None:
            return None
        if isinstance(gsig_filt, (int, float)):
            value = float(gsig_filt)
            if value <= 0:
                raise ValueError("gsig_filt must be > 0")
            return value, value
        if len(gsig_filt) == 1:
            value = float(gsig_filt[0])
            if value <= 0:
                raise ValueError("gsig_filt must be > 0")
            return value, value
        if len(gsig_filt) != 2:
            raise ValueError("gsig_filt must be one value or two values")
        gy, gx = float(gsig_filt[0]), float(gsig_filt[1])
        if gy <= 0 or gx <= 0:
            raise ValueError("gsig_filt values must be > 0")
        return gy, gx

    @staticmethod
    def _normalize_shifts(
        shifts: Union[np.ndarray, torch.Tensor, Sequence[Sequence[int]]],
        num_frames: int,
        allow_subpixel: bool = True,
    ) -> torch.Tensor:
        if isinstance(shifts, torch.Tensor):
            s = shifts.detach().cpu()
        else:
            s = torch.as_tensor(shifts)

        if s.ndim != 2 or s.shape[1] != 2:
            raise ValueError(f"shifts must have shape (T, 2), got {tuple(s.shape)}")

        if s.shape[0] != num_frames:
            raise ValueError(
                f"Number of shifts ({s.shape[0]}) does not match number of movie frames ({num_frames})"
            )

        if allow_subpixel:
            s = s.to(torch.float32)
            if not torch.isfinite(s).all():
                raise ValueError("shifts must contain only finite values")
            return s

        if not torch.is_floating_point(s):
            s = s.to(torch.int64)
        else:
            if not torch.allclose(s, torch.round(s)):
                raise ValueError(
                    "This MotionCorrect implementation supports integer-pixel shifts only"
                )
            s = torch.round(s).to(torch.int64)

        return s

    @staticmethod
    def _to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _estimate_movie_min(self, max_frames: int = 400) -> float:
        n = min(int(self.num_frames), int(max_frames))
        if n <= 0:
            raise ValueError("Movie has no frames")

        min_val = float("inf")
        chunk = min(int(self.frames_per_chunk), n)
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            frames = self.movie.read_frames(
                start,
                stop,
                as_tensor=False,
                dtype=np.float32,
            )
            if self.gsig_filt is not None:
                frames = self._caiman_high_pass_filter_space(frames, self.gsig_filt)
            cur = float(np.nanmin(frames))
            min_val = min(min_val, cur)

        if not np.isfinite(min_val):
            raise ValueError("Could not estimate a finite movie minimum")
        return min_val

    @staticmethod
    def _caiman_highpass_kernel_2d(
        gsig_filt: Tuple[float, float],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        gsig = tuple(float(v) for v in gsig_filt)
        ksize = tuple(int((3 * v) // 2 * 2 + 1) for v in gsig)
        # CaImAn's high_pass_filter_space builds the 2-D kernel from the first
        # axis' sigma/kernel. The benchmark datasets use the symmetric (10, 10).
        coords = torch.arange(ksize[0], dtype=dtype, device=device) - (ksize[0] - 1) / 2.0
        ker = torch.exp(-(coords * coords) / (2.0 * gsig[0] * gsig[0]))
        ker = (ker / ker.sum()).reshape(-1, 1)
        ker2d = ker.mm(ker.T)

        center = ksize[0] // 2
        idx = torch.arange(ksize[0], device=device, dtype=torch.int64) - center
        yy, xx = torch.meshgrid(idx, idx, indexing="ij")
        radius2 = yy * yy + xx * xx
        edge_radius2 = torch.as_tensor(center * center, device=device, dtype=torch.int64)
        # Match OpenCV's cv2.getGaussianKernel/dot thresholding in CaImAn:
        # the axial boundary pixels compare equal, while diagonal integer
        # boundary pixels fall just below the floating threshold.
        center_mask = (radius2 < edge_radius2) | (
            (radius2 == edge_radius2) & ((yy == 0) | (xx == 0))
        )
        ker2d = ker2d.clone()
        ker2d[center_mask] -= ker2d[center_mask].mean()
        ker2d[~center_mask] = 0
        return ker2d

    @staticmethod
    def _opencv_reflect_pad_nchw(frames_nchw: torch.Tensor, pad_y: int, pad_x: int) -> torch.Tensor:
        if pad_y == 0 and pad_x == 0:
            return frames_nchw
        height = int(frames_nchw.shape[-2])
        width = int(frames_nchw.shape[-1])
        y_idx = MotionCorrect._opencv_reflect_indices(
            torch.arange(-pad_y, height + pad_y, device=frames_nchw.device),
            height,
        )
        x_idx = MotionCorrect._opencv_reflect_indices(
            torch.arange(-pad_x, width + pad_x, device=frames_nchw.device),
            width,
        )
        return frames_nchw.index_select(-2, y_idx).index_select(-1, x_idx)

    @staticmethod
    def _caiman_high_pass_filter_space_tensor(
        img_orig: torch.Tensor,
        gsig_filt: Tuple[float, float],
        *,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        x = torch.as_tensor(img_orig, device=device, dtype=dtype)
        squeeze_time = False
        if x.ndim == 2:
            x = x.unsqueeze(0)
            squeeze_time = True
        if x.ndim != 3:
            raise ValueError(f"CaImAn high-pass filtering expects 2D or 3D input, got {tuple(x.shape)}")

        kernel = MotionCorrect._caiman_highpass_kernel_2d(
            gsig_filt,
            dtype=x.dtype,
            device=x.device,
        )
        pad_y = int(kernel.shape[0] // 2)
        pad_x = int(kernel.shape[1] // 2)
        frames = x.unsqueeze(1)
        padded = MotionCorrect._opencv_reflect_pad_nchw(frames, pad_y=pad_y, pad_x=pad_x)
        filtered = F.conv2d(padded, kernel.view(1, 1, *kernel.shape)).squeeze(1)
        if squeeze_time:
            filtered = filtered.squeeze(0)
        return filtered

    @staticmethod
    def _caiman_high_pass_filter_space(img_orig: np.ndarray, gsig_filt: Tuple[float, float]) -> np.ndarray:
        arr = np.asarray(img_orig, dtype=np.float32)
        filtered = MotionCorrect._caiman_high_pass_filter_space_tensor(
            torch.as_tensor(np.ascontiguousarray(arr)),
            gsig_filt,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        return filtered.detach().cpu().numpy().astype(np.float32, copy=False)

    @staticmethod
    def _caiman_bin_median_array(
        movie: np.ndarray,
        window: int = 10,
        exclude_nans: bool = True,
    ) -> np.ndarray:
        arr = np.asarray(movie, dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError(f"CaImAn bin_median compatibility expects (T, Y, X), got {arr.shape}")
        T, height, width = arr.shape
        if T <= 0:
            raise ValueError("Movie has no frames")
        window = min(int(window), T)
        num_windows = int(T // window)
        num_frames = num_windows * window
        if num_frames <= 0:
            raise ValueError("No frames available for bin_median")

        reshaped = np.reshape(arr[:num_frames], (window, num_windows, height, width))
        if exclude_nans:
            return np.nanmedian(np.nanmean(reshaped, axis=0), axis=0).astype(np.float32, copy=False)
        return np.median(np.mean(reshaped, axis=0), axis=0).astype(np.float32, copy=False)

    @staticmethod
    def _caiman_upsampled_dft(
        data: np.ndarray,
        upsampled_region_size,
        upsample_factor=1,
        axis_offsets=None,
    ) -> np.ndarray:
        data = np.asarray(data)
        if data.ndim != 2:
            raise ValueError(f"CaImAn local DFT refinement expects a 2D array, got {data.shape}")

        if not hasattr(upsampled_region_size, "__iter__"):
            upsampled_region_size = [upsampled_region_size] * data.ndim
        elif len(upsampled_region_size) != data.ndim:
            raise ValueError("upsampled_region_size must match the input dimensionality")
        upsampled_region_size = [int(size) for size in upsampled_region_size]

        if axis_offsets is None:
            axis_offsets = [0] * data.ndim
        elif len(axis_offsets) != data.ndim:
            raise ValueError("axis_offsets must match the input dimensionality")

        col_kernel = np.exp(
            (-1j * 2 * np.pi / (data.shape[1] * upsample_factor))
            * (
                np.fft.ifftshift(np.arange(data.shape[1]))[:, None]
                - np.floor(data.shape[1] // 2)
            ).dot(np.arange(upsampled_region_size[1])[None, :] - axis_offsets[1])
        )
        row_kernel = np.exp(
            (-1j * 2 * np.pi / (data.shape[0] * upsample_factor))
            * (np.arange(upsampled_region_size[0])[:, None] - axis_offsets[0]).dot(
                np.fft.ifftshift(np.arange(data.shape[0]))[None, :]
                - np.floor(data.shape[0] // 2)
            )
        )

        output = np.tensordot(row_kernel, data, axes=[1, 0])
        return np.tensordot(output, col_kernel, axes=[1, 0])

    @staticmethod
    def _shift_from_match_response(
        res: np.ndarray,
        *,
        max_shifts: Tuple[int, int],
    ) -> Tuple[float, float, float]:
        max_y, max_x = int(max_shifts[0]), int(max_shifts[1])
        peak_y, peak_x = np.unravel_index(int(np.argmax(res)), res.shape)
        avg_corr = float(np.mean(res))

        dy_offset = 0.0
        dx_offset = 0.0
        if (0 < peak_y < 2 * max_y - 1) and (0 < peak_x < 2 * max_x - 1):
            center = float(res[peak_y, peak_x])
            y_minus = float(res[peak_y - 1, peak_x])
            y_plus = float(res[peak_y + 1, peak_x])
            x_minus = float(res[peak_y, peak_x - 1])
            x_plus = float(res[peak_y, peak_x + 1])

            if min(center, y_minus, y_plus, x_minus, x_plus) > 0.0:
                log_center = np.log(center)
                log_y_minus = np.log(y_minus)
                log_y_plus = np.log(y_plus)
                log_x_minus = np.log(x_minus)
                log_x_plus = np.log(x_plus)
                four_log_center = 4 * log_center

                y_den = 2 * log_y_minus - four_log_center + 2 * log_y_plus
                x_den = 2 * log_x_minus - four_log_center + 2 * log_x_plus
                if np.isfinite(y_den) and y_den != 0.0:
                    dy_offset = float((log_y_minus - log_y_plus) / y_den)
                if np.isfinite(x_den) and x_den != 0.0:
                    dx_offset = float((log_x_minus - log_x_plus) / x_den)

        dy = -(float(peak_y) - float(max_y) + dy_offset)
        dx = -(float(peak_x) - float(max_x) + dx_offset)
        return dy, dx, avg_corr

    @staticmethod
    def _torch_extract_shifts(
        movie: np.ndarray,
        template: np.ndarray,
        max_shifts: Tuple[int, int],
        *,
        device: Union[str, torch.device] = "cpu",
        batch_size: int = 64,
        dtype: torch.dtype = torch.float64,
        eps: float = 1e-12,
    ) -> Tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(movie)
        if arr.ndim != 3:
            raise ValueError(f"Torch shift extraction expects (T, Y, X), got {arr.shape}")

        min_val = np.percentile(arr, 1)
        if min_val < -0.1:
            arr = arr - min_val
        arr = np.asarray(arr, dtype=np.float32)

        _, height, width = arr.shape
        max_y, max_x = int(max_shifts[0]), int(max_shifts[1])

        templ = np.asarray(template, dtype=np.float32)
        if np.percentile(templ, 8) < -0.1:
            templ = templ - np.percentile(templ, 1)
        templ = templ[max_y : height - max_y, max_x : width - max_x].astype(np.float32)
        if templ.size == 0:
            raise ValueError(
                f"Template crop is empty for frame shape {(height, width)} and max_shifts={max_shifts}"
            )

        torch_device = torch.device(device)
        torch_dtype = dtype
        template_t = torch.as_tensor(
            np.ascontiguousarray(templ),
            device=torch_device,
            dtype=torch_dtype,
        ).view(1, 1, templ.shape[0], templ.shape[1])
        ones = torch.ones_like(template_t)
        template_energy = torch.sum(template_t * template_t).clamp_min(float(eps))

        shifts = []
        xcorrs = []
        batch_size = max(1, int(batch_size))
        with torch.inference_mode():
            for start in range(0, arr.shape[0], batch_size):
                stop = min(start + batch_size, arr.shape[0])
                frames_t = torch.as_tensor(
                    np.ascontiguousarray(arr[start:stop]),
                    device=torch_device,
                    dtype=torch_dtype,
                ).unsqueeze(1)

                numerator = F.conv2d(frames_t, template_t)
                frame_energy = F.conv2d(frames_t * frames_t, ones).clamp_min(float(eps))
                res_t = numerator / torch.sqrt(frame_energy * template_energy)
                res_np = res_t[:, 0].detach().cpu().numpy()

                for res in res_np:
                    dy, dx, avg_corr = MotionCorrect._shift_from_match_response(
                        res,
                        max_shifts=max_shifts,
                    )
                    shifts.append([dy, dx])
                    xcorrs.append([avg_corr])

        return np.asarray(shifts, dtype=np.float32), np.asarray(xcorrs, dtype=np.float32)

    @staticmethod
    def _build_caiman_search_mask(
        height: int,
        width: int,
        max_shifts: Tuple[int, int],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        max_y, max_x = int(max_shifts[0]), int(max_shifts[1])
        mask = torch.ones((height, width), dtype=torch.bool, device=device)
        if max_y > 0:
            mask[max_y:-max_y, :] = False
        if max_x > 0:
            mask[:, max_x:-max_x] = False
        return mask

    @staticmethod
    def _upsampled_dft_batched_caiman(
        data: torch.Tensor,
        upsampled_region_size,
        upsample_factor,
        axis_offsets: torch.Tensor,
    ) -> torch.Tensor:
        if data.ndim != 3:
            raise ValueError(f"Batched local DFT expects (B, Y, X), got {tuple(data.shape)}")

        batch, height, width = data.shape
        if not hasattr(upsampled_region_size, "__iter__"):
            region_y = region_x = int(upsampled_region_size)
        else:
            if len(upsampled_region_size) != 2:
                raise ValueError("upsampled_region_size must match 2D data")
            region_y, region_x = int(upsampled_region_size[0]), int(upsampled_region_size[1])

        factor = float(upsample_factor)
        if axis_offsets.shape != (batch, 2):
            raise ValueError(f"axis_offsets must have shape {(batch, 2)}, got {tuple(axis_offsets.shape)}")

        real_dtype = torch.float64 if data.dtype == torch.complex128 else torch.float32
        device = data.device

        y_region = (
            torch.arange(region_y, device=device, dtype=real_dtype).unsqueeze(0)
            - axis_offsets[:, 0].to(device=device, dtype=real_dtype).unsqueeze(1)
        )
        x_region = (
            torch.arange(region_x, device=device, dtype=real_dtype).unsqueeze(0)
            - axis_offsets[:, 1].to(device=device, dtype=real_dtype).unsqueeze(1)
        )
        y_freq = torch.fft.ifftshift(torch.arange(height, device=device, dtype=real_dtype)) - torch.floor(
            torch.tensor(height // 2, device=device, dtype=real_dtype)
        )
        x_freq = torch.fft.ifftshift(torch.arange(width, device=device, dtype=real_dtype)) - torch.floor(
            torch.tensor(width // 2, device=device, dtype=real_dtype)
        )

        row_kernel = torch.exp(
            (-1j * 2.0 * torch.pi / (float(height) * factor))
            * y_region[:, :, None]
            * y_freq[None, None, :]
        ).to(dtype=data.dtype)
        col_kernel = torch.exp(
            (-1j * 2.0 * torch.pi / (float(width) * factor))
            * x_freq[None, :, None]
            * x_region[:, None, :]
        ).to(dtype=data.dtype)

        output = torch.einsum("brh,bhw->brw", row_kernel, data)
        return torch.einsum("brw,bwc->brc", output, col_kernel)

    @staticmethod
    def _torch_register_translation_batch(
        src_images: torch.Tensor,
        target_image: torch.Tensor,
        max_shifts: Tuple[int, int],
        *,
        upsample_factor: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if src_images.ndim != 3:
            raise ValueError(f"src_images must have shape (T, Y, X), got {tuple(src_images.shape)}")
        if target_image.ndim != 2:
            raise ValueError(f"target_image must have shape (Y, X), got {tuple(target_image.shape)}")
        if tuple(src_images.shape[-2:]) != tuple(target_image.shape):
            raise ValueError("src_images and target_image must have matching spatial shapes")

        upsample_factor = int(upsample_factor)
        if upsample_factor <= 0:
            raise ValueError("upsample_factor must be > 0")

        src_freq = torch.fft.fftn(src_images, dim=(-2, -1)) / float(src_images.shape[-2] * src_images.shape[-1])
        target_freq = torch.fft.fftn(target_image, dim=(-2, -1)) / float(target_image.numel())
        image_product = src_freq * torch.conj(target_freq).unsqueeze(0)
        cross_correlation = torch.fft.ifftn(image_product, dim=(-2, -1)) * float(target_image.numel())
        abs_corr = torch.abs(cross_correlation)

        height, width = int(src_images.shape[-2]), int(src_images.shape[-1])
        mask = MotionCorrect._build_caiman_search_mask(
            height,
            width,
            max_shifts,
            device=src_images.device,
        )
        masked_corr = abs_corr.masked_fill(~mask.unsqueeze(0), 0)
        peak_idx = torch.argmax(masked_corr.reshape(masked_corr.shape[0], -1), dim=1)
        maxima_y = torch.div(peak_idx, width, rounding_mode="floor")
        maxima_x = peak_idx % width

        shifts = torch.stack((maxima_y, maxima_x), dim=1).to(src_images.dtype)
        midpoints = torch.tensor(
            [np.fix(height // 2), np.fix(width // 2)],
            device=src_images.device,
            dtype=src_images.dtype,
        )
        shape = torch.tensor([height, width], device=src_images.device, dtype=src_images.dtype)
        shifts = torch.where(shifts > midpoints, shifts - shape, shifts)

        if upsample_factor > 1:
            factor = float(upsample_factor)
            shifts = torch.round(shifts * factor) / factor
            region_size = int(np.ceil(factor * 1.5))
            dftshift = float(np.fix(region_size / 2.0))
            normalization = float(src_freq[0].numel()) * factor**2
            sample_region_offset = dftshift - shifts * factor

            cross_correlation = MotionCorrect._upsampled_dft_batched_caiman(
                image_product.conj(),
                region_size,
                factor,
                sample_region_offset,
            ).conj()
            cross_correlation = cross_correlation / normalization

            refined_idx = torch.argmax(torch.abs(cross_correlation).reshape(src_images.shape[0], -1), dim=1)
            refined_y = torch.div(refined_idx, region_size, rounding_mode="floor")
            refined_x = refined_idx % region_size
            maxima = torch.stack((refined_y, refined_x), dim=1).to(src_images.dtype)
            maxima = maxima - dftshift
            shifts = shifts + maxima / factor

        if height == 1:
            shifts[:, 0] = 0
        if width == 1:
            shifts[:, 1] = 0

        correction_shifts = -shifts
        scores = masked_corr.reshape(masked_corr.shape[0], -1).amax(dim=1)
        return correction_shifts, scores

    def _estimate_caiman_rigid_shifts(
        self,
        template: torch.Tensor,
        *,
        return_scores: bool = False,
    ):
        if len(self.frame_shape) != 2:
            raise ValueError("template_strategy='caiman_rigid' currently supports 2D movies only")

        dtype = self.translation_kwargs.get("dtype", torch.float64)
        if not isinstance(dtype, torch.dtype):
            dtype = torch.float64

        template_t = torch.as_tensor(template, device=self.device, dtype=dtype)
        add = 0.0 if self.add_to_movie is None else float(self.add_to_movie)
        if add != 0.0:
            template_t = template_t + float(add)

        shifts_out = []
        scores_out = []
        for start in range(0, self.num_frames, self.frames_per_chunk):
            stop = min(start + self.frames_per_chunk, self.num_frames)
            frames = self.movie.read_frames(
                start,
                stop,
                as_tensor=True,
                dtype=np.float32,
            ).to(device=self.device, dtype=dtype)

            if self.gsig_filt is not None:
                reg_frames = self._caiman_high_pass_filter_space_tensor(
                    frames,
                    self.gsig_filt,
                    dtype=torch.float32,
                    device=self.device,
                ).to(dtype=dtype)
            else:
                reg_frames = frames

            if add != 0.0:
                reg_frames = reg_frames + float(add)

            shifts, scores = self._torch_register_translation_batch(
                reg_frames,
                template_t,
                self.max_shifts,
                upsample_factor=self.upsample_factor,
            )
            shifts_out.append(shifts.detach().cpu().to(torch.float32))
            if return_scores:
                scores_out.append(scores.detach().cpu().to(torch.float32))

        shifts_all = (
            torch.cat(shifts_out, dim=0)
            if shifts_out
            else torch.empty((0, 2), dtype=torch.float32)
        )
        if not return_scores:
            return shifts_all

        scores_all = (
            torch.cat(scores_out, dim=0)
            if scores_out
            else torch.empty((0,), dtype=torch.float32)
        )
        return shifts_all, scores_all

    @staticmethod
    def _apply_border_nan_policy_nchw(
        frames_nchw: torch.Tensor,
        shifts_chunk: torch.Tensor,
        border_nan: Union[bool, Literal["copy", "min", "nan"]],
        fill_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if border_nan is False:
            return frames_nchw

        out = frames_nchw.clone()
        height = int(out.shape[-2])
        width = int(out.shape[-1])
        shifts_np = shifts_chunk.detach().cpu().numpy()

        for i in range(out.shape[0]):
            dy = float(shifts_np[i, 0])
            dx = float(shifts_np[i, 1])

            top = int(np.ceil(max(0.0, dy)))
            bottom = int(np.floor(min(0.0, dy)))
            left = int(np.ceil(max(0.0, dx)))
            right = int(np.floor(min(0.0, dx)))

            if border_nan is True or border_nan == "nan":
                fill = torch.as_tensor(float("nan"), dtype=out.dtype, device=out.device)
            elif border_nan == "min":
                if fill_values is None:
                    fill = out[i].amin()
                else:
                    fill = fill_values[i]
            elif border_nan == "copy":
                fill = None
            else:
                raise ValueError(f"Unsupported border_nan policy {border_nan!r}")

            if top > 0:
                target_stop = min(top, height)
                if border_nan == "copy":
                    source = min(top, height - 1)
                    out[i, :, :target_stop, :] = out[i, :, source : source + 1, :]
                else:
                    out[i, :, :target_stop, :] = fill

            if bottom < 0:
                target_start = max(height + bottom, 0)
                if border_nan == "copy":
                    source = max(target_start - 1, 0)
                    out[i, :, target_start:, :] = out[i, :, source : source + 1, :]
                else:
                    out[i, :, target_start:, :] = fill

            if left > 0:
                target_stop = min(left, width)
                if border_nan == "copy":
                    source = min(left, width - 1)
                    out[i, :, :, :target_stop] = out[i, :, :, source : source + 1]
                else:
                    out[i, :, :, :target_stop] = fill

            if right < 0:
                target_start = max(width + right, 0)
                if border_nan == "copy":
                    source = max(target_start - 1, 0)
                    out[i, :, :, target_start:] = out[i, :, :, source : source + 1]
                else:
                    out[i, :, :, target_start:] = fill

        return out

    @staticmethod
    def _opencv_reflect_indices(indices: torch.Tensor, size: int) -> torch.Tensor:
        if size <= 0:
            raise ValueError("size must be > 0")
        period = 2 * int(size)
        wrapped = torch.remainder(indices, period)
        return torch.where(wrapped < size, wrapped, period - wrapped - 1).to(torch.long)

    @staticmethod
    def _opencv_cubic_weights(t: torch.Tensor) -> torch.Tensor:
        a = -0.75
        t2 = t * t
        t3 = t2 * t
        w0 = a * (t3 - 2.0 * t2 + t)
        w1 = (a + 2.0) * t3 - (a + 3.0) * t2 + 1.0
        w2 = -(a + 2.0) * t3 + (2.0 * a + 3.0) * t2 - a * t
        w3 = a * (t2 - t3)
        return torch.stack((w0, w1, w2, w3), dim=-1)

    @staticmethod
    def _opencv_interpolation_table_fraction(t: torch.Tensor) -> torch.Tensor:
        return torch.floor(t * 32.0 + 0.5) * (1.0 / 32.0)

    @staticmethod
    def _opencv_warp_affine_axis(
        size: int,
        shifts: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build one-dimensional OpenCV warpAffine source indices/weights.

        OpenCV's affine path does not use the same float maps as cv2.remap.
        It quantizes the inverse affine coordinate with AB_BITS=10, then
        stores interpolation-table coordinates with INTER_BITS=5. Reproducing
        that fixed-point path is necessary for arbitrary subpixel bootstrap
        shifts to match VolPy/CaImAn's cv2.warpAffine(..., INTER_CUBIC).
        """
        inter_bits = 5
        inter_tab_size = 1 << inter_bits
        ab_bits = 10
        ab_scale = 1 << ab_bits
        round_delta = ab_scale // inter_tab_size // 2

        # VolPy builds the affine matrix as float32 before passing it to OpenCV.
        shift_cv = shifts.to(device=device, dtype=torch.float32).to(torch.float64)
        base = torch.arange(size, device=device, dtype=torch.int64) * ab_scale
        offset = torch.round(-shift_cv * float(ab_scale)).to(torch.int64) + round_delta
        coord = torch.div(
            base.view(1, size) + offset.view(-1, 1),
            1 << (ab_bits - inter_bits),
            rounding_mode="floor",
        )
        coord_floor = torch.div(coord, inter_tab_size, rounding_mode="floor")
        frac_idx = coord - coord_floor * inter_tab_size

        offsets = torch.tensor([-1, 0, 1, 2], device=device, dtype=torch.long)
        indices = MotionCorrect._opencv_reflect_indices(
            coord_floor.unsqueeze(-1) + offsets,
            size,
        )
        fractions = frac_idx.to(dtype=dtype) * (1.0 / float(inter_tab_size))
        return indices, MotionCorrect._opencv_cubic_weights(fractions)

    def _apply_shifts_tensor_opencv_cubic(
        self,
        frames: torch.Tensor,
        shifts_chunk: torch.Tensor,
        *,
        add_to_movie: float = 0.0,
        border_nan: Union[bool, Literal["copy", "min", "nan"]] = False,
    ) -> torch.Tensor:
        if frames.shape[0] == 0:
            return frames.clone()

        frames_nchw, layout = self._frames_to_nchw(frames)
        input_dtype = frames_nchw.dtype

        if frames_nchw.dtype != torch.float32:
            frames_nchw = frames_nchw.to(dtype=torch.float32)

        work_nchw = frames_nchw
        add = float(add_to_movie)
        if add != 0.0:
            work_nchw = work_nchw + add

        n_frames, channels, height, width = work_nchw.shape
        device = work_nchw.device
        dtype = work_nchw.dtype

        shifts = shifts_chunk.to(device=device, dtype=dtype)
        dy = shifts[:, 0]
        dx = shifts[:, 1]

        y_idx, y_weights = self._opencv_warp_affine_axis(
            height,
            dy,
            device=device,
            dtype=dtype,
        )
        x_idx, x_weights = self._opencv_warp_affine_axis(
            width,
            dx,
            device=device,
            dtype=dtype,
        )

        vertical = torch.zeros_like(work_nchw)
        for ky in range(4):
            gathered = torch.gather(
                work_nchw,
                2,
                y_idx[:, :, ky].view(n_frames, 1, height, 1).expand(
                    n_frames,
                    channels,
                    height,
                    width,
                ),
            )
            vertical = vertical + gathered * y_weights[:, :, ky].view(n_frames, 1, height, 1)

        corrected_nchw = torch.zeros_like(work_nchw)
        for kx in range(4):
            gathered = torch.gather(
                vertical,
                3,
                x_idx[:, :, kx].view(n_frames, 1, 1, width).expand(
                    n_frames,
                    channels,
                    height,
                    width,
                ),
            )
            corrected_nchw = corrected_nchw + gathered * x_weights[:, :, kx].view(n_frames, 1, 1, width)

        if self.clip_interpolated:
            mins = work_nchw.amin(dim=(1, 2, 3), keepdim=True)
            maxs = work_nchw.amax(dim=(1, 2, 3), keepdim=True)
            corrected_nchw = torch.minimum(torch.maximum(corrected_nchw, mins), maxs)
        else:
            mins = work_nchw.amin(dim=(1, 2, 3), keepdim=True)

        if border_nan is not False:
            corrected_nchw = self._apply_border_nan_policy_nchw(
                corrected_nchw,
                shifts_chunk,
                border_nan,
                fill_values=mins,
            )

        if add != 0.0:
            corrected_nchw = corrected_nchw - add

        corrected = self._nchw_to_frames(corrected_nchw, layout)
        if torch.is_floating_point(frames) and corrected.dtype != input_dtype:
            corrected = corrected.to(dtype=input_dtype)
        return corrected

    def _torch_apply_shifts_array(
        self,
        movie: np.ndarray,
        shifts: np.ndarray,
        *,
        add_to_movie: float = 0.0,
        border_nan: Union[bool, Literal["copy", "min", "nan"]] = False,
    ) -> np.ndarray:
        arr = np.asarray(movie, dtype=np.float32)
        if arr.ndim not in (3, 4):
            raise ValueError(f"Torch shift application expects (T, Y, X[, C]), got {arr.shape}")

        frames = torch.as_tensor(np.ascontiguousarray(arr), device=self.device)
        shifts_t = torch.as_tensor(np.asarray(shifts, dtype=np.float32), device=self.device)
        corrected = self._apply_shifts_tensor_grid_sample(
            frames,
            shifts_t,
            add_to_movie=float(add_to_movie),
            border_nan=border_nan,
        )
        return corrected.detach().cpu().numpy().astype(np.float32, copy=False)

    def _motion_correct_array(
        self,
        movie: np.ndarray,
        template: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        arr = np.asarray(movie, dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError(f"CaImAn compatibility expects (T, Y, X), got {arr.shape}")
        reg_arr = (
            self._caiman_high_pass_filter_space(arr, self.gsig_filt)
            if self.gsig_filt is not None
            else arr
        )

        def extract_shifts(mov: np.ndarray, templ: np.ndarray):
            return self._torch_extract_shifts(
                mov,
                templ,
                self.max_shifts,
                device=self.device,
            )

        def apply_shifts(mov: np.ndarray, shift_values: np.ndarray):
            return self._torch_apply_shifts_array(mov, shift_values)

        if template is None:
            num_frames_template = 1e8 / float(arr.shape[1] * arr.shape[2])
            frames_to_skip = int(np.maximum(1, arr.shape[0] / num_frames_template))

            submov = reg_arr[::frames_to_skip].copy()
            templ = self._caiman_bin_median_array(submov)
            shifts, _ = extract_shifts(submov, templ)
            submov = apply_shifts(submov, shifts)
            template = self._caiman_bin_median_array(submov)

            full = reg_arr.copy()
            shifts, _ = extract_shifts(full, template)
            full = apply_shifts(full, shifts)
            template = self._caiman_bin_median_array(full)
        else:
            template = np.asarray(template, dtype=np.float32) - np.percentile(template, 8)

        shifts, xcorrs = extract_shifts(reg_arr, template)
        corrected = apply_shifts(reg_arr, shifts)
        return corrected, shifts, xcorrs, np.asarray(template, dtype=np.float32)

    def _application_add_to_movie(self) -> float:
        if self.gsig_filt is not None:
            return 0.0
        return 0.0 if self.add_to_movie is None else float(self.add_to_movie)

    def _build_caiman_rigid_template(self) -> torch.Tensor:
        if len(self.frame_shape) != 2:
            raise ValueError("template_strategy='caiman_rigid' currently supports 2D movies only")
        if self.high_pass_filter_size is not None:
            raise ValueError("template_strategy='caiman_rigid' does not support high_pass_filter_size")

        step = int(self.num_frames) // 50
        corrected_slicer_step = step + 1
        sample = self.movie.read_frames(
            0,
            self.num_frames,
            step=corrected_slicer_step,
            as_tensor=False,
            dtype=np.float32,
        )
        corrected, _, _, _ = self._motion_correct_array(sample)
        template = self._caiman_bin_median_array(corrected)
        if self.add_to_movie is None:
            self.add_to_movie = -self._estimate_movie_min()
        return torch.as_tensor(template, device=self.device, dtype=torch.float32)

    @torch.inference_mode()
    def build_template(self, force: bool = False):
        """
        Build and cache the motion-correction template using Template.
        """
        if self.template is not None and not force:
            return self.template

        if self.template_strategy == "caiman_rigid":
            self.template_builder = None
            self.template = self._build_caiman_rigid_template()
            return self.template

        kwargs = dict(self.template_kwargs)
        kwargs.setdefault("movie", self.movie)
        kwargs.setdefault("device", self.device)
        kwargs.setdefault("high_pass_filter_size", self.high_pass_filter_size)

        self.template_builder = Template(**kwargs)
        self.template = self.template_builder.build()
        if self.add_to_movie is None:
            self.add_to_movie = 0.0
        return self.template

    @torch.inference_mode()
    def estimate_shifts(self, force: bool = False, return_scores: bool = False):
        """
        Estimate and cache per-frame correction shifts.
        """
        if not force and self.shifts is not None:
            if return_scores:
                if self.scores is not None:
                    return self.shifts, self.scores
            else:
                return self.shifts

        import time

        s = time.time()
        template = self.build_template(force=force)
        e = time.time()
        print("Template build took {:.2f} seconds".format(e - s))

        if self.template_strategy == "caiman_rigid":
            s = time.time()
            if return_scores:
                shifts, scores = self._estimate_caiman_rigid_shifts(
                    template,
                    return_scores=True,
                )
                self.shifts = self._normalize_shifts(shifts, self.num_frames)
                self.scores = scores.detach().cpu()
                return self.shifts, self.scores

            shifts = self._estimate_caiman_rigid_shifts(
                template,
                return_scores=False,
            )
            self.shifts = self._normalize_shifts(shifts, self.num_frames)
            e = time.time()
            print("Translation took {:.2f} seconds".format(e - s))
            return self.shifts

        kwargs = dict(self.translation_kwargs)
        kwargs.setdefault("movie", self.movie)
        kwargs.setdefault("template", template)
        kwargs.setdefault("max_shifts", self.max_shifts)
        kwargs.setdefault("frames_per_chunk", self.frames_per_chunk)
        kwargs.setdefault("device", self.device)
        kwargs.setdefault("high_pass_filter_size", self.high_pass_filter_size)
        kwargs.setdefault("upsample_factor", self.upsample_factor)

        # CaImAn/VolPy rigid correction uses unwindowed, unnormalized FFT
        # cross-correlation and refines shifts to 1/10 pixel by default.
        kwargs.setdefault("use_hann", False)
        kwargs.setdefault("center", False)
        kwargs.setdefault("normalization", "none")
        kwargs.setdefault("subpixel_method", "dft")
        kwargs.setdefault("add_to_movie", 0.0 if self.add_to_movie is None else self.add_to_movie)

        self.translation_estimator = Translation(**kwargs)

        s = time.time()
        if return_scores:
            shifts, scores = self.translation_estimator(return_scores=True)
            self.shifts = self._normalize_shifts(shifts, self.num_frames)
            self.scores = scores.detach().cpu()
            return self.shifts, self.scores

        shifts = self.translation_estimator(return_scores=False)
        self.shifts = self._normalize_shifts(shifts, self.num_frames)
        e = time.time()
        print("Translation took {:.2f} seconds".format(e - s))
        return self.shifts

    @torch.inference_mode()
    def prepare(self, force: bool = False, return_scores: bool = False):
        """
        Convenience method to build template and estimate shifts.
        """
        template = self.build_template(force=force)
        if return_scores:
            shifts, scores = self.estimate_shifts(force=force, return_scores=True)
            return template, shifts, scores
        shifts = self.estimate_shifts(force=force, return_scores=False)
        return template, shifts

    def _pad_frames_for_correction(
        self,
        frames: torch.Tensor,
        pad_y: int,
        pad_x: int,
    ) -> torch.Tensor:
        """
        Convert frames to NCHW, pad spatially, and return padded NCHW tensor.
        """
        if frames.ndim == 3:
            # (T, H, W) -> (T, 1, H, W)
            frames_nchw = frames.unsqueeze(1)
        elif frames.ndim == 4:
            # (T, H, W, C) -> (T, C, H, W)
            frames_nchw = frames.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Unexpected frame batch shape {tuple(frames.shape)}")

        if pad_y == 0 and pad_x == 0:
            return frames_nchw

        H = int(frames_nchw.shape[-2])
        W = int(frames_nchw.shape[-1])

        if self.padding_mode == "reflection":
            if (pad_y > 0 and H <= 1) or (pad_x > 0 and W <= 1):
                raise ValueError(
                    "Reflection padding requires padded spatial dimensions > 1, "
                    f"got pad_y={pad_y}, pad_x={pad_x}, H={H}, W={W}"
                )
            if pad_y >= H or pad_x >= W:
                raise ValueError(
                    "Reflection padding requires padding size < input size in each padded dimension, "
                    f"got pad_y={pad_y}, pad_x={pad_x}, H={H}, W={W}"
                )
            padded = F.pad(
                frames_nchw,
                (pad_x, pad_x, pad_y, pad_y),
                mode="reflect",
            )
        elif self.padding_mode == "border":
            # F.pad names copied-border padding "replicate"; grid_sample calls it "border".
            padded = F.pad(
                frames_nchw,
                (pad_x, pad_x, pad_y, pad_y),
                mode="replicate",
            )
        elif self.padding_mode == "zeros":
            padded = F.pad(
                frames_nchw,
                (pad_x, pad_x, pad_y, pad_y),
                mode="constant",
                value=0,
            )
        else:
            raise ValueError(f"Unsupported padding_mode {self.padding_mode!r}")

        return padded

    @staticmethod
    def _frames_to_nchw(frames: torch.Tensor) -> Tuple[torch.Tensor, str]:
        if frames.ndim == 3:
            return frames.unsqueeze(1), "tyx"
        if frames.ndim == 4:
            return frames.permute(0, 3, 1, 2), "tyxc"
        raise ValueError(f"Unexpected frame batch shape {tuple(frames.shape)}")

    @staticmethod
    def _nchw_to_frames(frames_nchw: torch.Tensor, layout: str) -> torch.Tensor:
        if layout == "tyx":
            return frames_nchw[:, 0, :, :]
        if layout == "tyxc":
            return frames_nchw.permute(0, 2, 3, 1)
        raise ValueError(f"Unknown frame layout {layout!r}")

    def _base_grid(
        self,
        height: int,
        width: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        key = (height, width, device.type, device.index, dtype, self.align_corners)
        cached = self._base_grid_cache.get(key)
        if cached is not None:
            return cached

        if self.align_corners:
            ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
            xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        else:
            ys = (torch.arange(height, device=device, dtype=dtype) + 0.5) * (2.0 / height) - 1.0
            xs = (torch.arange(width, device=device, dtype=dtype) + 0.5) * (2.0 / width) - 1.0

        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack((xx, yy), dim=-1).unsqueeze(0)
        self._base_grid_cache[key] = grid
        return grid

    def _subpixel_grid_for_shifts(
        self,
        shifts_chunk: torch.Tensor,
        height: int,
        width: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        shifts = shifts_chunk.to(device=device, dtype=dtype)
        dy = shifts[:, 0]
        dx = shifts[:, 1]

        base = self._base_grid(height, width, device=device, dtype=dtype)

        if self.align_corners:
            if height <= 1 or width <= 1:
                raise ValueError(
                    "align_corners=True subpixel correction requires spatial dimensions > 1"
                )
            offset_y = -2.0 * dy / float(height - 1)
            offset_x = -2.0 * dx / float(width - 1)
        else:
            offset_y = -2.0 * dy / float(height)
            offset_x = -2.0 * dx / float(width)

        offsets = torch.stack((offset_x, offset_y), dim=1).view(-1, 1, 1, 2)
        return base + offsets

    def _correct_chunk_integer(
        self,
        frames: torch.Tensor,
        shifts_chunk: torch.Tensor,
    ) -> torch.Tensor:
        rounded = torch.round(shifts_chunk)
        if not torch.allclose(shifts_chunk.to(rounded.dtype), rounded):
            raise ValueError("integer interpolation requires integer-valued shifts")

        shifts_int = rounded.to(torch.int64)

        H = int(frames.shape[1])
        W = int(frames.shape[2])

        if frames.shape[0] == 0:
            return frames.clone()

        pad_y = int(torch.abs(shifts_int[:, 0]).max().item())
        pad_x = int(torch.abs(shifts_int[:, 1]).max().item())

        if pad_y == 0 and pad_x == 0:
            return frames.clone()

        padded = self._pad_frames_for_correction(frames, pad_y=pad_y, pad_x=pad_x)

        corrected_nchw = torch.empty(
            (padded.shape[0], padded.shape[1], H, W),
            dtype=padded.dtype,
            device=padded.device,
        )

        for i in range(frames.shape[0]):
            dy = int(shifts_int[i, 0].item())
            dx = int(shifts_int[i, 1].item())

            y0 = pad_y - dy
            x0 = pad_x - dx
            y1 = y0 + H
            x1 = x0 + W

            corrected_nchw[i] = padded[i, :, y0:y1, x0:x1]

        if frames.ndim == 3:
            return corrected_nchw[:, 0, :, :]
        return corrected_nchw.permute(0, 2, 3, 1)

    def _correct_chunk_grid_sample(
        self,
        frames: torch.Tensor,
        shifts_chunk: torch.Tensor,
    ) -> torch.Tensor:
        return self._apply_shifts_tensor_grid_sample(
            frames,
            shifts_chunk,
            add_to_movie=self._application_add_to_movie(),
            border_nan=self.border_nan,
        )

    def _apply_shifts_tensor_grid_sample(
        self,
        frames: torch.Tensor,
        shifts_chunk: torch.Tensor,
        *,
        add_to_movie: float = 0.0,
        border_nan: Union[bool, Literal["copy", "min", "nan"]] = False,
    ) -> torch.Tensor:
        if self.interpolation == "opencv_cubic":
            return self._apply_shifts_tensor_opencv_cubic(
                frames,
                shifts_chunk,
                add_to_movie=add_to_movie,
                border_nan=border_nan,
            )

        if frames.shape[0] == 0:
            return frames.clone()

        frames_nchw, layout = self._frames_to_nchw(frames)
        input_dtype = frames_nchw.dtype

        if not torch.is_floating_point(frames_nchw):
            frames_nchw = frames_nchw.to(dtype=torch.float32)
        elif frames_nchw.dtype not in (torch.float32, torch.float64):
            frames_nchw = frames_nchw.to(dtype=torch.float32)

        work_nchw = frames_nchw
        add = float(add_to_movie)
        if add != 0.0:
            work_nchw = work_nchw + add

        H = int(frames_nchw.shape[-2])
        W = int(frames_nchw.shape[-1])

        grid = self._subpixel_grid_for_shifts(
            shifts_chunk,
            H,
            W,
            device=work_nchw.device,
            dtype=work_nchw.dtype,
        )

        corrected_nchw = F.grid_sample(
            work_nchw,
            grid,
            mode=self.interpolation,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )

        if self.clip_interpolated:
            mins = work_nchw.amin(dim=(-2, -1), keepdim=True)
            maxs = work_nchw.amax(dim=(-2, -1), keepdim=True)
            corrected_nchw = torch.minimum(torch.maximum(corrected_nchw, mins), maxs)

        if border_nan is not False:
            corrected_nchw = self._apply_border_nan_policy_nchw(
                corrected_nchw,
                shifts_chunk,
                border_nan,
                fill_values=work_nchw.amin(dim=(-2, -1), keepdim=True),
            )
        elif self.padding_mode == "border" and self.copy_border_strips:
            corrected_nchw = self._copy_shifted_border_strips(corrected_nchw, shifts_chunk)

        if add != 0.0:
            corrected_nchw = corrected_nchw - add

        corrected = self._nchw_to_frames(corrected_nchw, layout)
        if torch.is_floating_point(frames) and corrected.dtype != input_dtype:
            corrected = corrected.to(dtype=input_dtype)
        return corrected

    @staticmethod
    def _copy_shifted_border_strips(
        frames_nchw: torch.Tensor,
        shifts_chunk: torch.Tensor,
    ) -> torch.Tensor:
        """
        Match CaImAn/VolPy border_nan='copy' after subpixel interpolation.
        """
        out = frames_nchw.clone()
        height = int(out.shape[-2])
        width = int(out.shape[-1])

        shifts_np = shifts_chunk.detach().cpu().numpy()

        for i in range(out.shape[0]):
            dy = float(shifts_np[i, 0])
            dx = float(shifts_np[i, 1])

            top = int(np.ceil(max(0.0, dy)))
            bottom = int(np.floor(min(0.0, dy)))
            left = int(np.ceil(max(0.0, dx)))
            right = int(np.floor(min(0.0, dx)))

            if top > 0:
                target_stop = min(top, height)
                source = min(top, height - 1)
                out[i, :, :target_stop, :] = out[i, :, source : source + 1, :]

            if bottom < 0:
                target_start = max(height + bottom, 0)
                source = max(target_start - 1, 0)
                out[i, :, target_start:, :] = out[i, :, source : source + 1, :]

            if left > 0:
                target_stop = min(left, width)
                source = min(left, width - 1)
                out[i, :, :, :target_stop] = out[i, :, :, source : source + 1]

            if right < 0:
                target_start = max(width + right, 0)
                source = max(target_start - 1, 0)
                out[i, :, :, target_start:] = out[i, :, :, source : source + 1]

        return out

    @torch.inference_mode()
    def _correct_chunk(
        self,
        frames: torch.Tensor,
        shifts_chunk: torch.Tensor,
    ) -> torch.Tensor:
        if frames.ndim not in (3, 4):
            raise ValueError(f"Unexpected frame batch shape {tuple(frames.shape)}")

        if shifts_chunk.ndim != 2 or shifts_chunk.shape[1] != 2:
            raise ValueError(
                f"shifts_chunk must have shape (T, 2), got {tuple(shifts_chunk.shape)}"
            )

        if shifts_chunk.shape[0] != frames.shape[0]:
            raise ValueError(
                f"shifts_chunk length ({shifts_chunk.shape[0]}) does not match "
                f"number of frames ({frames.shape[0]})"
            )

        if frames.shape[0] == 0:
            return frames.clone()

        if torch.count_nonzero(shifts_chunk).item() == 0:
            return frames.clone()

        if self.interpolation == "integer":
            return self._correct_chunk_integer(frames, shifts_chunk)

        return self._correct_chunk_grid_sample(frames, shifts_chunk)

    @torch.inference_mode()
    def save(
        self,
        out_h5_path: Union[str, Path],
        dataset: str = "movie",
        overwrite: bool = False,
        compression: Optional[str] = None,
        compression_opts: Optional[int] = None,
        output_dtype: Optional[Union[str, np.dtype]] = None,
        save_shifts_dataset: Optional[str] = "shifts",
        save_template_dataset: Optional[str] = "template",
        save_scores_dataset: Optional[str] = None,
        extra_attrs: Optional[dict] = None,
    ):
        """
        Build template/shifts if needed, apply shifts to the full movie,
        and save corrected frames to disk.

        Returns
        -------
        Movie
            A Movie instance opened on the corrected dataset.
        """
        import time

        out_h5_path = str(out_h5_path)

        need_scores = save_scores_dataset is not None
        if need_scores:
            shifts, scores = self.estimate_shifts(return_scores=True)
        else:
            shifts = self.estimate_shifts(return_scores=False)
            scores = None

        if output_dtype is None:
            output_dtype = self.movie.dtype
        output_dtype = np.dtype(output_dtype)

        attrs = {
            "motion_corrected": True,
            "source_h5_path": getattr(self.movie, "h5_path", ""),
            "source_dataset": getattr(self.movie, "dataset", ""),
            "frames_per_chunk": self.frames_per_chunk,
            "max_shifts": str(self.max_shifts),
            "upsample_factor": self.upsample_factor,
            "template_strategy": self.template_strategy,
            "motion_backend": "torch",
            "border_nan": str(self.border_nan),
            "add_to_movie": 0.0 if self.add_to_movie is None else float(self.add_to_movie),
            "gsig_filt": "" if self.gsig_filt is None else str(self.gsig_filt),
            "interpolation": self.interpolation,
            "padding_mode": self.padding_mode,
            "copy_border_strips": self.copy_border_strips,
            "align_corners": self.align_corners,
            "clip_interpolated": self.clip_interpolated,
            "template_built_inside_class": True,
            "translation_estimated_inside_class": True,
        }

        if self.template_builder is not None:
            try:
                attrs.update(
                    {f"template_{k}": v for k, v in self.template_builder.summary().items()}
                )
            except Exception:
                pass

        if extra_attrs:
            attrs.update(extra_attrs)

        Movie.create_empty(
            h5_path=out_h5_path,
            frame_shape=self.movie.frame_shape,
            dtype=output_dtype,
            dataset=dataset,
            overwrite=overwrite,
            chunk_frames=self.frames_per_chunk,
            compression=compression,
            compression_opts=compression_opts,
            attrs=attrs,
        )

        torch_output_dtype = torch.from_numpy(np.empty((), dtype=output_dtype)).dtype

        s = time.time()

        for start in range(0, self.num_frames, self.frames_per_chunk):
            stop = min(start + self.frames_per_chunk, self.num_frames)

            frames = self.movie.read_frames(
                start,
                stop,
                as_tensor=True,
            ).to(self.device)

            shifts_chunk = shifts[start:stop].to(self.device)
            corrected = self._correct_chunk(frames, shifts_chunk)

            if corrected.dtype != torch_output_dtype:
                corrected = corrected.to(dtype=torch_output_dtype)

            Movie.append_tensor(
                h5_path=out_h5_path,
                data=corrected.cpu(),
                dataset=dataset,
            )

        e = time.time()
        print("Motion Correction took {:.2f} seconds".format(e - s))

        with h5py.File(out_h5_path, "a") as f:
            if save_shifts_dataset is not None:
                if save_shifts_dataset in f:
                    del f[save_shifts_dataset]
                f.create_dataset(
                    save_shifts_dataset,
                    data=shifts.cpu().numpy(),
                    compression=compression,
                    compression_opts=compression_opts,
                )

            if save_template_dataset is not None:
                template_np = self._to_numpy(self.template)
                if save_template_dataset in f:
                    del f[save_template_dataset]
                f.create_dataset(
                    save_template_dataset,
                    data=template_np,
                    compression=compression,
                    compression_opts=compression_opts,
                )

            if save_scores_dataset is not None and scores is not None:
                if save_scores_dataset in f:
                    del f[save_scores_dataset]
                f.create_dataset(
                    save_scores_dataset,
                    data=scores.cpu().numpy(),
                    compression=compression,
                    compression_opts=compression_opts,
                )

        return Movie(out_h5_path, dataset=dataset, mode="r")

    def summary(self) -> dict:
        out = {
            "num_frames": self.num_frames,
            "frame_shape": self.frame_shape,
            "frames_per_chunk": self.frames_per_chunk,
            "device": str(self.device),
            "max_shifts": self.max_shifts,
            "upsample_factor": self.upsample_factor,
            "interpolation": self.interpolation,
            "padding_mode": self.padding_mode,
            "motion_backend": "torch",
            "copy_border_strips": self.copy_border_strips,
            "has_template": self.template is not None,
            "has_shifts": self.shifts is not None,
            "has_scores": self.scores is not None,
        }
        return out

    def __call__(self, *args, **kwargs):
        return self.save(*args, **kwargs)
