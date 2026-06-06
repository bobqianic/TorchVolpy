from __future__ import annotations

from typing import Literal, Optional, Union

import numpy as np
import torch

from ..filter import Filter


class Template:
    """
    Build a motion-correction template from a lazily readable Movie.

    Assumptions
    -----------
    - axis 0 is time
    - all remaining axes are preserved as frame dimensions
      (e.g. Y,X or Y,X,C)
    - "binmedian" is implemented as:
        1) split frames into temporal bins
        2) mean within each bin
        3) median across bin means

    Memory behavior
    ---------------
    - never loads the whole movie into memory
    - reads only one frame block at a time from Movie

    Optional filtering
    ------------------
    - if high_pass_filter_size is not None, spatial high-pass Gaussian filtering is applied
      to each chunk/bin immediately after reading frames
    - filtering happens before:
        1) minimum estimation
        2) bin-mean / bin-median template calculation
    """

    def __init__(
        self,
        movie,
        window: int = 10,
        min_value: Optional[float] = None,
        estimate_min_frames: int = 400,
        negative_floor: float = -10.0,
        keep_partial_last_bin: bool = False,
        exclude_nans: bool = True,
        binning: Literal["consecutive", "caiman"] = "consecutive",
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = "cpu",
        high_pass_filter_size: Optional[int] = None,
    ) -> None:
        if window <= 0:
            raise ValueError("window must be > 0")
        if estimate_min_frames <= 0:
            raise ValueError("estimate_min_frames must be > 0")

        if high_pass_filter_size is not None:
            high_pass_filter_size = int(high_pass_filter_size)
            if high_pass_filter_size <= 0:
                raise ValueError("high_pass_filter_size must be > 0")
            if high_pass_filter_size % 2 == 0:
                raise ValueError("high_pass_filter_size must be odd")

        self.movie = movie
        self.window = int(window)
        self._user_min_value = None if min_value is None else float(min_value)
        self.estimate_min_frames = int(estimate_min_frames)
        self.negative_floor = float(negative_floor)
        self.keep_partial_last_bin = bool(keep_partial_last_bin)
        self.exclude_nans = bool(exclude_nans)
        if binning not in ("consecutive", "caiman"):
            raise ValueError("binning must be 'consecutive' or 'caiman'")
        self.binning = binning
        self.dtype = dtype

        # Interpret this as OUTPUT device, not compute device.
        self.device = torch.device(device)

        # Gaussian filter options
        self.high_pass_filter_size = high_pass_filter_size

        self.template_raw: Optional[torch.Tensor] = None
        self.template: Optional[torch.Tensor] = None
        self.min_value: Optional[float] = None
        self.add_to_movie: float = 0.0
        self.template_p01_raw: Optional[float] = None
        self.template_p01: Optional[float] = None
        self.num_bins_used: int = 0
        self.num_frames_used: int = 0

    @staticmethod
    def _finite_only(x: torch.Tensor) -> torch.Tensor:
        return x[torch.isfinite(x)]

    @torch.inference_mode()
    def _maybe_filter(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Apply spatial high-pass Gaussian filtering if enabled.

        Accepts:
            - (T, X, Y)
            - (T, X, Y, C)
            - (X, Y)
            - (X, Y, C)

        Returns the same shape as the input.
        """
        if self.high_pass_filter_size is None:
            return frames

        squeeze_time = False
        if frames.ndim == len(self.movie.frame_shape):
            # single frame without explicit time dimension
            frames = frames.unsqueeze(0)
            squeeze_time = True

        filtered = Filter.hp_gaussian(
            size=self.high_pass_filter_size,
            movie=frames,
            sigma=None,
            padding_mode="reflect",
            dtype=torch.float32,
            device=self.device,
        )

        if squeeze_time:
            filtered = filtered.squeeze(0)

        return filtered

    @torch.inference_mode()
    def estimate_min(self) -> float:
        if self._user_min_value is not None:
            self.min_value = float(self._user_min_value)
            return self.min_value

        n = min(int(self.movie.num_frames), self.estimate_min_frames)
        if n == 0:
            raise ValueError("Movie has no frames")

        min_val = float("inf")
        chunk = min(self.window, n)

        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            frames = self.movie.read_frames(
                start,
                stop,
                as_tensor=True,
                dtype=np.float32,
            )

            frames = self._maybe_filter(frames)

            if self.exclude_nans:
                finite = self._finite_only(frames)
                if finite.numel() == 0:
                    continue
                cur = torch.min(finite)
            else:
                cur = torch.min(frames)

            min_val = min(min_val, float(cur.item()))

        if not np.isfinite(min_val):
            raise ValueError("Could not estimate a finite minimum from the movie")

        self.min_value = min_val
        return min_val

    @torch.inference_mode()
    def _frame_block_mean(self, frames: torch.Tensor) -> torch.Tensor:
        # keep on CPU; only cast dtype
        frames = frames.to(dtype=self.dtype)
        if self.exclude_nans:
            return torch.nanmean(frames, dim=0)
        return torch.mean(frames, dim=0)

    @torch.inference_mode()
    def compute_binmedian(self) -> torch.Tensor:
        T = int(self.movie.num_frames)
        if T == 0:
            raise ValueError("Movie has no frames")

        bin_ranges = []
        if self.binning == "caiman":
            window = min(self.window, T)
            num_windows = T // window
            num_frames = num_windows * window
            if num_windows > 0:
                for window_index in range(num_windows):
                    bin_ranges.append((window_index, num_frames, num_windows))
        elif T < self.window:
            bin_ranges.append((0, T, 1))
        else:
            full_frames = (T // self.window) * self.window
            for start in range(0, full_frames, self.window):
                bin_ranges.append((start, start + self.window, 1))
            if self.keep_partial_last_bin and full_frames < T:
                bin_ranges.append((full_frames, T, 1))

        if not bin_ranges:
            raise RuntimeError("No bins were generated")

        stacked = torch.empty(
            (len(bin_ranges), *self.movie.frame_shape),
            dtype=self.dtype,
            device="cpu",
        )

        for b, (start, stop, step) in enumerate(bin_ranges):
            frames = self.movie.read_frames(
                start,
                stop,
                step=step,
                as_tensor=True,
                dtype=np.float32,
            )

            if frames.ndim == len(self.movie.frame_shape):
                frames = frames.unsqueeze(0)

            frames = self._maybe_filter(frames)

            mean_frame = self._frame_block_mean(frames)
            stacked[b].copy_(mean_frame)

            del frames, mean_frame

        if self.exclude_nans:
            raw = torch.nanmedian(stacked, dim=0).values
        else:
            raw = torch.median(stacked, dim=0).values

        self.num_bins_used = len(bin_ranges)
        self.num_frames_used = sum(len(range(start, stop, step)) for start, stop, step in bin_ranges)
        self.template_raw = raw
        return raw

    @torch.inference_mode()
    def _safe_quantile(self, x: torch.Tensor, q: float) -> float:
        flat = x.reshape(-1)
        if self.exclude_nans:
            flat = self._finite_only(flat)

        if flat.numel() == 0:
            raise ValueError("Template contains no finite values")

        return float(torch.quantile(flat, q).item())

    @torch.inference_mode()
    def build(self) -> torch.Tensor:
        min_value = self.estimate_min()
        raw = self.compute_binmedian()  # CPU

        add_to_movie = max(0.0, -float(min_value))

        raw_p01 = self._safe_quantile(raw, 0.01)
        if raw_p01 + add_to_movie < self.negative_floor:
            add_to_movie += self.negative_floor - (raw_p01 + add_to_movie)

        # compute summary stats on CPU first
        template_cpu = raw + add_to_movie
        template_p01 = self._safe_quantile(template_cpu, 0.01)

        # move only final output
        template = template_cpu.to(self.device, dtype=self.dtype)

        self.add_to_movie = float(add_to_movie)
        self.template_p01_raw = raw_p01
        self.template_p01 = template_p01
        self.template = template
        return template

    def __call__(self) -> torch.Tensor:
        return self.build()
