from __future__ import annotations

import math
from typing import Tuple

import torch


class Summary:
    """
    Build two 2D summary images from a Movie:
      1) Mean image
      2) 8-neighbor correlation image after percentile-baseline removal

    Assumptions:
    - movie is an already-created Movie object
    - movie shape is (T, Y, X)
    - build() returns two tensors of shape (Y, X)

    Memory behavior:
    - reads one non-overlapping temporal window at a time
    - baseline is computed by sorting spatial tiles across frames
    - correlation image is computed in 2 passes to keep memory low
    """

    _OFFSETS = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),            (0, 1),
        (1, -1),  (1, 0),   (1, 1),
    )

    def __init__(
        self,
        movie,
        window_size: int = 100,
        baseline_percentile: float = 8.0,
        baseline_tile_shape: Tuple[int, int] = (128, 128),
        compute_dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
        eps: float = 1e-6,
    ) -> None:
        if movie is None:
            raise ValueError("movie must be a Movie instance, not None")

        if not hasattr(movie, "shape") or not hasattr(movie, "read_frames"):
            raise TypeError("movie must be a Movie-like object with shape and read_frames()")

        if len(movie.shape) != 3:
            raise ValueError(
                f"Summary expects a grayscale movie with shape (T, Y, X), got {movie.shape}"
            )

        if window_size <= 0:
            raise ValueError("window_size must be > 0")

        if not (0.0 <= baseline_percentile <= 100.0):
            raise ValueError("baseline_percentile must be in [0, 100]")

        tile_y, tile_x = baseline_tile_shape
        if tile_y <= 0 or tile_x <= 0:
            raise ValueError("baseline_tile_shape values must both be > 0")

        self.movie = movie
        self.window_size = int(window_size)
        self.baseline_percentile = float(baseline_percentile)
        self.baseline_tile_shape = (int(tile_y), int(tile_x))
        self.compute_dtype = compute_dtype
        self.device = torch.device(device)
        self.eps = float(eps)

        self.num_frames = int(movie.shape[0])
        self.height = int(movie.shape[1])
        self.width = int(movie.shape[2])

    @torch.inference_mode()
    def build(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        mean_image : torch.Tensor
            Shape (Y, X), normalized to zero mean and unit std over pixels

        corr_image : torch.Tensor
            Shape (Y, X), average temporal correlation with 8 neighbors
            after percentile-baseline removal, then normalized over pixels
        """
        t_total = float(self.num_frames)
        h, w = self.height, self.width

        # ---------- pass 1 ----------
        # raw movie mean image
        raw_sum = torch.zeros((h, w), dtype=self.compute_dtype, device=self.device)

        # high-pass movie stats for z-scoring each pixel over time
        hp_sum = torch.zeros((h, w), dtype=self.compute_dtype, device=self.device)
        hp_sumsq = torch.zeros((h, w), dtype=self.compute_dtype, device=self.device)

        for window in self._iter_windows():
            raw_sum += window.sum(dim=0)

            baseline = self._percentile_baseline(window)
            hp = window - baseline.unsqueeze(0)

            hp_sum += hp.sum(dim=0)
            hp_sumsq += hp.square().sum(dim=0)

        n_windows = (self.num_frames + self.window_size - 1) // self.window_size
        mean_image = raw_sum / t_total
        mean_image = self._normalize_image(mean_image)

        hp_mean = hp_sum / t_total
        hp_var = hp_sumsq / t_total - hp_mean.square()
        hp_var.clamp_(min=0.0)

        hp_std = hp_var.sqrt()
        hp_std_safe = hp_std.clamp_min(self.eps)
        valid_std = (hp_std >= self.eps).to(self.compute_dtype)

        # ---------- pass 2 ----------
        # accumulate average neighbor correlation
        corr_acc = torch.zeros((h, w), dtype=self.compute_dtype, device=self.device)
        neighbor_count = self._neighbor_count(h, w)

        for window in self._iter_windows():
            baseline = self._percentile_baseline(window)
            hp = window - baseline.unsqueeze(0)

            z = (hp - hp_mean.unsqueeze(0)) / hp_std_safe.unsqueeze(0)
            z = z * valid_std.unsqueeze(0)

            for dy, dx in self._OFFSETS:
                cy, cx, ny, nx = self._pair_slices(dy, dx, h, w)
                corr_acc[cy, cx] += (z[:, cy, cx] * z[:, ny, nx]).sum(dim=0)

        corr_image = corr_acc / t_total
        corr_image = corr_image / neighbor_count.clamp_min(1.0)
        corr_image = self._normalize_image(corr_image)

        return mean_image, corr_image

    def _iter_windows(self):
        """
        Iterate over non-overlapping temporal windows:
        [0:window_size], [window_size:2*window_size], ...
        """
        for start in range(0, self.num_frames, self.window_size):
            stop = min(start + self.window_size, self.num_frames)
            window = self.movie.read_frames(start, stop, as_tensor=True)
            yield window.to(device=self.device, dtype=self.compute_dtype)

    def _percentile_baseline(self, window: torch.Tensor) -> torch.Tensor:
        """
        Compute per-pixel percentile baseline over one temporal window.

        To keep peak memory lower, the sort is done in spatial tiles:
            tile shape = (T, tile_y, tile_x)
        and sorted only along T.
        """
        n_frames, h, w = window.shape
        k = self._percentile_index(n_frames)

        baseline = torch.empty((h, w), dtype=window.dtype, device=window.device)
        tile_h, tile_w = self.baseline_tile_shape

        for y0 in range(0, h, tile_h):
            y1 = min(y0 + tile_h, h)
            for x0 in range(0, w, tile_w):
                x1 = min(x0 + tile_w, w)

                tile = window[:, y0:y1, x0:x1]
                sorted_tile = tile.sort(dim=0).values
                baseline[y0:y1, x0:x1] = sorted_tile[k]

        return baseline

    def _percentile_index(self, n_frames: int) -> int:
        """
        Order-statistic index for the requested percentile.
        Uses the lower order statistic:
            floor(p * (n_frames - 1))
        """
        if n_frames <= 0:
            raise ValueError("window must contain at least one frame")

        idx = int(math.floor((self.baseline_percentile / 100.0) * (n_frames - 1)))
        return max(0, min(n_frames - 1, idx))

    def _neighbor_count(self, h: int, w: int) -> torch.Tensor:
        """
        Number of valid neighbors for each pixel.
        Interior pixels get 8, borders get fewer.
        """
        count = torch.zeros((h, w), dtype=self.compute_dtype, device=self.device)
        for dy, dx in self._OFFSETS:
            cy, cx, _, _ = self._pair_slices(dy, dx, h, w)
            count[cy, cx] += 1.0
        return count

    @staticmethod
    def _axis_pair(delta: int, size: int):
        """
        Returns slices for:
        - center pixels
        - corresponding neighbor pixels
        """
        if delta > 0:
            return slice(0, size - delta), slice(delta, size)
        if delta < 0:
            return slice(-delta, size), slice(0, size + delta)
        return slice(0, size), slice(0, size)

    @classmethod
    def _pair_slices(cls, dy: int, dx: int, h: int, w: int):
        cy, ny = cls._axis_pair(dy, h)
        cx, nx = cls._axis_pair(dx, w)
        return cy, cx, ny, nx

    def _normalize_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Normalize a 2D image by subtracting its pixel mean
        and dividing by its pixel std.
        """
        mean = image.mean()
        std = image.std(unbiased=False).clamp_min(self.eps)
        return (image - mean) / std
