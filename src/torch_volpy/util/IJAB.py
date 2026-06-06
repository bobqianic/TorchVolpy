import numpy as np
from typing import Optional, Tuple


class IJAB:
    """ImageJ-like Auto Brightness/Contrast for NumPy arrays."""

    def __init__(self, clicks: int = 1, bins: int = 256):
        self.clicks = max(1, int(clicks))
        self.bins = bins

    import numpy as np

    @classmethod
    def imagej_fp32_to_uint8(cls, arr: np.ndarray,
                             display_min: float | None = None,
                             display_max: float | None = None) -> np.ndarray:
        """
      :contentReference[oaicite:0]{index=0}style display scaling.

        ImageJ behavior for 32-bit -> 8-bit display:
          - Linearly maps display_min..display_max to 0..255
          - Clips values below display_min to 0
          - Clips values above display_max to 255
          - Rounds with +0.5 before integer conversion
          - If display_min/display_max are not given, uses image min/max
            while ignoring NaN for range calculation

        Parameters
        ----------
        arr : np.ndarray
            Input image as a NumPy array.
        display_min : float | None
            Lower bound of display range. If None, computed from data.
        display_max : float | None
            Upper bound of display range. If None, computed from data.

        Returns
        -------
        np.ndarray
            uint8 image with the same shape as input.
        """
        a = np.asarray(arr, dtype=np.float32)

        if display_min is None or display_max is None:
            valid = ~np.isnan(a)
            if not np.any(valid):
                return np.zeros(a.shape, dtype=np.uint8)
            if display_min is None:
                display_min = float(np.min(a[valid]))
            if display_max is None:
                display_max = float(np.max(a[valid]))

        # Constant-image guard: ImageJ's Java path gets awkward here;
        # returning zeros is the practical display-safe behavior.
        if not np.isfinite(display_min) or not np.isfinite(display_max) or display_max <= display_min:
            out = np.zeros_like(a, dtype=np.uint8)
            out[a > display_max] = 255
            return out

        scale = 255.0 / (display_max - display_min)

        # ImageJ-like mapping:
        # value = pixel - min
        # if value < 0: value = 0
        # ivalue = int(value * scale + 0.5)
        # if ivalue > 255: ivalue = 255
        scaled = a.astype(np.float32) - np.float32(display_min)
        scaled = np.maximum(scaled, 0.0)
        scaled = np.floor(scaled * scale + 0.5)

        # Emulate Java's NaN->0 cast behavior safely in NumPy
        scaled = np.where(np.isnan(scaled), 0.0, scaled)
        scaled = np.clip(scaled, 0.0, 255.0)

        return scaled.astype(np.uint8)

    @staticmethod
    def _threshold_divisor(clicks: int) -> int:
        auto_threshold = 0
        for _ in range(clicks):
            if auto_threshold < 10:
                auto_threshold = 5000
            else:
                auto_threshold //= 2
        return auto_threshold

    def display_range(
        self,
        arr: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[float, float]:
        a = np.asarray(arr)
        if a.size == 0:
            raise ValueError("arr must not be empty")

        if mask is not None:
            m = np.asarray(mask, dtype=bool)
            if m.shape != a.shape:
                raise ValueError("mask must have the same shape as arr")
            values = a[m]
        else:
            values = a.ravel()

        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError("arr contains no finite values")

        data_min = float(values.min())
        data_max = float(values.max())

        if data_min == data_max:
            return data_min, data_max

        hist, _ = np.histogram(values, bins=self.bins, range=(data_min, data_max))

        pixel_count = int(values.size)
        limit = pixel_count // 10
        threshold = pixel_count // self._threshold_divisor(self.clicks)

        hist_work = hist.copy()
        hist_work[hist_work > limit] = 0

        hmin = 0
        while hmin < self.bins - 1 and not (hist_work[hmin] > threshold):
            hmin += 1

        hmax = self.bins - 1
        while hmax > 0 and not (hist_work[hmax] > threshold):
            hmax -= 1

        if hmax < hmin:
            return data_min, data_max

        bin_size = (data_max - data_min) / self.bins
        display_min = data_min + hmin * bin_size
        display_max = data_min + hmax * bin_size

        if display_min == display_max:
            return data_min, data_max

        return display_min, display_max

    def apply(
        self,
        arr: np.ndarray,
        mask: Optional[np.ndarray] = None,
        output_dtype=np.float32,
        out_range: Optional[Tuple[float, float]] = None,
        return_limits: bool = False,
    ):
        a = np.asarray(arr)
        display_min, display_max = self.display_range(a, mask=mask)

        if display_max <= display_min:
            adjusted = np.zeros_like(a, dtype=output_dtype)
            return (adjusted, (display_min, display_max)) if return_limits else adjusted

        clipped = np.clip(a, display_min, display_max)
        scaled = (clipped - display_min) / (display_max - display_min)

        dtype = np.dtype(output_dtype)
        if out_range is None:
            if np.issubdtype(dtype, np.integer):
                info = np.iinfo(dtype)
                lo, hi = info.min, info.max
            else:
                lo, hi = 0.0, 1.0
        else:
            lo, hi = out_range

        adjusted = scaled * (hi - lo) + lo

        if np.issubdtype(dtype, np.integer):
            adjusted = np.rint(adjusted).astype(dtype)
        else:
            adjusted = adjusted.astype(dtype)

        return (adjusted, (display_min, display_max)) if return_limits else adjusted

    def __call__(self, arr: np.ndarray, **kwargs):
        return self.apply(arr, **kwargs)