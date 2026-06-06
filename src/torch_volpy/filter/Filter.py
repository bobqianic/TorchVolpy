from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F


class Filter:
    """
    Gaussian-derived spatial high-pass filtering for movies using PyTorch.

    Supported input shapes:
        - (T, X, Y)
        - (T, X, Y, C)

    Notes:
        - Filtering is applied per frame
        - For multi-channel data, each channel is filtered independently
        - The frame axis is NOT blurred
        - This matches the kernel logic from the original OpenCV/Caiman code:
          build a 2D Gaussian, keep only the central support, and subtract the
          mean over that support so the kernel becomes zero-mean / high-pass-like
    """

    @staticmethod
    def _caiman_highpass_kernel_2d(
        size: int,
        sigma: Optional[float] = None,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        if size <= 0:
            raise ValueError("size must be > 0")
        if size % 2 == 0:
            raise ValueError("size must be odd")

        if sigma is None:
            # Same common fallback used in your original helper
            sigma = 0.3 * ((size - 1) * 0.5 - 1) + 0.8

        coords = torch.arange(size, dtype=dtype, device=device) - (size - 1) / 2
        g1 = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g1 = g1 / g1.sum()

        ker2d = torch.outer(g1, g1)
        ker2d = ker2d / ker2d.sum()

        # Match the original logic:
        # nz = np.nonzero(ker2D >= ker2D[:, 0].max())
        # zz = np.nonzero(ker2D < ker2D[:, 0].max())
        # ker2D[nz] -= ker2D[nz].mean()
        # ker2D[zz] = 0
        threshold = ker2d[:, 0].max()
        center_mask = ker2d >= threshold
        outer_mask = ~center_mask

        ker2d = ker2d.clone()
        ker2d[center_mask] -= ker2d[center_mask].mean()
        ker2d[outer_mask] = 0

        return ker2d

    @staticmethod
    def _to_tensor(
        movie: Union[np.ndarray, torch.Tensor],
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        if isinstance(movie, np.ndarray):
            movie = torch.from_numpy(movie)

        if not isinstance(movie, torch.Tensor):
            raise TypeError("movie must be a numpy array or torch tensor")

        if dtype is not None:
            movie = movie.to(dtype=dtype)
        if device is not None:
            movie = movie.to(device=device)

        return movie

    @classmethod
    def hp_gaussian(
        cls,
        size: int,
        movie: Union[np.ndarray, torch.Tensor],
        sigma: Optional[float] = None,
        padding_mode: str = "reflect",
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        """
        Apply the same Gaussian-derived high-pass kernel as the original code.

        Args:
            size:
                Odd kernel size, e.g. 3, 5, 7
            movie:
                Shape (T, X, Y) or (T, X, Y, C)
            sigma:
                Gaussian sigma. If None, inferred from size
            padding_mode:
                Passed to torch.nn.functional.pad
            dtype:
                Internal compute dtype
            device:
                CPU or CUDA device

        Returns:
            Filtered movie as a torch.Tensor with the same shape as input
        """
        x = cls._to_tensor(movie, dtype=dtype, device=device)

        if x.ndim not in (3, 4):
            raise ValueError(
                f"movie must have shape (T, X, Y) or (T, X, Y, C), got {tuple(x.shape)}"
            )

        original_shape = x.shape

        # Convert to conv2d format: (N, C, H, W)
        if x.ndim == 3:
            # (T, X, Y) -> (T, 1, X, Y)
            x = x.unsqueeze(1)
            out_mode = "3d"
        else:
            # (T, X, Y, C) -> (T, C, X, Y)
            x = x.permute(0, 3, 1, 2).contiguous()
            out_mode = "4d"

        channels = x.shape[1]

        kernel = cls._caiman_highpass_kernel_2d(
            size=size,
            sigma=sigma,
            dtype=x.dtype,
            device=x.device,
        )

        # Depthwise convolution: same kernel for each channel
        weight = kernel.view(1, 1, size, size).repeat(channels, 1, 1, 1)

        pad = size // 2
        x = F.pad(x, (pad, pad, pad, pad), mode=padding_mode)
        y = F.conv2d(x, weight, bias=None, stride=1, padding=0, groups=channels)

        # Convert back to original layout
        if out_mode == "3d":
            y = y.squeeze(1)  # (T, X, Y)
        else:
            y = y.permute(0, 2, 3, 1).contiguous()  # (T, X, Y, C)

        if y.shape != original_shape:
            raise RuntimeError(
                f"Output shape mismatch: expected {original_shape}, got {tuple(y.shape)}"
            )

        return y