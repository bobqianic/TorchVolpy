"""PyTorch tools for voltage imaging movie processing and signal extraction."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torch-volpy")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["__version__"]
