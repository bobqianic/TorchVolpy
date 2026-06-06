from .Summary import Summary

try:
    from .Cellpose import Cellpose
except ImportError as exc:
    _cellpose_import_error = exc

    class Cellpose:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Cellpose support requires the package's segmentation dependencies. "
                "Reinstall torch-volpy so its base dependencies are installed."
            ) from _cellpose_import_error


__all__ = ["Cellpose", "Summary"]
