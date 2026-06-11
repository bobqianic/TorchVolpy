from __future__ import annotations

import gc
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import tifffile
import torch

from torch_volpy.extraction import ALI, Spikepursuit
from torch_volpy.model import Cellpose, Summary
from torch_volpy.motion import MotionCorrect
from torch_volpy.movie import Movie

PathLike = Union[str, Path]
Rect = Tuple[float, float, float, float]
Point = Tuple[float, float]


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


def _fit_max_shifts_to_frame_shape(
    max_shifts: Union[int, Tuple[int, int]],
    frame_shape: Sequence[int],
) -> Tuple[int, int]:
    max_y, max_x = _normalize_max_shifts(max_shifts)
    shape = tuple(int(v) for v in frame_shape)
    if len(shape) < 2:
        return max_y, max_x

    height, width = shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError(f"Movie spatial dimensions must be positive, got {(height, width)}")

    limit_y = max(0, (height - 1) // 2)
    limit_x = max(0, (width - 1) // 2)
    return min(max_y, limit_y), min(max_x, limit_x)


class TiffMovie:
    """Small Movie-compatible adapter for TIFF stacks used by the GUI."""

    def __init__(self, path: PathLike) -> None:
        self.path = str(path)
        self._tif = None
        self._series = None
        try:
            self._data = tifffile.memmap(self.path)
        except Exception:
            self._data = None
            self._tif = tifffile.TiffFile(self.path)
            self._series = self._tif.series[0]

        if self._data is not None:
            if self._data.ndim == 2:
                self._data = self._data[np.newaxis, ...]

            if self._data.ndim < 3:
                raise ValueError(f"TIFF movie must have at least 3 dimensions, got {self._data.shape}")

            self.shape = tuple(int(v) for v in self._data.shape)
            self.dtype = np.dtype(self._data.dtype)
        else:
            assert self._series is not None
            shape = tuple(int(v) for v in self._series.shape)
            if len(shape) == 2:
                shape = (1, *shape)
            if len(shape) < 3:
                raise ValueError(f"TIFF movie must have at least 3 dimensions, got {shape}")
            self.shape = shape
            self.dtype = np.dtype(self._series.dtype)

    @property
    def num_frames(self) -> int:
        """Number of frames in the TIFF stack."""
        return int(self.shape[0])

    @property
    def frame_shape(self) -> Tuple[int, ...]:
        """Shape of one TIFF frame, excluding the time axis."""
        return tuple(self.shape[1:])

    def close(self) -> None:
        """Close the TIFF file or memory map backing this adapter."""
        data = getattr(self, "_data", None)
        if isinstance(data, np.memmap):
            mmap = getattr(data, "_mmap", None)
            if mmap is not None:
                mmap.close()
        self._data = None
        tif = getattr(self, "_tif", None)
        if tif is not None:
            tif.close()
        self._tif = None
        self._series = None

    def _read_lazy_time_index(self, time_index):
        if self._tif is None:
            raise RuntimeError("TIFF file is closed")

        if time_index is Ellipsis:
            time_index = slice(None)

        if isinstance(time_index, (int, np.integer)):
            time_index = int(time_index)
            if time_index < 0:
                time_index += self.num_frames
            return self._tif.asarray(key=time_index)

        return self._tif.asarray(key=time_index)

    def _read_lazy(self, index):
        if index is Ellipsis:
            return self._read_lazy_time_index(slice(None))

        if not isinstance(index, tuple):
            return self._read_lazy_time_index(index)

        if len(index) == 0:
            return self._read_lazy_time_index(slice(None))

        if index[0] is Ellipsis:
            return self._read_lazy_time_index(slice(None))[index]

        time_index = index[0]
        rest_index = index[1:]
        arr = self._read_lazy_time_index(time_index)

        if not rest_index:
            return arr

        if isinstance(time_index, (int, np.integer)):
            return arr[rest_index]

        return arr[(slice(None), *rest_index)]

    def read(
        self,
        index=...,
        as_tensor: bool = True,
        dtype: Optional[Union[np.dtype, str]] = None,
        device: Optional[Union[str, torch.device]] = None,
        copy: bool = False,
    ):
        """Read a TIFF frame or frame range as a numpy array or tensor."""
        if self._data is None:
            arr = self._read_lazy(index)
        else:
            arr = self._data[index]
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        if not as_tensor:
            return np.array(arr, copy=True) if copy else arr

        if copy:
            arr = np.array(arr, copy=True, order="C")
        else:
            arr = np.asarray(arr, order="C")
        tensor = torch.from_numpy(arr)
        if device is not None:
            tensor = tensor.to(device)
        return tensor

    def read_frames(
        self,
        start: int,
        stop: Optional[int] = None,
        step: int = 1,
        as_tensor: bool = True,
        dtype: Optional[Union[np.dtype, str]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """Read a contiguous frame range from the TIFF stack."""
        return self.read(
            slice(start, stop, step),
            as_tensor=as_tensor,
            dtype=dtype,
            device=device,
        )


class ChannelMovie:
    """Expose one channel of a movie as a grayscale `(T, Y, X)` movie."""

    def __init__(self, movie, channel: Optional[int] = None) -> None:
        shape = tuple(movie.shape)
        if len(shape) == 3:
            self.movie = movie
            self.channel = None
            self.shape = shape
        elif len(shape) == 4:
            self.movie = movie
            self.channel = 0 if channel is None else int(channel)
            self.shape = (int(shape[0]), int(shape[1]), int(shape[2]))
        else:
            raise ValueError(f"Unsupported movie shape: {shape}")
        self.dtype = movie.dtype

    def read_frames(
        self,
        start: int,
        stop: Optional[int] = None,
        step: int = 1,
        as_tensor: bool = True,
        dtype: Optional[Union[np.dtype, str]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """Read grayscale frames from the selected channel."""
        if self.channel is None:
            index = slice(start, stop, step)
        else:
            index = (slice(start, stop, step), slice(None), slice(None), self.channel)
        return self.movie.read(index, as_tensor=as_tensor, dtype=dtype, device=device)


def open_movie(path: PathLike, dataset: str = "movie"):
    """Open an HDF5 Movie or TIFF stack with a Movie-compatible interface."""

    suffix = Path(path).suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        return Movie(path, dataset=dataset, mode="r")
    if suffix in {".tif", ".tiff"}:
        return TiffMovie(path)
    raise ValueError(f"Unsupported movie file type: {suffix}")


def default_h5_path_for_movie(path: PathLike) -> Path:
    """Return the default HDF5 path corresponding to a source movie path."""
    path = Path(path)
    if path.suffix.lower() in {".h5", ".hdf5"}:
        return path
    return path.with_suffix(".h5")


def default_corrected_h5_path(path: PathLike) -> Path:
    """Return the default corrected HDF5 output path for a source movie."""
    path = Path(path)
    if path.name.startswith("corrected_") and path.suffix.lower() in {".h5", ".hdf5"}:
        return path
    return path.with_name(f"corrected_{path.stem}.h5")


def convert_tiff_to_h5(
    tiff_path: PathLike,
    h5_path: Optional[PathLike] = None,
    *,
    dataset: str = "movie",
    overwrite: bool = False,
    chunk_frames: int = 16,
    compression: Optional[str] = None,
    compression_opts: Optional[int] = None,
    progress_callback=None,
) -> Path:
    """Convert a TIFF stack to HDF5 with optional percent progress callback."""

    tiff_path = Path(tiff_path)
    h5_path = Path(h5_path) if h5_path is not None else default_h5_path_for_movie(tiff_path)

    if h5_path.exists() and not overwrite:
        if progress_callback is not None:
            progress_callback(100, f"Using existing HDF5: {h5_path.name}")
        return h5_path

    with tifffile.TiffFile(str(tiff_path)) as tif:
        n_pages = len(tif.pages)
        if n_pages == 0:
            raise ValueError(f"No TIFF pages found in {tiff_path}")

        first = tif.pages[0].asarray()
        frame_shape = first.shape
        frame_dtype = first.dtype

        file_mode = "a" if h5_path.exists() else "w"
        with h5py.File(str(h5_path), file_mode) as f:
            if dataset in f:
                del f[dataset]

            dset = f.create_dataset(
                dataset,
                shape=(n_pages, *frame_shape),
                maxshape=(None, *frame_shape),
                dtype=frame_dtype,
                chunks=Movie._normalize_chunks((n_pages, *frame_shape), chunk_frames),
                compression=compression,
                compression_opts=compression_opts,
            )
            dset.attrs["source_tiff"] = str(tiff_path)
            dset[0] = first
            if progress_callback is not None:
                progress_callback(int(round(100 / n_pages)), "Converting TIFF to HDF5")

            batch = []
            batch_start = 1
            for page_index in range(1, n_pages):
                batch.append(tif.pages[page_index].asarray())
                if len(batch) >= chunk_frames:
                    dset[batch_start : batch_start + len(batch)] = np.stack(batch, axis=0)
                    batch_start += len(batch)
                    batch.clear()
                    if progress_callback is not None:
                        progress_callback(
                            int(round(100 * batch_start / n_pages)),
                            "Converting TIFF to HDF5",
                        )

            if batch:
                dset[batch_start : batch_start + len(batch)] = np.stack(batch, axis=0)

        if progress_callback is not None:
            progress_callback(100, "TIFF conversion complete")

    return h5_path


def is_motion_corrected_h5(path: PathLike, dataset: str = "movie") -> bool:
    """Return whether an HDF5 dataset is marked as motion corrected."""
    path = Path(path)
    if path.suffix.lower() not in {".h5", ".hdf5"} or not path.exists():
        return False
    try:
        with h5py.File(path, "r") as f:
            if dataset not in f:
                return False
            return bool(f[dataset].attrs.get("motion_corrected", False))
    except OSError:
        return False


@torch.inference_mode()
def motion_correct_movie(
    movie,
    out_h5_path: Optional[PathLike] = None,
    *,
    dataset: str = "movie",
    overwrite: bool = False,
    max_shifts: Union[int, Tuple[int, int]] = (15, 15),
    frames_per_chunk: int = 256,
    device: Union[str, torch.device] = "cpu",
    progress_callback=None,
    source_path: Optional[PathLike] = None,
    source_dataset: Optional[str] = None,
    close_movie: bool = False,
    compression: Optional[str] = None,
    compression_opts: Optional[int] = None,
) -> Path:
    """Motion-correct a Movie-like object into a corrected HDF5 path."""

    if out_h5_path is None:
        inferred_path = source_path or getattr(movie, "h5_path", None) or getattr(movie, "path", None)
        if inferred_path is None:
            raise ValueError("out_h5_path is required when the movie has no source path")
        out_h5_path = default_corrected_h5_path(inferred_path)
    out_h5_path = Path(out_h5_path)

    if out_h5_path.exists() and not overwrite and is_motion_corrected_h5(out_h5_path, dataset=dataset):
        if progress_callback is not None:
            progress_callback(100, f"Using existing corrected movie: {out_h5_path.name}")
        return out_h5_path

    max_shifts = _fit_max_shifts_to_frame_shape(max_shifts, movie.frame_shape)

    try:
        mc = MotionCorrect(
            movie=movie,
            max_shifts=max_shifts,
            frames_per_chunk=frames_per_chunk,
            device=device,
        )
        if progress_callback is not None:
            progress_callback(10, "Building motion template")
        mc.build_template()

        if progress_callback is not None:
            progress_callback(35, "Estimating motion shifts")
        shifts = mc.estimate_shifts()

        if progress_callback is not None:
            progress_callback(60, "Applying motion correction")

        output_dtype = np.dtype(movie.dtype)
        torch_output_dtype = torch.from_numpy(np.empty((), dtype=output_dtype)).dtype
        num_frames = int(movie.num_frames)
        frame_shape = tuple(movie.frame_shape)
        source_path_value = source_path or getattr(movie, "h5_path", None) or getattr(movie, "path", "")
        source_dataset_value = source_dataset
        if source_dataset_value is None:
            source_dataset_value = getattr(movie, "dataset", "")

        file_mode = "a" if out_h5_path.exists() else "w"
        with h5py.File(out_h5_path, file_mode) as f:
            if dataset in f:
                if not (overwrite or out_h5_path.exists()):
                    raise FileExistsError(
                        f"Dataset '{dataset}' already exists in {out_h5_path}. "
                        "Use overwrite=True to replace it."
                    )
                del f[dataset]

            dset = f.create_dataset(
                dataset,
                shape=(num_frames, *frame_shape),
                maxshape=(None, *frame_shape),
                dtype=output_dtype,
                chunks=Movie._normalize_chunks((num_frames, *frame_shape), frames_per_chunk),
                compression=compression,
                compression_opts=compression_opts,
            )
            dset.attrs["motion_corrected"] = False
            dset.attrs["source_h5_path"] = str(source_path_value)
            dset.attrs["source_dataset"] = str(source_dataset_value)
            dset.attrs["frames_per_chunk"] = int(frames_per_chunk)
            dset.attrs["max_shifts"] = str(max_shifts)
            dset.attrs["upsample_factor"] = int(mc.upsample_factor)
            dset.attrs["interpolation"] = str(mc.interpolation)
            dset.attrs["padding_mode"] = str(mc.padding_mode)
            dset.attrs["align_corners"] = bool(mc.align_corners)
            dset.attrs["clip_interpolated"] = bool(mc.clip_interpolated)

            for start in range(0, num_frames, int(frames_per_chunk)):
                stop = min(start + int(frames_per_chunk), num_frames)
                frames = movie.read_frames(start, stop, as_tensor=True).to(mc.device)
                shifts_chunk = shifts[start:stop].to(mc.device)
                corrected = mc._correct_chunk(frames, shifts_chunk)
                if corrected.dtype != torch_output_dtype:
                    corrected = corrected.to(dtype=torch_output_dtype)
                dset[start:stop] = corrected.detach().cpu().contiguous().numpy()
                if progress_callback is not None:
                    progress_callback(
                        60 + int(round(35 * stop / max(1, num_frames))),
                        f"Applying motion correction ({stop}/{num_frames})",
                    )
                del frames, shifts_chunk, corrected

            dset.attrs["motion_corrected"] = True
            if "shifts" in f:
                del f["shifts"]
            f.create_dataset(
                "shifts",
                data=shifts.cpu().numpy(),
                compression=compression,
                compression_opts=compression_opts,
            )
            if "template" in f:
                del f["template"]
            f.create_dataset(
                "template",
                data=mc.template.detach().cpu().numpy(),
                compression=compression,
                compression_opts=compression_opts,
            )

        if progress_callback is not None:
            progress_callback(100, "Motion correction complete")
    finally:
        if close_movie and hasattr(movie, "close"):
            movie.close()
        release_torch_memory(device)

    return out_h5_path


@torch.inference_mode()
def motion_correct_h5(
    h5_path: PathLike,
    out_h5_path: Optional[PathLike] = None,
    *,
    dataset: str = "movie",
    overwrite: bool = False,
    max_shifts: Union[int, Tuple[int, int]] = (15, 15),
    frames_per_chunk: int = 256,
    device: Union[str, torch.device] = "cpu",
    progress_callback=None,
    compression: Optional[str] = None,
    compression_opts: Optional[int] = None,
) -> Path:
    """Motion-correct an HDF5 movie into a corrected HDF5 path."""

    h5_path = Path(h5_path)
    movie = Movie(h5_path, dataset=dataset, mode="r")
    return motion_correct_movie(
        movie,
        out_h5_path=out_h5_path if out_h5_path is not None else default_corrected_h5_path(h5_path),
        dataset=dataset,
        overwrite=overwrite,
        max_shifts=max_shifts,
        frames_per_chunk=frames_per_chunk,
        device=device,
        progress_callback=progress_callback,
        source_path=h5_path,
        source_dataset=dataset,
        close_movie=True,
        compression=compression,
        compression_opts=compression_opts,
    )


def _frame_index(movie, frame_index: int, channel: Optional[int]):
    shape = tuple(movie.shape)
    if len(shape) == 3:
        return (int(frame_index), slice(None), slice(None))
    if len(shape) == 4:
        if channel is None:
            channel = 0
        return (int(frame_index), slice(None), slice(None), int(channel))
    raise ValueError(f"Unsupported movie shape: {shape}")


def read_display_frame(movie, frame_index: int, channel: Optional[int] = None) -> np.ndarray:
    """Read one movie frame as a 2D numpy array for display."""

    frame = movie.read(_frame_index(movie, frame_index, channel), as_tensor=False, copy=True)
    frame = np.asarray(frame)
    if frame.ndim == 3:
        frame = frame.mean(axis=-1)
    if frame.ndim != 2:
        raise ValueError(f"Display frame must be 2D after channel selection, got {frame.shape}")
    return frame


def normalize_to_uint8(
    image: np.ndarray,
    low_percentile: float = 1.0,
    high_percentile: float = 99.5,
) -> np.ndarray:
    """Robustly normalize a 2D image into uint8 for Qt display."""

    arr = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype=np.uint8)

    values = arr[finite]
    lo = float(np.percentile(values, low_percentile))
    hi = float(np.percentile(values, high_percentile))
    if hi <= lo:
        hi = float(values.max())
        lo = float(values.min())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)

    out = (arr - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0)
    out[~finite] = 0.0
    return (out * 255.0).astype(np.uint8)


def release_torch_memory(
    device: Optional[Union[str, torch.device]] = None,
    *,
    flush_cuda_cache: bool = True,
) -> None:
    """Release Python references and flush CUDA allocator caches when available."""

    gc.collect()
    if not torch.cuda.is_available():
        return
    if device is not None and not str(device).startswith("cuda"):
        return
    if not flush_cuda_cache:
        return
    try:
        torch.cuda.synchronize()
    except RuntimeError:
        pass
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except RuntimeError:
        pass


@torch.inference_mode()
def build_summary_image(
    movie,
    *,
    channel: Optional[int] = None,
    window_size: int = 1000,
    baseline_percentile: float = 8.0,
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """Build the Cellpose input image used in the project tests: `[mean, mean, corr]`."""

    grayscale_movie = ChannelMovie(movie, channel=channel)
    summary_builder = Summary(
        grayscale_movie,
        window_size=int(window_size),
        baseline_percentile=float(baseline_percentile),
        device=device,
    )
    mean, corr = summary_builder.build()
    return torch.stack([mean, mean, corr])


@torch.inference_mode()
def build_cellpose_rois(
    movie,
    *,
    model_path: PathLike,
    channel: Optional[int] = None,
    summary_window_size: int = 1000,
    baseline_percentile: float = 8.0,
    device: Union[str, torch.device] = "cpu",
    gpu: bool = True,
    save_to_disk: bool = False,
    save_dir: Optional[PathLike] = None,
) -> Tuple[np.ndarray, torch.Tensor]:
    """Build a summary image and run Cellpose segmentation to produce labeled ROIs."""

    summary_img = None
    cellpose_builder = None
    rois = None
    try:
        summary_img = build_summary_image(
            movie,
            channel=channel,
            window_size=summary_window_size,
            baseline_percentile=baseline_percentile,
            device=device,
        )
        cellpose_builder = Cellpose(model_path=model_path, gpu=bool(gpu), device=device)
        rois = cellpose_builder.build(
            summary_img,
            save_to_disk=bool(save_to_disk),
            save_dir=save_dir,
        )
        rois_np = np.asarray(rois.detach().cpu().numpy(), dtype=np.int32)
        if rois_np.ndim != 2:
            raise ValueError(f"Cellpose returned a non-2D ROI mask with shape {rois_np.shape}")
        summary_cpu = summary_img.detach().cpu()
        return rois_np, summary_cpu
    finally:
        if cellpose_builder is not None and hasattr(cellpose_builder, "model"):
            try:
                del cellpose_builder.model
            except AttributeError:
                pass
        del cellpose_builder
        del rois
        del summary_img
        release_torch_memory(device)


def rectangle_to_mask(
    rect: Rect,
    shape: Tuple[int, int],
    label: int = 1,
) -> np.ndarray:
    """Convert an image-space rectangle `(x0, y0, x1, y1)` to a labeled mask."""

    h, w = int(shape[0]), int(shape[1])
    x0, y0, x1, y1 = rect
    xa, xb = sorted((int(np.floor(x0)), int(np.ceil(x1))))
    ya, yb = sorted((int(np.floor(y0)), int(np.ceil(y1))))
    xa = max(0, min(w, xa))
    xb = max(0, min(w, xb))
    ya = max(0, min(h, ya))
    yb = max(0, min(h, yb))

    mask = np.zeros((h, w), dtype=np.int32)
    if xb > xa and yb > ya:
        mask[ya:yb, xa:xb] = int(label)
    return mask


def polygon_to_mask(
    points: Sequence[Point],
    shape: Tuple[int, int],
    label: int = 1,
) -> np.ndarray:
    """Rasterize a polygon into a labeled mask using pixel-center tests."""

    h, w = int(shape[0]), int(shape[1])
    pts = np.asarray(points, dtype=np.float64)
    mask = np.zeros((h, w), dtype=np.int32)
    if pts.shape[0] < 3:
        return mask

    yy, xx = np.mgrid[0:h, 0:w]
    x = xx + 0.5
    y = yy + 0.5
    inside = np.zeros((h, w), dtype=bool)

    xj, yj = pts[-1]
    for xi, yi in pts:
        crosses = (yi > y) != (yj > y)
        denom = yj - yi
        denom = denom if abs(denom) > 1e-12 else 1e-12
        x_intersect = (xj - xi) * (y - yi) / denom + xi
        inside ^= crosses & (x < x_intersect)
        xj, yj = xi, yi

    mask[inside] = int(label)
    return mask


def freehand_to_mask(
    points: Sequence[Point],
    shape: Tuple[int, int],
    radius: float = 2.5,
    label: int = 1,
    fill_closed: bool = True,
) -> np.ndarray:
    """Rasterize a freehand stroke into a labeled mask."""

    h, w = int(shape[0]), int(shape[1])
    mask = np.zeros((h, w), dtype=np.int32)
    if h <= 0 or w <= 0:
        return mask

    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] == 0:
        return mask

    radius = max(0.5, float(radius))
    radius_sq = radius * radius
    segments = [(pts[0], pts[0])] if pts.shape[0] == 1 else list(zip(pts[:-1], pts[1:]))
    for start, end in segments:
        x0, y0 = float(start[0]), float(start[1])
        x1, y1 = float(end[0]), float(end[1])
        xa = max(0, int(np.floor(min(x0, x1) - radius - 1.0)))
        xb = min(w, int(np.ceil(max(x0, x1) + radius + 1.0)))
        ya = max(0, int(np.floor(min(y0, y1) - radius - 1.0)))
        yb = min(h, int(np.ceil(max(y0, y1) + radius + 1.0)))
        if xb <= xa or yb <= ya:
            continue

        yy, xx = np.mgrid[ya:yb, xa:xb]
        px = xx + 0.5
        py = yy + 0.5
        dx = x1 - x0
        dy = y1 - y0
        length_sq = dx * dx + dy * dy
        if length_sq <= 1e-12:
            dist_sq = (px - x0) ** 2 + (py - y0) ** 2
        else:
            t = ((px - x0) * dx + (py - y0) * dy) / length_sq
            t = np.clip(t, 0.0, 1.0)
            proj_x = x0 + t * dx
            proj_y = y0 + t * dy
            dist_sq = (px - proj_x) ** 2 + (py - proj_y) ** 2
        patch = mask[ya:yb, xa:xb]
        patch[dist_sq <= radius_sq] = int(label)

    if fill_closed and pts.shape[0] >= 3:
        close_distance = float(np.hypot(*(pts[-1] - pts[0])))
        if close_distance <= max(3.0, radius * 2.5):
            filled = polygon_to_mask(pts, (h, w), label=label)
            mask[filled > 0] = int(label)
    return mask


def available_roi_ids(mask: np.ndarray) -> np.ndarray:
    """Return non-background ROI ids present in a label mask or instance stack."""
    arr = np.asarray(mask)
    if arr.ndim == 3:
        planes = arr.reshape(arr.shape[0], -1)
        ids = np.nonzero(np.any(planes != 0, axis=1))[0] + 1
        return ids.astype(np.int64, copy=False)

    ids = np.unique(arr)
    return ids[ids > 0].astype(np.int64)


def bounding_box_from_mask(
    mask: np.ndarray,
    roi_id: int = 1,
    padding: int = 0,
) -> Tuple[int, int, int, int]:
    """Return `(x0, y0, x1, y1)` bounds for a labeled ROI."""

    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"ROI mask must be 2D, got {arr.shape}")

    ys, xs = np.where(arr == int(roi_id))
    if xs.size == 0:
        raise ValueError(f"ROI id {roi_id} not found in mask")

    pad = max(0, int(padding))
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(arr.shape[0], int(ys.max()) + 1 + pad)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(arr.shape[1], int(xs.max()) + 1 + pad)
    return x0, y0, x1, y1


def _movie_read_batch(movie, start: int, stop: int, channel: Optional[int], device: torch.device):
    shape = tuple(movie.shape)
    if len(shape) == 3:
        index = (slice(start, stop), slice(None), slice(None))
    elif len(shape) == 4:
        if channel is None:
            channel = 0
        index = (slice(start, stop), slice(None), slice(None), int(channel))
    else:
        raise ValueError(f"Unsupported movie shape: {shape}")

    return movie.read(index, as_tensor=True, device=device).to(torch.float32)


@torch.inference_mode()
def extract_mean_traces(
    movie,
    roi_mask: np.ndarray,
    roi_ids: Optional[Iterable[int]] = None,
    *,
    channel: Optional[int] = None,
    batch_size: int = 256,
    device: Union[str, torch.device] = "cpu",
) -> dict[int, torch.Tensor]:
    """Extract mean intensity traces for multiple labeled ROIs in one pass."""

    work_device = torch.device(device)
    if torch.is_tensor(roi_mask):
        mask_arr = roi_mask.detach().cpu().numpy()
    else:
        mask_arr = np.asarray(roi_mask)

    movie_shape = tuple(movie.shape)
    if len(movie_shape) not in {3, 4}:
        raise ValueError(f"Unsupported movie shape: {movie_shape}")
    ids = [int(roi_id) for roi_id in (available_roi_ids(mask_arr) if roi_ids is None else roi_ids)]
    ids = list(dict.fromkeys(ids))
    if mask_arr.ndim == 3:
        if tuple(mask_arr.shape[1:]) != tuple(movie_shape[1:3]):
            raise ValueError(f"ROI instance stack shape {mask_arr.shape} does not match movie frame shape {movie_shape[1:3]}")
        labels = np.zeros(tuple(int(v) for v in mask_arr.shape[1:]), dtype=np.int32)
        for roi_id in ids:
            plane_index = int(roi_id) - 1
            if plane_index < 0 or plane_index >= int(mask_arr.shape[0]):
                continue
            plane = mask_arr[plane_index]
            if np.issubdtype(plane.dtype, np.floating):
                roi_pixels = np.isfinite(plane) & (plane > _float_mask_threshold(plane))
            else:
                roi_pixels = plane.astype(bool, copy=False)
            labels[roi_pixels] = int(roi_id)
        mask_arr = labels
    if mask_arr.ndim != 2:
        raise ValueError("roi_mask must be 2D or a 3D instance stack")
    if tuple(mask_arr.shape) != tuple(movie_shape[1:3]):
        raise ValueError(f"ROI mask shape {mask_arr.shape} does not match movie frame shape {movie_shape[1:3]}")
    if not ids:
        return {}

    flat_mask = mask_arr.reshape(-1)
    ids_arr = np.asarray(ids, dtype=np.int64)
    selected = np.isin(flat_mask, ids_arr)
    pixel_indices_np = np.nonzero(selected)[0].astype(np.int64, copy=False)
    if pixel_indices_np.size == 0:
        raise ValueError(f"None of the requested ROI ids were found: {ids}")

    labels_np = flat_mask[pixel_indices_np].astype(np.int64, copy=False)
    sort_order = np.argsort(ids_arr)
    sorted_ids = ids_arr[sort_order]
    positions = np.searchsorted(sorted_ids, labels_np)
    col_indices_np = sort_order[positions].astype(np.int64, copy=False)

    counts = np.bincount(col_indices_np, minlength=len(ids)).astype(np.float32, copy=False)
    missing = [roi_id for roi_id, count in zip(ids, counts) if count <= 0]
    if missing:
        raise ValueError(f"ROI id {missing[0]} is empty")

    pixel_indices = torch.as_tensor(pixel_indices_np, device=work_device, dtype=torch.long)
    col_indices = torch.as_tensor(col_indices_np, device=work_device, dtype=torch.long)
    weights = torch.as_tensor(1.0 / counts[col_indices_np], device=work_device, dtype=torch.float32)

    batch_size = max(1, int(batch_size))
    frame_count = int(movie_shape[0])
    traces = torch.empty((frame_count, len(ids)), dtype=torch.float32)

    for start in range(0, frame_count, batch_size):
        stop = min(start + batch_size, frame_count)
        batch = _movie_read_batch(movie, start, stop, channel, work_device)
        batch_flat = batch.reshape(batch.shape[0], -1)
        selected_values = batch_flat.index_select(1, pixel_indices) * weights.unsqueeze(0)
        chunk = torch.zeros((batch.shape[0], len(ids)), device=work_device, dtype=torch.float32)
        chunk.scatter_add_(1, col_indices.unsqueeze(0).expand(batch.shape[0], -1), selected_values)
        traces[start:stop] = chunk.detach().cpu()
        del batch, batch_flat, selected_values, chunk

    return {roi_id: traces[:, index] for index, roi_id in enumerate(ids)}


@torch.inference_mode()
def extract_mean_trace(
    movie,
    roi_mask: np.ndarray,
    roi_id: int = 1,
    *,
    channel: Optional[int] = None,
    batch_size: int = 256,
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """Extract a simple mean intensity trace from one labeled ROI."""

    return extract_mean_traces(
        movie,
        roi_mask,
        roi_ids=[int(roi_id)],
        channel=channel,
        batch_size=batch_size,
        device=device,
    )[int(roi_id)]


def crop_movie_hwt(
    movie,
    roi_mask: np.ndarray,
    roi_id: int = 1,
    *,
    channel: Optional[int] = None,
    padding: int = 0,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    """Read the ROI bounding crop and return it as `[H, W, T]` for ALI."""

    x0, y0, x1, y1 = bounding_box_from_mask(roi_mask, roi_id=roi_id, padding=padding)
    shape = tuple(movie.shape)
    if len(shape) == 3:
        index = (slice(None), slice(y0, y1), slice(x0, x1))
    elif len(shape) == 4:
        if channel is None:
            channel = 0
        index = (slice(None), slice(y0, y1), slice(x0, x1), int(channel))
    else:
        raise ValueError(f"Unsupported movie shape: {shape}")

    data = movie.read(index, as_tensor=True, device=device).to(torch.float32)
    return data.permute(1, 2, 0).contiguous(), (x0, y0, x1, y1)


@torch.inference_mode()
def run_spikepursuit(
    movie,
    roi_mask: np.ndarray,
    roi_id: int,
    *,
    frame_rate: float,
    channel: Optional[int] = None,
    device: Union[str, torch.device] = "cpu",
    flip_signal: bool = True,
    **spikepursuit_options,
):
    """Run SpikePursuit extraction for one ROI from the GUI pipeline."""
    spikepursuit_options, _, _ = _split_spikepursuit_batch_options(spikepursuit_options)
    mask = torch.as_tensor(np.asarray(roi_mask), dtype=torch.int32)
    extractor = Spikepursuit(
        movie,
        roi_mask=mask,
        channel=channel,
        fr=float(frame_rate),
        device=device,
        flip_signal=bool(flip_signal),
        **spikepursuit_options,
    )
    return extractor.fit_roi(int(roi_id))


def _split_spikepursuit_batch_options(options: dict) -> tuple[dict, Optional[int], Optional[int]]:
    options = dict(options)
    batch_patch_mb = options.pop("roi_batch_patch_mb", None)
    max_rois_per_batch = options.pop("roi_batch_max_rois", None)

    batch_patch_bytes = None
    if batch_patch_mb is not None and float(batch_patch_mb) > 0:
        batch_patch_bytes = int(float(batch_patch_mb) * 1024 * 1024)

    if max_rois_per_batch is not None:
        max_rois_per_batch = int(max_rois_per_batch)
        if max_rois_per_batch <= 0:
            max_rois_per_batch = None

    return options, batch_patch_bytes, max_rois_per_batch


@torch.inference_mode()
def iter_spikepursuit_results(
    movie,
    roi_mask: np.ndarray,
    roi_ids: Iterable[int],
    *,
    frame_rate: float,
    channel: Optional[int] = None,
    device: Union[str, torch.device] = "cpu",
    flip_signal: bool = True,
    **spikepursuit_options,
) -> Iterator:
    """Yield SpikePursuit results for multiple ROIs from the GUI pipeline."""
    options, batch_patch_bytes, max_rois_per_batch = _split_spikepursuit_batch_options(spikepursuit_options)
    mask = torch.as_tensor(np.asarray(roi_mask), dtype=torch.int32)
    extractor = Spikepursuit(
        movie,
        roi_mask=mask,
        channel=channel,
        fr=float(frame_rate),
        device=device,
        flip_signal=bool(flip_signal),
        **options,
    )
    yield from extractor.iter_fit(
        roi_ids,
        batch_patch_bytes=batch_patch_bytes,
        max_rois_per_batch=max_rois_per_batch,
    )


@torch.inference_mode()
def run_ali(
    movie,
    roi_mask: np.ndarray,
    roi_id: int,
    *,
    frame_rate: float,
    channel: Optional[int] = None,
    device: Union[str, torch.device] = "cpu",
    padding: int = 0,
    **ali_options,
):
    """Run ALI extraction for one ROI and return the result with its crop box."""
    data, bbox = crop_movie_hwt(
        movie,
        roi_mask,
        roi_id=roi_id,
        channel=channel,
        padding=padding,
        device=device,
    )
    ali_kwargs = {"coarse_threshold_std": 3.0}
    ali_kwargs.update(ali_options)
    extractor = ALI(fs=int(round(float(frame_rate))), device=str(device), **ali_kwargs)
    return extractor(data), bbox


_MASK_DATASET_NAMES = ("masks", "mask", "roi_mask", "labels", "label", "segmentation")
_MASK_GROUP_NAMES = ("ROI", "roi", "rois")


def _first_h5_dataset(group: h5py.Group) -> h5py.Dataset:
    for key in _MASK_DATASET_NAMES:
        if key in group and isinstance(group[key], h5py.Dataset):
            return group[key]

    for key in _MASK_GROUP_NAMES:
        if key not in group:
            continue
        value = group[key]
        if isinstance(value, h5py.Dataset):
            return value
        if isinstance(value, h5py.Group):
            try:
                return _first_h5_dataset(value)
            except ValueError:
                pass

    for _, value in group.items():
        if isinstance(value, h5py.Dataset):
            return value
        if isinstance(value, h5py.Group):
            try:
                return _first_h5_dataset(value)
            except ValueError:
                pass
    raise ValueError("No dataset found in HDF5 mask file")


def _float_mask_threshold(arr: np.ndarray) -> float:
    finite = arr[np.isfinite(arr)]
    if finite.size and float(finite.min()) >= 0.0 and float(finite.max()) <= 1.0:
        return 0.5
    return 0.0


def _instance_stack_to_label_mask(stack: np.ndarray) -> np.ndarray:
    stack = np.asarray(stack)
    if stack.ndim != 3:
        raise ValueError(f"Mask-RCNN instance stack must be 3D, got {stack.shape}")

    labels = np.zeros(tuple(int(v) for v in stack.shape[1:]), dtype=np.int32)
    next_label = 1
    for plane in stack:
        if np.issubdtype(plane.dtype, np.floating):
            mask = np.isfinite(plane) & (plane > _float_mask_threshold(plane))
        else:
            mask = plane.astype(bool, copy=False)
        if not np.any(mask):
            continue
        labels[mask] = next_label
        next_label += 1
    return labels


def _instance_stack_to_binary_stack(stack: np.ndarray) -> np.ndarray:
    stack = np.asarray(stack)
    if stack.ndim != 3:
        raise ValueError(f"Mask-RCNN instance stack must be 3D, got {stack.shape}")

    out = np.zeros(tuple(int(v) for v in stack.shape), dtype=np.int32)
    for index, plane in enumerate(stack):
        if np.issubdtype(plane.dtype, np.floating):
            mask = np.isfinite(plane) & (plane > _float_mask_threshold(plane))
        else:
            mask = plane.astype(bool, copy=False)
        out[index] = mask.astype(np.int32, copy=False)
    return out


def _looks_like_instance_values(arr: np.ndarray) -> bool:
    if arr.dtype == np.dtype(bool):
        return True
    if arr.size == 0:
        return True
    if np.issubdtype(arr.dtype, np.floating):
        finite = arr[np.isfinite(arr)]
        return finite.size == 0 or (float(finite.min()) >= 0.0 and float(finite.max()) <= 1.0)
    if np.issubdtype(arr.dtype, np.integer):
        if int(arr.min()) >= 0 and int(arr.max()) <= 1:
            return True
        values = np.unique(arr)
        return values.size <= 2 and all(int(value) in {0, 255} for value in values)
    return False


def _looks_like_color_mask(arr: np.ndarray) -> bool:
    return arr.ndim == 3 and arr.shape[-1] in {3, 4} and not _looks_like_instance_values(arr)


def _mask_stack_instance_axis(arr: np.ndarray) -> int:
    if arr.ndim != 3:
        raise ValueError(f"Mask-RCNN instance stack must be 3D, got {arr.shape}")
    sizes = np.asarray(arr.shape, dtype=np.int64)
    min_size = int(sizes.min())
    candidates = [axis for axis, size in enumerate(sizes) if int(size) == min_size]
    if 0 in candidates:
        return 0
    if 2 in candidates:
        return 2
    return int(candidates[0])


def _coerce_mask_array(arr, *, preserve_instances: bool = False) -> np.ndarray:
    if isinstance(arr, np.lib.npyio.NpzFile):
        key = "masks" if "masks" in arr.files else arr.files[0]
        arr = arr[key]

    arr = np.asarray(arr)
    if arr.dtype == object and arr.shape == ():
        obj = arr.item()
        if isinstance(obj, dict):
            for key in ("masks", "mask", "labels", "roi_mask"):
                if key in obj:
                    arr = np.asarray(obj[key])
                    break
            else:
                raise ValueError("Object mask file did not contain a masks-like key")

    arr = np.asarray(arr)
    min_ndim = 3 if preserve_instances else 2
    while arr.ndim > min_ndim and arr.shape[0] == 1:
        arr = arr[0]
    while arr.ndim > min_ndim and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0, ...]
    if arr.ndim == 3:
        if _looks_like_color_mask(arr):
            arr = arr[..., 0]
        else:
            instance_axis = _mask_stack_instance_axis(arr)
            stack = np.moveaxis(arr, instance_axis, 0)
            if preserve_instances:
                return _instance_stack_to_binary_stack(stack)
            arr = _instance_stack_to_label_mask(stack)
    if arr.ndim != 2:
        expected = "2D or an instance stack" if preserve_instances else "2D"
        raise ValueError(f"Mask must be {expected} after loading, got {arr.shape}")
    return arr.astype(np.int32, copy=False)


def load_mask_file(
    path: PathLike,
    dataset: Optional[str] = None,
    *,
    preserve_instances: bool = False,
) -> np.ndarray:
    """Load a labeled ROI mask from TIFF, NPY/NPZ, or HDF5."""

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        return _coerce_mask_array(tifffile.imread(path), preserve_instances=preserve_instances)
    if suffix in {".npy", ".npz"}:
        return _coerce_mask_array(np.load(path, allow_pickle=True), preserve_instances=preserve_instances)
    if suffix in {".h5", ".hdf5"}:
        with h5py.File(path, "r") as f:
            dset = f[dataset] if dataset else _first_h5_dataset(f)
            return _coerce_mask_array(dset[()], preserve_instances=preserve_instances)
    raise ValueError(f"Unsupported mask file type: {suffix}")


def ensure_shape_matches(mask: np.ndarray, frame_shape: Iterable[int]) -> None:
    """Raise if an ROI mask shape does not match a movie frame shape."""
    frame_shape = tuple(int(v) for v in frame_shape)
    if len(frame_shape) == 3:
        frame_shape = frame_shape[:2]
    mask_shape = tuple(np.asarray(mask).shape)
    spatial_shape = mask_shape[-2:] if len(mask_shape) == 3 else mask_shape
    if tuple(spatial_shape) != tuple(frame_shape[:2]):
        raise ValueError(f"Mask shape {mask_shape} does not match frame shape {frame_shape[:2]}")
