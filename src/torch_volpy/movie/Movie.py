from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset

PathLike = Union[str, Path]


class Movie(Dataset):
    """
    HDF5-backed movie reader/writer for PyTorch.

    Assumptions:
    - The HDF5 dataset shape is usually (T, Y, X) or (T, Y, X, C)
    - Axis 0 is the frame/time axis
    - For TIFF conversion, the TIFF is a standard multi-page TIFF where each page is one frame

    Key properties:
    - Lazy HDF5 opening (safe pattern for PyTorch DataLoader workers)
    - Partial reads via slicing, so only requested frames enter memory
    - Save/append torch tensors to HDF5
    - Convert TIFF -> HDF5 page by page
    """

    def __init__(
        self,
        h5_path: PathLike,
        dataset: str = "movie",
        mode: str = "r",
        transform=None,
        rdcc_nbytes: Optional[int] = None,
    ) -> None:
        self.h5_path = str(h5_path)
        self.dataset = dataset
        self.mode = mode
        self.transform = transform
        self.rdcc_nbytes = rdcc_nbytes

        self._file: Optional[h5py.File] = None
        self._dset: Optional[h5py.Dataset] = None
        self._shape: Tuple[int, ...]
        self._dtype: np.dtype

        with self._open_file_temporarily() as f:
            if self.dataset not in f:
                raise KeyError(f"Dataset '{self.dataset}' not found in {self.h5_path}")
            dset = f[self.dataset]
            self._shape = tuple(dset.shape)
            self._dtype = np.dtype(dset.dtype)

        if len(self._shape) < 1:
            raise ValueError("Movie dataset must have at least 1 dimension")

    # ---------- internal helpers ----------

    def _h5_open_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self.rdcc_nbytes is not None:
            kwargs["rdcc_nbytes"] = int(self.rdcc_nbytes)
        return kwargs

    def _open_file_temporarily(self) -> h5py.File:
        return h5py.File(self.h5_path, self.mode, **self._h5_open_kwargs())

    def _ensure_open(self) -> None:
        if self._file is None:
            self._file = h5py.File(self.h5_path, self.mode, **self._h5_open_kwargs())
            self._dset = self._file[self.dataset]

    @staticmethod
    def _to_numpy(data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().contiguous().numpy()
        return np.asarray(data)

    @staticmethod
    def _normalize_chunks(shape: Sequence[int], chunk_frames: int = 1) -> Tuple[int, ...]:
        """
        Default chunking: chunk by frame blocks along axis 0, keep full frame shape.
        Good when most reads are frame-wise or short contiguous frame ranges.
        """
        normalized_shape = tuple(int(s) for s in shape)
        if len(normalized_shape) == 0:
            raise ValueError("Shape must have at least one dimension")

        first = max(1, int(chunk_frames))
        if len(normalized_shape) == 1:
            return (first,)
        return (first, *normalized_shape[1:])

    @staticmethod
    def _prepare_movie_array(data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        arr = Movie._to_numpy(data)

        if arr.ndim < 1:
            raise ValueError("Movie tensor must have at least 1 dimension")

        # Treat 2D arrays as single-frame grayscale movies
        if arr.ndim == 2:
            arr = arr[None, ...]

        return arr

    def refresh(self) -> None:
        """Refresh shape/dtype metadata from disk."""
        if self._file is not None and self._dset is not None:
            self._shape = tuple(self._dset.shape)
            self._dtype = np.dtype(self._dset.dtype)
            return

        with self._open_file_temporarily() as f:
            dset = f[self.dataset]
            self._shape = tuple(dset.shape)
            self._dtype = np.dtype(dset.dtype)

    # ---------- public metadata ----------

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def num_frames(self) -> int:
        return int(self._shape[0])

    @property
    def frame_shape(self) -> Tuple[int, ...]:
        return tuple(self._shape[1:])

    # ---------- file lifecycle ----------

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
        self._file = None
        self._dset = None

    def __enter__(self) -> "Movie":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # ---------- PyTorch Dataset API ----------

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, idx):
        """
        For DataLoader usage:
        - int -> one frame tensor
        - slice / tuple -> tensor block
        """
        return self.read(idx, as_tensor=True)

    # ---------- reading ----------

    def read(
        self,
        index=...,
        as_tensor: bool = True,
        dtype: Optional[Union[np.dtype, str]] = None,
        device: Optional[Union[str, torch.device]] = None,
        copy: bool = False,
    ):
        """
        Read any valid h5py slice/index.

        Examples:
            movie.read(0)                  -> one frame
            movie.read(slice(10, 20))      -> frames 10:20
            movie.read((slice(10, 20), ...))
            movie.read((0, slice(None), slice(None)))
        """
        self._ensure_open()
        assert self._dset is not None

        arr = self._dset[index]

        if dtype is not None:
            arr = arr.astype(dtype, copy=False)

        if not as_tensor:
            return np.array(arr, copy=True) if copy else arr

        # Make sure PyTorch gets a CPU ndarray
        if copy:
            arr = np.array(arr, copy=True, order="C")
        else:
            arr = np.asarray(arr, order="C")
        tensor = torch.from_numpy(arr)

        if self.transform is not None:
            tensor = self.transform(tensor)

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
        """Read a frame range along axis 0 only."""
        return self.read(
            slice(start, stop, step),
            as_tensor=as_tensor,
            dtype=dtype,
            device=device,
        )

    def iter_chunks(
        self,
        frames_per_chunk: int,
        as_tensor: bool = True,
        dtype: Optional[Union[np.dtype, str]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Iterator[Union[np.ndarray, torch.Tensor]]:
        """Iterate through the movie in frame chunks."""
        if frames_per_chunk <= 0:
            raise ValueError("frames_per_chunk must be > 0")

        for start in range(0, self.num_frames, frames_per_chunk):
            stop = min(start + frames_per_chunk, self.num_frames)
            yield self.read_frames(
                start,
                stop,
                as_tensor=as_tensor,
                dtype=dtype,
                device=device,
            )

    # ---------- writing ----------

    @classmethod
    def create_empty(
        cls,
        h5_path: PathLike,
        frame_shape: Sequence[int],
        dtype: Union[str, np.dtype],
        dataset: str = "movie",
        overwrite: bool = False,
        chunk_frames: int = 1,
        compression: Optional[str] = None,
        compression_opts: Optional[int] = None,
        attrs: Optional[dict] = None,
    ) -> "Movie":
        """
        Create an empty appendable movie dataset with shape (0, *frame_shape).
        """
        h5_path = str(h5_path)
        file_mode = "a" if Path(h5_path).exists() else "w"
        frame_shape = tuple(frame_shape)

        with h5py.File(h5_path, file_mode) as f:
            if dataset in f:
                if not overwrite:
                    raise FileExistsError(
                        f"Dataset '{dataset}' already exists in {h5_path}. "
                        "Use overwrite=True to replace it."
                    )
                del f[dataset]

            dset = f.create_dataset(
                dataset,
                shape=(0, *frame_shape),
                maxshape=(None, *frame_shape),
                dtype=np.dtype(dtype),
                chunks=cls._normalize_chunks((0, *frame_shape), chunk_frames),
                compression=compression,
                compression_opts=compression_opts,
            )

            if attrs:
                for key, value in attrs.items():
                    dset.attrs[key] = value

        return cls(h5_path, dataset=dataset, mode="r")

    @staticmethod
    def _save_hdf5(
        h5_path: PathLike,
        data: Union[np.ndarray, torch.Tensor],
        overwrite: bool = False,
        dataset: str = "movie",
        chunk_frames: int = 1,
        compression: Optional[str] = None,
        compression_opts: Optional[int] = None,
        attrs: Optional[dict] = None,
    ) -> bool:
        data = Movie._prepare_movie_array(data)

        h5_path = str(h5_path)
        file_mode = "a" if Path(h5_path).exists() else "w"

        with h5py.File(h5_path, file_mode) as f:
            if dataset in f:
                if not overwrite:
                    raise FileExistsError(
                        f"Dataset '{dataset}' already exists in {h5_path}. "
                        "Use overwrite=True to replace it."
                    )
                del f[dataset]

            dset = f.create_dataset(
                dataset,
                data=data,
                maxshape=(None, *data.shape[1:]),
                chunks=Movie._normalize_chunks(data.shape, chunk_frames),
                compression=compression,
                compression_opts=compression_opts,
            )

            if attrs:
                for key, value in attrs.items():
                    dset.attrs[key] = value

        return True

    @staticmethod
    def _save_tiff(
        tiff_path: PathLike,
        data: Union[np.ndarray, torch.Tensor],
        overwrite: bool = False,
        attrs: Optional[dict] = None,
    ) -> bool:
        data = Movie._prepare_movie_array(data)

        path = Path(tiff_path)
        if path.suffix.lower() not in (".tif", ".tiff"):
            path = path.with_suffix(".tif")

        if path.exists() and not overwrite:
            raise FileExistsError(
                f"File '{path}' already exists. Use overwrite=True to replace it."
            )

        if data.ndim == 3 and data.shape[0] in (3, 4):
            data = np.transpose(data, (1, 2, 0))
        elif data.ndim == 4 and data.shape[1] in (3, 4):
            data = np.transpose(data, (0, 2, 3, 1))

        tifffile.imwrite(
            str(path),
            data,
            compression="lzw",
            metadata=attrs or None,
        )

        return True

    @classmethod
    def save_tensor(
        cls,
        path: PathLike,
        data: Union[np.ndarray, torch.Tensor],
        *,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> bool:
        path = Path(path)

        if path.suffix.lower() in {".tif", ".tiff"}:
            return cls._save_tiff(path, data, overwrite=overwrite, **kwargs)
        elif path.suffix.lower() in {".h5", ".hdf5"}:
            return cls._save_hdf5(path, data, overwrite=overwrite, **kwargs)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")

    @classmethod
    def append_tensor(
        cls,
        h5_path: PathLike,
        data: Union[np.ndarray, torch.Tensor],
        dataset: str = "movie",
    ) -> None:
        """
        Append frames along axis 0.
        Accepts shape:
            (T, ...)
        or a single frame shape:
            (...)
        """
        arr = cls._to_numpy(data)

        with h5py.File(str(h5_path), "a") as f:
            if dataset not in f:
                raise KeyError(f"Dataset '{dataset}' not found in {h5_path}")

            dset = f[dataset]

            if arr.ndim == dset.ndim - 1:
                arr = arr[np.newaxis, ...]

            if arr.ndim != dset.ndim:
                raise ValueError(
                    f"Rank mismatch: data has ndim={arr.ndim}, dataset has ndim={dset.ndim}"
                )

            if tuple(arr.shape[1:]) != tuple(dset.shape[1:]):
                raise ValueError(
                    f"Frame shape mismatch: data {arr.shape[1:]} vs dataset {dset.shape[1:]}"
                )

            old_n = dset.shape[0]
            new_n = old_n + arr.shape[0]
            dset.resize(new_n, axis=0)
            dset[old_n:new_n] = arr

    # ---------- TIFF -> HDF5 ----------

    @classmethod
    def from_tiff(
        cls,
        tiff_path: PathLike,
        h5_path: PathLike,
        dataset: str = "movie",
        overwrite: bool = False,
        chunk_frames: int = 16,
        compression: Optional[str] = None,
        compression_opts: Optional[int] = None,
        attrs: Optional[dict] = None,
    ) -> "Movie":
        """
        Convert a standard multi-page TIFF movie to HDF5 without loading the whole movie.

        Memory behavior:
        - reads one TIFF page at a time
        - writes small frame batches into HDF5

        Notes:
        - This implementation assumes each TIFF page is one movie frame.
        - For more complex OME-TIFF axis layouts, adapt this method using tifffile metadata.
        """
        tiff_path = str(tiff_path)
        h5_path = str(h5_path)

        with tifffile.TiffFile(tiff_path) as tif:
            n_pages = len(tif.pages)
            if n_pages == 0:
                raise ValueError(f"No TIFF pages found in {tiff_path}")

            first = tif.pages[0].asarray()
            frame_shape = first.shape
            frame_dtype = first.dtype

            file_mode = "a" if Path(h5_path).exists() else "w"
            with h5py.File(h5_path, file_mode) as f:
                if dataset in f:
                    if not overwrite:
                        raise FileExistsError(
                            f"Dataset '{dataset}' already exists in {h5_path}. "
                            "Use overwrite=True to replace it."
                        )
                    del f[dataset]

                dset = f.create_dataset(
                    dataset,
                    shape=(n_pages, *frame_shape),
                    maxshape=(None, *frame_shape),
                    dtype=frame_dtype,
                    chunks=cls._normalize_chunks((n_pages, *frame_shape), chunk_frames),
                    compression=compression,
                    compression_opts=compression_opts,
                )

                merged_attrs = dict(attrs or {})
                merged_attrs["source_tiff"] = tiff_path
                for key, value in merged_attrs.items():
                    dset.attrs[key] = value

                # Write the first page
                dset[0] = first

                # Write remaining pages in small batches
                batch: list[np.ndarray] = []
                batch_start = 1

                for page_index in range(1, n_pages):
                    batch.append(tif.pages[page_index].asarray())

                    if len(batch) >= chunk_frames:
                        dset[batch_start : batch_start + len(batch)] = np.stack(batch, axis=0)
                        batch_start += len(batch)
                        batch.clear()

                if batch:
                    dset[batch_start : batch_start + len(batch)] = np.stack(batch, axis=0)

        return cls(h5_path, dataset=dataset, mode="r")

    @classmethod
    def to_tiff(
        cls,
        h5_path: PathLike,
        tiff_path: PathLike,
        dataset: str = "movie",
        dtype: Optional[Union[np.dtype, str]] = None,
        bigtiff: Optional[bool] = None,
        overwrite: bool = False,
        rdcc_nbytes: Optional[int] = None,
        batch_size: int = 64,
    ) -> Path:
        h5_path = str(h5_path)
        tiff_path = Path(tiff_path)

        if tiff_path.exists() and not overwrite:
            raise FileExistsError(
                f"{tiff_path} already exists. Use overwrite=True to replace it."
            )

        with cls(
            h5_path,
            dataset=dataset,
            mode="r",
            rdcc_nbytes=rdcc_nbytes,
        ) as movie:
            out_dtype = np.dtype(dtype) if dtype is not None else movie.dtype

            if bigtiff is None:
                nbytes = (
                    int(movie.num_frames)
                    * int(np.prod(movie.frame_shape))
                    * out_dtype.itemsize
                )
                bigtiff = nbytes >= 4 * 1024**3

            with tifffile.TiffWriter(str(tiff_path), bigtiff=bigtiff, imagej=False) as tif:
                for start in range(0, movie.num_frames, batch_size):
                    stop = min(start + batch_size, movie.num_frames)

                    batch = movie.read(
                        slice(start, stop),
                        as_tensor=False,
                        dtype=dtype,
                        copy=False,
                    )

                    # Write one frame per TIFF page
                    for frame in batch:
                        write_kwargs = {"contiguous": True}

                        # RGB frame: (Y, X, C)
                        if frame.ndim == 3 and frame.shape[-1] in (3, 4):
                            write_kwargs["photometric"] = "rgb"

                        # Do NOT write shaped metadata here
                        tif.write(frame, **write_kwargs)

        return tiff_path
