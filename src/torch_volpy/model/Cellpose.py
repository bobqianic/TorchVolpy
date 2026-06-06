import numpy as np
from pathlib import Path
from typing import Union, List, Optional

import torch
from cellpose import models, core, io, plot
from matplotlib import pyplot as plt

from ..util import IJAB


class Cellpose:
    def __init__(
        self,
        model_path: Union[str, Path],
        gpu: bool = True,
        batch_size: int = 32,
        flow_threshold: float = 0.4,
        cellprob_threshold: float = 0.0,
        tile_norm_blocksize: int = 0,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        Initialize a Cellpose segmentation model.

        Args:
            model_path: Path to the pretrained Cellpose model.
            gpu: Whether to use GPU.
            batch_size: Batch size passed to model.eval().
            flow_threshold: Flow threshold for Cellpose.
            cellprob_threshold: Cell probability threshold for Cellpose.
            tile_norm_blocksize: Normalization block size.
        """
        self.model_path = str(model_path)
        self.gpu = gpu
        self.batch_size = batch_size
        self.flow_threshold = flow_threshold
        self.cellprob_threshold = cellprob_threshold
        self.tile_norm_blocksize = tile_norm_blocksize
        self.device = torch.device(device) if device is not None else None

        if self.device is not None:
            if self.device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError(f"device={self.device} requested, but CUDA is not available.")
            if self.device.type == "mps" and not torch.backends.mps.is_available():
                raise RuntimeError(f"device={self.device} requested, but MPS is not available.")
            self.gpu = self.device.type in {"cuda", "mps"}
        elif self.gpu and not core.use_gpu():
            raise RuntimeError("GPU was requested, but no GPU is available.")

        self.model = models.CellposeModel(
            gpu=self.gpu,
            pretrained_model=self.model_path,
            device=self.device,
        )

    def _prepare_image(
        self,
        img: np.ndarray,
        selected_channels: Optional[List[int]] = None,
    ) -> np.ndarray:
        """
        Select channels from the image if requested.
        Assumes channel dimension is last: H x W x C
        """
        if selected_channels is None:
            return img

        if img.ndim < 3:
            raise ValueError("selected_channels was provided, but image has no channel dimension.")

        n_channels = img.shape[-1]
        for c in selected_channels:
            if c < 0 or c >= n_channels:
                raise ValueError(
                    f"Invalid channel index {c}. Valid range is [0, {n_channels - 1}]."
                )

        img_selected = np.zeros_like(img)
        img_selected[:, :, :len(selected_channels)] = img[:, :, selected_channels]
        return img_selected

    def _resolve_save_dir(
        self,
        image_source: Optional[Union[str, Path]],
        save_dir: Optional[Union[str, Path]],
    ) -> Path:
        """
        Resolve the directory to save outputs into.
        """
        if save_dir is None:
            if image_source is None:
                raise ValueError(
                    "save_dir must be provided when saving outputs for numpy array or torch tensor inputs."
                )
            return Path(image_source).parent
        return Path(save_dir)

    def _save_mask(
        self,
        mask: np.ndarray,
        image_source: Optional[Union[str, Path]],
        save_dir: Optional[Union[str, Path]],
        index: int,
    ) -> Path:
        """
        Save mask to disk as a TIFF file.
        """
        save_dir = self._resolve_save_dir(image_source=image_source, save_dir=save_dir)

        save_dir.mkdir(parents=True, exist_ok=True)

        if image_source is not None:
            stem = Path(image_source).stem
            out_path = save_dir / f"{stem}_masks.tif"
        else:
            out_path = save_dir / f"image_{index}_masks.tif"

        io.imsave(str(out_path), mask.astype(np.uint16))
        return out_path

    def _save_segmentation_plot(
            self,
            img: np.ndarray,
            masks: np.ndarray,
            flows,
            image_source: Optional[Union[str, Path]],
            save_dir: Optional[Union[str, Path]],
            index: int,
    ) -> Path:
        """
        Save Cellpose segmentation visualization to disk,
        with mask IDs overlaid on subplot (1, 4, 2).
        """
        save_dir = self._resolve_save_dir(image_source=image_source, save_dir=save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        if image_source is not None:
            stem = Path(image_source).stem
            out_path = save_dir / f"{stem}_segmentation.png"
        else:
            out_path = save_dir / f"image_{index}_segmentation.png"



        fig = plt.figure(figsize=(12, 5))
        plot.show_segmentation(fig, IJAB.imagej_fp32_to_uint8(img), masks, flows[0])

        # Cellpose axes:
        # fig.axes[0] -> (1, 4, 1) original image
        # fig.axes[1] -> (1, 4, 2) predicted outlines
        # fig.axes[2] -> (1, 4, 3) predicted masks
        # fig.axes[3] -> (1, 4, 4) predicted cell pose
        ax = fig.axes[2]  # subplot (1, 4, 3)

        mask_ids = np.unique(masks)
        mask_ids = mask_ids[mask_ids > 0]  # skip background

        for mask_id in mask_ids:
            ys, xs = np.nonzero(masks == mask_id)
            if len(xs) == 0:
                continue

            # label position: median is often more stable than mean
            x = int(np.median(xs))
            y = int(np.median(ys))

            ax.text(
                x,
                y,
                str(mask_id),
                color="white",
                fontsize=7,
                ha="center",
                va="center",
                bbox=dict(facecolor="none", alpha=0.0, edgecolor="none", pad=0.2)
            )

        plt.tight_layout()
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        return out_path

    @torch.inference_mode()
    def build(
            self,
            images: Union[
                str,
                Path,
                np.ndarray,
                torch.Tensor,
                List[Union[str, Path, np.ndarray, torch.Tensor]],
            ],
            selected_channels: Optional[List[int]] = None,
            save_to_disk: bool = False,
            save_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Run segmentation.

        Args:
            images:
                A single image path, a single numpy array, a single torch tensor,
                or a list of image paths / numpy arrays / torch tensors.
            selected_channels:
                List of channel indices to keep, e.g. [0, 1, 2].
                If None, uses the original image as-is.
            save_to_disk:
                Whether to save masks to disk.
            save_dir:
                Directory to save masks into. Required for numpy/tensor inputs
                if save_to_disk=True.

        Returns:
            For a single input: mask (torch.Tensor)
            For multiple inputs: list of masks
        """
        single_input = not isinstance(images, list)
        if single_input:
            images = [images]

        masks_out = []

        for idx, item in enumerate(images):
            image_source = None

            if isinstance(item, (str, Path)):
                image_source = item
                img = io.imread(str(item))
            elif isinstance(item, np.ndarray):
                img = item
            elif isinstance(item, torch.Tensor):
                img = item.detach().cpu().numpy()
            else:
                raise TypeError(
                    "Each image must be a file path (str/Path), numpy.ndarray, or torch.Tensor."
                )

            img_prepared = self._prepare_image(img, selected_channels)

            import time
            s = time.time()
            masks, flows, styles = self.model.eval(
                img_prepared,
                batch_size=self.batch_size,
                flow_threshold=self.flow_threshold,
                cellprob_threshold=self.cellprob_threshold,
                normalize={"tile_norm_blocksize": self.tile_norm_blocksize},
            )
            e = time.time()
            print("Segmentation took {:.2f} seconds".format(e - s))

            masks_out.append(torch.from_numpy(masks).to(dtype=torch.int32))

            if save_to_disk:
                self._save_mask(
                    mask=masks,
                    image_source=image_source,
                    save_dir=save_dir,
                    index=idx,
                )
                self._save_segmentation_plot(
                    img=img_prepared,
                    masks=masks,
                    flows=flows,
                    image_source=image_source,
                    save_dir=save_dir,
                    index=idx,
                )

        return masks_out[0] if single_input else masks_out
