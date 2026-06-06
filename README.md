# torch-volpy

PyTorch tools for voltage imaging movie processing and signal extraction.

The package currently provides:

- HDF5-backed movie I/O through `Movie`
- Motion template building, translation estimation, and motion correction
- Gaussian high-pass filtering
- Summary image generation
- Cellpose-based segmentation support
- ALI and SpikePursuit signal extraction

## Installation

```bash
pip install torch-volpy
```

Install the GUI extra for interactive movie viewing and ROI trace extraction:

```bash
pip install "torch-volpy[gui]"
torch-volpy-gui
```

For local development from this repository:

```bash
python -m pip install -e ".[gui,dev]"
```

## Basic Imports

```python
from torch_volpy.movie import Movie
from torch_volpy.motion import MotionCorrect, Template, Translation
from torch_volpy.filter import Filter
from torch_volpy.model import Summary
from torch_volpy.extraction import ALI, Spikepursuit
from torch_volpy.model import Cellpose
```

## GUI

The PyQt GUI opens HDF5 movies (`.h5`/`.hdf5`, dataset defaults to `movie`) and
TIFF stacks (`.tif`/`.tiff`). It provides frame playback, Cellpose ROI
generation from a summary image, loading existing ROI masks (`.tif`, `.npy`,
`.npz`, `.h5`/`.hdf5`), click-to-select ROI picking, and trace extraction with:

- `Spikepursuit` as the default extraction method
- `ALI` for cropped ROI activity localization
- a simple mean-ROI trace for quick inspection

The Cellpose ROI button follows the test workflow in `_test/test_cellpose.py`:
build `Summary(movie)`, stack `[mean, mean, corr]`, and pass that image to
`Cellpose.build(...)`. The resulting labeled mask is shown as an overlay; click
an ROI in `Select` mode to choose which label is used for trace extraction.

When opening a TIFF stack, the GUI first converts it to a sibling HDF5 file,
runs motion correction into `corrected_<name>.h5`, and then displays the
corrected HDF5 movie. A progress bar in the Movie panel reports conversion and
motion-correction phases.

<p align="center">
  <img src="./media/2026-06-06%20231838.png" width="800" alt="App Screenshot">
</p>
<p align="center">
  <img src="./media/2026-06-06%20232535.png" width="800" alt="App Screenshot">
</p>

## Build And Publish

```bash
python -m build
python -m twine upload dist/*
```

Before publishing publicly, add a `LICENSE` file and matching license metadata to `pyproject.toml`.
