<p align="center">
  <img src="docs/source/_static/text_logo.png" width="200" alt="DeepSpatial">
</p>

<p align="center">
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/Pypi-0.1.0-317EC2.svg" alt="PyPI(TODO)"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/Paper-bioRxiv-00AAB5.svg" alt="BioRxiv(TODO)"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-f773a8.svg" alt="License"></a>
</p>


**DeepSpatial** is a package for true 3D reconstruction of spatial omics tissues from serial 2D slices, built with `PyTorch` and designed to work smoothly with `AnnData`/`Scanpy` workflows.

## 3D spatial omics reconstruction

DeepSpatial provides an end-to-end framework for learning continuous 3D tissue representations:

- Reconstructs missing biological structure between adjacent sections.
- Jointly models spatial coordinates, gene expression, and cell identities.
- Supports large-scale training and sampling with GPU acceleration.
- Produces outputs that can be directly used in downstream single-cell/spatial analysis.

The package exposes a high-level API (`DeepSpatial`) for data setup, model training, and 3D reconstruction with minimal boilerplate.

## Installation

Recommended (PyPI):

```bash
pip install deepspatial
```

From source (development):

```bash
git clone https://github.com/yyh030806/DeepSpatial.git
cd DeepSpatial
pip install -e .
```

If you use GPU, install a PyTorch build matching your CUDA version.

## Quick start

1. Download an example dataset first: [Google Drive dataset folder](https://drive.google.com/drive/folders/11MICO4KmGRDlKfFf6LQ_aZC3mtF9NrNG?usp=sharing)

   Then place files under `data/merfish_mouse_hypothalamus/`.

2. Run DeepSpatial:

```python
import glob
import scanpy as sc
import deepspatial as ds

adatas = [
    sc.read_h5ad(p)
    for p in sorted(glob.glob("data/merfish_mouse_hypothalamus/merfish_*.h5ad"))
]

model = ds.DeepSpatial()
model.setup_data(adatas)
model.build_model()
model.fit(max_epochs=100)

adata_3d = model.reconstruct_full_volume(adatas, thickness=10.0)
```

## Resources

- Documentation, tutorials, and API reference: https://yyh030806.github.io/DeepSpatial/docs/
- Homepage: https://yyh030806.github.io/DeepSpatial/
- Bug reports and feature requests: https://github.com/yyh030806/DeepSpatial/issues

## Citation(TODO)

If you use DeepSpatial in your research, please cite:

```bibtex
@article{yang2026deepspatial,
  author = {Yuhang Yang},
  title = {DeepSpatial: Reconstructing True 3D Spatial Transcriptomics at Single-Cell Resolution},
  year = {2026},
  journal = {In preparation},
  url = {https://github.com/yyh030806/DeepSpatial}
}
```

## License

DeepSpatial is released under the MIT License.
