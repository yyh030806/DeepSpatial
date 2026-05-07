<p align="center">
  <img src="docs/source/_static/text_logo.png" width="200" alt="DeepSpatial">
</p>

<p align="center">
  <a href="https://pypi.org/project/deepspatial/"><img src="https://img.shields.io/badge/Pypi-0.1.0-317EC2.svg" alt="PyPI"></a>
  <a href="https://yyh030806.github.io/DeepSpatial/"><img src="https://img.shields.io/badge/Homepage-deepspatial-f773a8.svg" alt="Homepage"></a>
  <a href="https://yyh030806.github.io/DeepSpatial/docs/"><img src="https://img.shields.io/badge/Documentation-latest-4CAF50.svg" alt="Docs"></a>
  <a href="https://doi.org/10.64898/2026.04.28.721395"><img src="https://img.shields.io/badge/Paper-bioRxiv-00AAB5.svg" alt="BioRxiv"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-FF9800.svg" alt="License"></a>
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

## Citation

If you use DeepSpatial in your research, please cite:

```bibtex
@article {yang2026deepspatial,
	author = {Yang, Yuhang and Luo, Yiming and Zhang, Kai and Bu, Yonggan and Xia, Zheng and Peng, Haoxin and Yan, Rui and Liu, Qi and Chen, Yang and Shen, Lin and Chen, Enhong},
	title = {Reconstructing True 3D Spatial Omics at Single-Cell Resolution},
	year = {2026},
	doi = {10.64898/2026.04.28.721395},
	journal = {bioRxiv}
}
```

## License

DeepSpatial is released under the MIT License.
