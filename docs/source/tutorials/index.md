# Tutorials

This section provides end-to-end DeepSpatial tutorials across multiple spatial omics datasets.
Each tutorial follows a consistent workflow:

1. Import dependencies.
2. Prepare slice-level data.
3. Set up and train DeepSpatial.
4. Reconstruct a full 3D volume.
5. Visualize and inspect results.

## Tutorial Catalog

- **Mouse Hypothalamus MERFISH**: Reconstruction from serial MERFISH slices with physical inter-slice spacing.
- **Human Breast Cancer IMC**: Reconstruction from IMC slices using 3D coordinates split into XY (`spatial`) and Z (`z_coord`).
- **Mouse Brain DeepStarMap**: 3D validation workflow for evaluating reconstructed mouse brain structures.

```{toctree}
:maxdepth: 1
:titlesonly: true

merfish_mouse_hypothalamus
imc_human_breastcancer
deepstarmap_mouse_brain
```