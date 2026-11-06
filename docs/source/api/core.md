# Core

The `deepspatial.core` module serves as the primary entry point for the DeepSpatial framework. It provides a user-friendly, high-level API designed to facilitate the seamless reconstruction of continuous 3D tissue volumes from discrete 2D spatial transcriptomics slices.

```{eval-rst}
.. currentmodule:: deepspatial.core
```

## Main Class

`DeepSpatial` is the primary user-facing class. It orchestrates internal dataset states, model weights, and the mapping statistics required for physical coordinate space transformations.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   DeepSpatial
```

## Workflow Methods

The following methods define the standard workflow for 3D volume reconstruction utilizing the `DeepSpatial` class:

### Data Preparation

Prior to modeling, the data pipeline must be initialized. This process extracts the physical coordinate boundaries, writes normalized internal coordinate keys, constructs adjacent-slice UOT pairs, and records the UOT configuration in `data_config`. The standard pipeline expects cell-type annotations because they are used in UOT, flow matching, and output decoding.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   DeepSpatial.setup_data
```

### Model & Training

Configures the GiT architecture and joint continuous Flow Matching dynamics. Spatial coordinates, gene expression, and relaxed one-hot cell-type states use velocity-matching losses; cell type is not trained with cross-entropy. Saved metadata includes data, model, and training configurations.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   DeepSpatial.build_model
   DeepSpatial.fit
   DeepSpatial.load_checkpoint
```

### 3D Reconstruction

The core generation and inference API. It uses density-aware target depths and depth-dependent parent directions. Forward samples follow the learned ODE from physical time 0, while backward samples use explicit reverse-time integration from physical time 1 without swapping depth endpoints or flipping a forward trajectory. Generated expression is sparsified to the parent cell's non-zero gene count. There is no oversampling or spatial subset pruning.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   DeepSpatial.reconstruct_full_volume
   DeepSpatial.reconstruct_between_slices
```

The internal generation helper is named `_generate_virtual_cells`, reflecting
its actual role; it does not implement a pruning optimization.
