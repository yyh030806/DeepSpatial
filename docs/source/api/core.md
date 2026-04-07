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

Prior to modeling, the data pipeline must be initialized. This process automatically extracts the physical coordinate boundaries from the 2D slices and computes the normalization statistics necessary for accurate physical dimension scaling.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   DeepSpatial.setup_data
```

### Model & Training

Configures the core Transformer-based architecture and Flow Matching dynamics, while managing the training lifecycle and model checkpoints.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   DeepSpatial.build_model
   DeepSpatial.fit
   DeepSpatial.load_checkpoint
```

### 3D Reconstruction

The core generation and inference API. By leveraging the trained continuous vector field, these methods synthesize single-cell resolution gene expression profiles and spatial coordinates within the unobserved physical space between adjacent slices.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   DeepSpatial.reconstruct_full_volume
   DeepSpatial.reconstruct_between_slices
```