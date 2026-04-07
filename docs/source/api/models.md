# Models

The `deepspatial.models` module contains the core neural network architectures that power the Flow Matching continuous mapping. It features the **GiT** network, which is specifically designed to simultaneously process spatial coordinates, high-dimensional gene expressions, and categorical cell types under physical Z-depth conditioning.

```{eval-rst}
.. currentmodule:: deepspatial.models.git
```

## Main Architecture
The primary transformer-based backbone responsible for learning the multi-modal continuous vector fields.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   GiT
```

## Building Blocks
The internal neural network components used to construct the GiT model, including modality-specific embedders and Adaptive Layer Normalization (adaLN-Zero) transformer blocks.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   GiTBlock
   PatchEmbedder
```