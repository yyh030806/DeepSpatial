# Models

The `deepspatial.models` module contains the GiT network used to predict joint spatial, gene-expression, and cell-type velocity fields. Cell type enters GiT as a continuously relaxed one-hot vector, not as a discrete embedding lookup.

```{eval-rst}
.. currentmodule:: deepspatial.models.git
```

## Main Architecture
The primary transformer-based backbone responsible for learning the multimodal continuous vector fields. Coordinates are projected with `Linear(2, H)`, gene-expression patches use a two-layer MLP embedder, and the $K$-dimensional cell-type state uses `Linear(K, H)` as global conditioning. GiT does not use a two-layer spatial MLP or `nn.Embedding` for cell type.

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
