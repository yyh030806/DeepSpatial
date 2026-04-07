# Data Utilities

The Data Utilities module provides the foundational data structures and mathematical solvers required to construct continuous training trajectories. It handles the complex task of linking discrete cells across adjacent 2D slices using Unbalanced Optimal Transport (UOT).

---

## PyTorch Dataset

The core dataset class. It automatically computes cross-slice pairings and formats spatial coordinates, sparse gene expression matrices, and cell-type annotations into optimized PyTorch tensors for Flow Matching.

```{eval-rst}
.. currentmodule:: deepspatial.data_utils
```
```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   DeepSpatialDataset
```

## Optimal Transport Solvers
The mathematical engine for aligning disjointed spatial slices. It calculates a hybrid cost matrix that heavily penalizes class mismatches while balancing spatial Euclidean distances and gene expression cosine similarities, ultimately solving the entropy-regularized Unbalanced Sinkhorn distance.

```{eval-rst}
.. currentmodule:: deepspatial.data_utils
```

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   compute_uot_coupling
   compute_cost_matrix
```