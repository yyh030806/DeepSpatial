# Data Utilities

The Data Utilities module provides the foundational data structures and mathematical solvers required to construct continuous training trajectories. It handles the complex task of linking discrete cells across adjacent 2D slices using Unbalanced Optimal Transport (UOT).

---

## PyTorch Dataset

The core dataset class. It computes adjacent-slice UOT pairings and formats spatial coordinates, gene expression, and one-hot cell-type annotations as PyTorch tensors. `n_samples_base` is a total trajectory-pair budget distributed across slice pairs in proportion to the product of their cell counts.

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
The mathematical engine for aligning disjointed spatial slices. The cost is

$$
C=\alpha d_{\mathrm{spatial}}+(1-\alpha)d_{\mathrm{gene}}
  +\lambda_{\mathrm{type}}d_{\mathrm{type}},
$$

where the spatial and gene distance matrices are independently max-normalized.
The solver uses uniform endpoint marginals and entropy-regularized unbalanced
Sinkhorn transport. `cell_type_penalty`, `uot_reg`, `uot_tau`, `num_itermax`,
and `stop_thr` are explicit API parameters rather than hidden constants.

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
