# Module

The `deepspatial.module` serves as the training and inference orchestrator. It encapsulates joint continuous Flow Matching, modality-normalized velocity MSE losses, Exponential Moving Average (EMA) updates, and continuous integration solvers. Spatial coordinates, gene expression, and one-hot cell-type states are interpolated along the same linear path; no cross-entropy loss is used for cell-type evolution.

```{eval-rst}
.. currentmodule:: deepspatial.module
```

## Lightning Module
The core engine responsible for managing the optimization lifecycle and the generation phase.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   DeepSpatialModule
```

## Key Methods
Fundamental operations managed by the module. `sample(..., reverse_time=True)` performs explicit reverse-time ODE integration through the equivalent increasing-time parameterization $q(u)=s(1-u)$ and $dq/du=-v(q,1-u)$. Original depth endpoints and the forward-oriented `delta_z` remain unchanged. Reverse-time sampling is supported for ODE mode only.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   DeepSpatialModule.configure_optimizers
   DeepSpatialModule.training_step
   DeepSpatialModule.sample
```
