# Module

The `deepspatial.module` serves as the training and inference orchestrator for the framework. Built upon PyTorch Lightning, it elegantly encapsulates the Flow Matching training objective, multi-modal loss computation, Exponential Moving Average (EMA) weight updates, and continuous integration solvers.

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
Fundamental operations managed by the module. The sample method is particularly critical as it executes the ODE/SDE integration process for 3D volume reconstruction.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   DeepSpatialModule.training_step
   DeepSpatialModule.sample
```