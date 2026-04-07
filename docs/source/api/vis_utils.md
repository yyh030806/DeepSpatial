# Visualization Utilities

The `deepspatial.vis_utils` module provides a comprehensive suite of plotting tools tailored for 3D spatial transcriptomics. It supports both publication-quality static rendering via `matplotlib` and dynamic, browser-based exploration via `plotly` and `ipywidgets`.

```{eval-rst}
.. currentmodule:: deepspatial.vis_utils
```


## Static Plotting
Generate high-resolution, static visualizations optimized for publications and reports. These functions support rendering full 3D scatter plots, simulating physical tissue sectioning (virtual slices), and projecting 3D volumes onto 2D orthogonal planes.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   plot_3d_labels
   plot_virtual_slice
   plot_orthogonal_projections
   plot_z_distribution
```

## Interactive Plotting
Browser-based, interactive rendering built on plotly and Jupyter Widgets. These tools allow users to dynamically rotate, zoom, and slice through the reconstructed 3D tissue volumes in real-time.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   interactive_3d_labels
   interactive_3d_expression
   interactive_spatial_range_widget
```