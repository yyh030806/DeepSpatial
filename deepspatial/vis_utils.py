from typing import Optional, Union, Dict, List, Tuple

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse
from scipy.ndimage import gaussian_filter1d

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.express as px
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display


# ==============================================================================
# Internal Helper Functions
# ==============================================================================

def _extract_coords(
    adata: ad.AnnData, 
    spatial_key: str = 'spatial', 
    z_key: str = 'z_coord',
    max_points: Optional[int] = None,
    random_state: int = 42
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Extract spatial coordinates from AnnData with optional downsampling.
    """
    if spatial_key not in adata.obsm:
        raise KeyError(f"'{spatial_key}' not found in adata.obsm.")
    if z_key not in adata.obs:
        raise KeyError(f"'{z_key}' not found in adata.obs.")

    n_cells = adata.n_obs
    mask = np.ones(n_cells, dtype=bool)
    
    if max_points is not None and n_cells > max_points:
        np.random.seed(random_state)
        idx = np.random.choice(n_cells, max_points, replace=False)
        mask = np.zeros(n_cells, dtype=bool)
        mask[idx] = True

    df = pd.DataFrame({
        'x': adata.obsm[spatial_key][mask, 0],
        'y': adata.obsm[spatial_key][mask, 1],
        'z': adata.obs[z_key].values[mask]
    }, index=adata.obs_names[mask])
    
    return df, mask


def _apply_plotly_layout(
    fig: go.Figure, 
    title: str, 
    bg_color: str,
    width: Optional[int] = None,
    height: Optional[int] = None
) -> None:
    """
    Standardize layout and styling for 3D Plotly figures.
    """
    fig.update_layout(
        title=title,
        width=width,   
        height=height,  
        scene=dict(
            aspectmode='data',
            xaxis=dict(showbackground=False, visible=False),
            yaxis=dict(showbackground=False, visible=False),
            zaxis=dict(showbackground=False, visible=False)
        ),
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        margin=dict(l=0, r=0, b=0, t=40),
        font=dict(color='white' if bg_color == 'black' else 'black'),
        legend=dict(itemsizing='constant', font=dict(size=14))
    )


# ==============================================================================
# Static Plotting
# ==============================================================================

def plot_3d_labels(
    adata: ad.AnnData,
    color_col: str = 'cell_class',
    palette: Optional[Dict[str, str]] = None,
    spatial_key: str = 'spatial',
    z_key: str = 'z_coord',
    azim: float = -60.0,
    elev: float = 30.0,
    z_stretch: float = 1.0,
    point_size: float = 1.0,
    alpha: float = 0.8,
    max_points: int = 100000,
    bg_color: str = "white",
    save_pdf: Optional[str] = None,
    show: bool = True
) -> Optional[plt.Figure]:
    """
    Generate a static 3D scatter plot colored by categorical labels.

    Parameters
    ----------
    adata : ad.AnnData
        Annotated data matrix.
    color_col : str
        Column name in `adata.obs` representing the category.
    palette : dict, optional
        Mapping of categories to hex color codes. Defaults to 'tab20'.
    spatial_key : str
        Key in `adata.obsm` containing XY spatial coordinates.
    z_key : str
        Key in `adata.obs` containing Z coordinates.
    azim : float
        Azimuthal viewing angle in degrees.
    elev : float
        Elevation viewing angle in degrees.
    z_stretch : float
        Scaling factor for the Z-axis aspect ratio.
    point_size : float
        Scatter point size.
    alpha : float
        Marker opacity (0.0 to 1.0).
    max_points : int
        Max number of cells to render to prevent memory overflow.
    bg_color : str
        Figure background color.
    save_pdf : str, optional
        Path to save the output as a PDF file.
    show : bool
        Whether to display the plot immediately. If False, returns the figure.

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object if `show=False`, else None.
    """
    df, mask = _extract_coords(adata, spatial_key, z_key, max_points)
    labels = adata.obs[color_col].astype(str).values[mask]
    categories = np.unique(labels)
    
    if palette is None:
        cmap = plt.get_cmap('tab20')
        palette = {cat: mcolors.to_hex(cmap(i % 20)) for i, cat in enumerate(categories)}
        
    colors = [palette.get(lbl, '#808080') for lbl in labels]

    fig = plt.figure(figsize=(10, 8), facecolor=bg_color)
    ax = fig.add_subplot(111, projection='3d', facecolor=bg_color)
    
    ax.scatter(df['x'], df['y'], df['z'], c=colors, s=point_size, alpha=alpha, edgecolors='none')
    ax.view_init(elev=elev, azim=azim)
    
    x_ptp = df['x'].max() - df['x'].min()
    y_ptp = df['y'].max() - df['y'].min()
    z_ptp = df['z'].max() - df['z'].min()
    max_range = max(x_ptp, y_ptp)
    if max_range > 0:
        ax.set_box_aspect((x_ptp / max_range, y_ptp / max_range, (z_ptp / max_range) * z_stretch))

    ax.grid(False)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.set_edgecolor('w')
        pane.set_alpha(0)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    if bg_color != "white":
        for label in [ax.xaxis.label, ax.yaxis.label, ax.zaxis.label]:
            label.set_color('white')
        ax.tick_params(colors='white')

    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label=cat,
                   markerfacecolor=palette.get(cat, '#808080'), markersize=8) 
        for cat in categories
    ]
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.05, 0.5), 
              frameon=False, labelcolor='white' if bg_color != 'white' else 'black')

    plt.tight_layout()
    if save_pdf:
        plt.savefig(save_pdf, format='pdf', dpi=300, bbox_inches='tight', facecolor=bg_color)
        
    if show:
        plt.show()
        return None
    return fig


def plot_virtual_slice(
    adata: ad.AnnData,
    plane_normal: Union[str, Tuple[float, float, float]] = 'sagittal',
    thickness: float = 10.0,
    color_col: str = 'cell_class',
    palette: Optional[Dict[str, str]] = None,
    center: Optional[Tuple[float, float, float]] = None,
    spatial_key: str = 'spatial',
    z_key: str = 'z_coord',
    azim: float = -60.0,
    elev: float = 30.0,
    point_size: float = 2.0,
    alpha: float = 0.8,
    bg_color: str = "white",
    save_pdf: Optional[str] = None,
    return_adata: bool = False,
    show: bool = True
) -> Union[None, ad.AnnData, plt.Figure, Tuple[plt.Figure, ad.AnnData]]:
    """
    Simulate physical sectioning and generate a virtual 2D/3D slice plot.
    
    Parameters
    ----------
    adata : ad.AnnData
        Annotated data matrix.
    plane_normal : str or tuple
        Normal vector of the cutting plane. Accepts predefined strings 
        ('coronal', 'sagittal', 'transverse', 'axial') or a custom 3D vector.
    thickness : float
        Thickness of the virtual slice.
    color_col : str
        Column in `adata.obs` for coloring points.
    palette : dict, optional
        Mapping of categories to hex colors.
    center : tuple, optional
        The (X, Y, Z) point the plane passes through. Defaults to data centroid.
    spatial_key : str
        Key in `adata.obsm` for XY coordinates.
    z_key : str
        Key in `adata.obs` for Z coordinate.
    azim : float
        Azimuthal viewing angle (only applies to custom 3D angled slices).
    elev : float
        Elevation viewing angle (only applies to custom 3D angled slices).
    point_size : float
        Marker size.
    alpha : float
        Marker opacity.
    bg_color : str
        Figure background color.
    save_pdf : str, optional
        Path to save output PDF.
    return_adata : bool
        If True, returns the subsetted AnnData object containing only the slice.
    show : bool
        If True, display the plot immediately.

    Returns
    -------
    Mixed
        Depends on `return_adata` and `show` toggles.
    """
    df, _ = _extract_coords(adata, spatial_key, z_key, max_points=None)
    coords = df[['x', 'y', 'z']].values
    center_pt = np.array(center) if center else coords.mean(axis=0)
    
    predefined_planes = {
        'coronal': {'n': [1, 0, 0], 'h': 'y', 'v': 'z', 'desc': 'Coronal Slice (Y-Z Plane)'},
        'sagittal': {'n': [0, 1, 0], 'h': 'x', 'v': 'z', 'desc': 'Sagittal Slice (X-Z Plane)'},
        'transverse': {'n': [0, 0, 1], 'h': 'x', 'v': 'y', 'desc': 'Transverse Slice (X-Y Plane)'},
        'axial': {'n': [0, 0, 1], 'h': 'x', 'v': 'y', 'desc': 'Axial Slice (X-Y Plane)'}
    }
    
    use_2d_projection = False
    normal_key = None
    
    if isinstance(plane_normal, str):
        normal_key = plane_normal.lower()
        if normal_key not in predefined_planes:
            raise ValueError(f"Unknown predefined plane: {plane_normal}")
        normal_vec = predefined_planes[normal_key]['n']
        use_2d_projection = True
    else:
        normal_vec = plane_normal
        norm_arr = np.array(normal_vec, dtype=float)
        norm_arr /= np.linalg.norm(norm_arr)
        
        # Fallback to 2D standard projection if custom vector aligns with standard axes
        for key, info in predefined_planes.items():
            if np.allclose(norm_arr, info['n']):
                use_2d_projection = True
                normal_key = key
                break

    normal = np.array(normal_vec, dtype=float)
    normal /= np.linalg.norm(normal)
    
    distances = np.abs(np.dot(coords - center_pt, normal))
    mask = distances <= (thickness / 2.0)
    
    sliced_adata = adata[mask].copy()
    sliced_adata.obs['slice_distance'] = distances[mask]
        
    df_slice = df[mask].copy()
    labels = sliced_adata.obs[color_col].astype(str).values
    categories = np.unique(labels)
    
    if palette is None:
        cmap = plt.get_cmap('tab20')
        palette = {cat: mcolors.to_hex(cmap(i % 20)) for i, cat in enumerate(categories)}
    colors = [palette.get(lbl, '#808080') for lbl in labels]

    if use_2d_projection:
        plane_info = predefined_planes[normal_key]
        h_axis, v_axis = plane_info['h'], plane_info['v']
        
        fig, ax = plt.subplots(figsize=(10, 8), facecolor=bg_color)
        ax.set_facecolor(bg_color)
        ax.scatter(df_slice[h_axis], df_slice[v_axis], c=colors, s=point_size, alpha=alpha, edgecolors='none')
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_xlabel(h_axis.upper())
        ax.set_ylabel(v_axis.upper())
        title_text = plane_info['desc']

    else:
        fig = plt.figure(figsize=(10, 8), facecolor=bg_color)
        ax = fig.add_subplot(111, projection='3d', facecolor=bg_color)
        ax.scatter(df_slice['x'], df_slice['y'], df_slice['z'], c=colors, s=point_size, alpha=alpha, edgecolors='none')
        ax.view_init(elev=elev, azim=azim)
        
        x_ptp = df_slice['x'].max() - df_slice['x'].min()
        y_ptp = df_slice['y'].max() - df_slice['y'].min()
        z_ptp = df_slice['z'].max() - df_slice['z'].min()
        max_range = max(x_ptp, y_ptp)
        if max_range > 0:
            ax.set_box_aspect((x_ptp / max_range, y_ptp / max_range, z_ptp / max_range))

        ax.grid(False)
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.set_edgecolor('w')
            pane.set_alpha(0)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        title_text = f"Custom Angled Slice ({thickness}µm thick)"

    if bg_color != "white":
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        if not use_2d_projection: ax.zaxis.label.set_color('white')
        ax.tick_params(colors='white')

    plt.title(title_text, color='w' if bg_color != 'white' else 'k')
    plt.tight_layout()
    
    if save_pdf:
        plt.savefig(save_pdf, format='pdf', dpi=300, bbox_inches='tight', facecolor=bg_color)
        
    if show:
        plt.show()
        return sliced_adata if return_adata else None
    
    plt.close()
    return (fig, sliced_adata) if return_adata else fig


def plot_z_distribution(
    adata: ad.AnnData,
    color_col: str = 'cell_class',
    palette: Optional[Dict[str, str]] = None,
    spatial_key: str = 'spatial',
    z_key: str = 'z_coord',
    n_points: int = 200,
    smooth_sigma: float = 3.0,
    fig_height: float = 3.5,
    width_per_z_unit: float = 0.05,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    z_range: Optional[Tuple[float, float]] = None,
    show_legend: bool = True,
    save_pdf: Optional[str] = None,
    show: bool = True
) -> Optional[plt.Figure]:
    """
    Render a smoothed stacked area chart representing cell proportions along the Z-axis.
    
    Parameters
    ----------
    adata : ad.AnnData
        Annotated data matrix.
    color_col : str
        Column in `adata.obs` representing the cell category.
    palette : dict, optional
        Mapping of categories to colors.
    spatial_key : str
        Key in `adata.obsm` for XY coordinates.
    z_key : str
        Key in `adata.obs` for Z coordinate.
    n_points : int
        Number of interpolation bins along the Z-axis.
    smooth_sigma : float
        Standard deviation for the Gaussian smoothing kernel.
    fig_height : float
        Fixed height of the figure in inches.
    width_per_z_unit : float
        Dynamic width scaling factor (inches per unit of Z-axis span).
    x_range, y_range, z_range : tuple of float, optional
        (min, max) coordinates to mask the data before analysis.
    show_legend : bool
        Whether to draw the category legend.
    save_pdf : str, optional
        Path to save PDF output.
    show : bool
        If True, display the plot immediately.
        
    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object if `show=False`, else None.
    """
    if spatial_key not in adata.obsm or z_key not in adata.obs:
        raise KeyError(f"Required spatial/Z keys not found in AnnData.")

    coords = adata.obsm[spatial_key]
    z_coords = adata.obs[z_key].values
    mask = np.ones(len(z_coords), dtype=bool)

    if x_range is not None: 
        mask &= (coords[:, 0] >= x_range[0]) & (coords[:, 0] <= x_range[1])
    if y_range is not None: 
        mask &= (coords[:, 1] >= y_range[0]) & (coords[:, 1] <= y_range[1])
    if z_range is not None: 
        mask &= (z_coords >= z_range[0]) & (z_coords <= z_range[1])

    df = pd.DataFrame({
        'Z': z_coords[mask],
        'CellType': adata.obs[color_col].iloc[mask].values
    })
        
    if df.empty:
        return None

    z_min, z_max = np.floor(df['Z'].min()), np.ceil(df['Z'].max())
    z_span = max(z_max - z_min, 1.0)
    fig_width = max(z_span * width_per_z_unit, 2.0) 
    
    bins = np.linspace(z_min, z_max, n_points + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2 
    df['Z_bin'] = pd.cut(df['Z'], bins=bins)
    
    count_table = pd.crosstab(df['Z_bin'], df['CellType'], dropna=False)
    
    if pd.api.types.is_categorical_dtype(adata.obs[color_col]):
        all_cell_types = adata.obs[color_col].cat.categories
    else:
        all_cell_types = np.unique(adata.obs[color_col].dropna().astype(str))
        
    count_table = count_table.reindex(columns=all_cell_types, fill_value=0)

    smoothed_counts = {}
    for ct in all_cell_types:
        smoothed = gaussian_filter1d(count_table[ct].values.astype(float), sigma=smooth_sigma)
        smoothed_counts[ct] = np.clip(smoothed, a_min=0, a_max=None)
        
    smoothed_df = pd.DataFrame(smoothed_counts, index=count_table.index)
    prop_table = smoothed_df.div(smoothed_df.sum(axis=1), axis=0).fillna(0)

    if palette is None:
        cmap = plt.get_cmap('tab20')
        palette = {cat: mcolors.to_hex(cmap(i % 20)) for i, cat in enumerate(all_cell_types)}
    plot_colors = [palette.get(ct, '#CCCCCC') for ct in all_cell_types]

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor='white')
    y_data = [prop_table[ct].values for ct in all_cell_types]

    ax.stackplot(
        bin_centers, y_data, labels=all_cell_types, 
        colors=plot_colors, edgecolor='none', alpha=0.95        
    )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(z_min, z_max)
    ax.set_ylim(0, 1.0)
    ax.set_xticks([]) 
    ax.set_yticks([0, 1])

    if show_legend:
        ax.legend(title=color_col, bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)

    plt.tight_layout()
    if save_pdf:
        plt.savefig(save_pdf, dpi=300, bbox_inches='tight', transparent=True)
        
    if show:
        plt.show()
        return None
    return fig


def plot_orthogonal_projections(
    adata: ad.AnnData, 
    color_col: str = 'cell_class',
    palette: Optional[Dict[str, str]] = None,
    spatial_key: str = 'spatial',
    z_key: str = 'z_coord',
    point_size: float = 0.5,
    alpha: float = 0.5,
    max_points: Optional[int] = None,
    bg_color: str = "white",
    save_png: Optional[str] = None,
    show: bool = True
) -> Optional[plt.Figure]:
    """
    Generate static 2D orthogonal projections (XY, XZ, YZ) of the 3D data.
    
    Parameters
    ----------
    adata : ad.AnnData
        Annotated data matrix.
    color_col : str
        Column in `adata.obs` representing categories.
    palette : dict, optional
        Mapping of categories to colors.
    spatial_key : str
        Key in `adata.obsm` for XY coordinates.
    z_key : str
        Key in `adata.obs` for Z coordinate.
    point_size : float
        Scatter marker size.
    alpha : float
        Marker transparency.
    max_points : int, optional
        Limit number of points rendered.
    bg_color : str
        Background color.
    save_png : str, optional
        Path to save as a static image.
    show : bool
        If True, display the plot immediately.
        
    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object if `show=False`, else None.
    """
    df, mask = _extract_coords(adata, spatial_key, z_key, max_points)
    labels = adata.obs[color_col].astype(str).values[mask]
    categories = np.unique(labels)
    
    if palette is None:
        cmap = plt.get_cmap('tab20')
        palette = {cat: mcolors.to_hex(cmap(i % 20)) for i, cat in enumerate(categories)}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    if bg_color != "white":
        fig.patch.set_facecolor(bg_color)
        for ax in axes:
            ax.set_facecolor(bg_color)
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
            ax.tick_params(colors='white')

    projections = [
        ('x', 'y', 'X-Y Plane', axes[0]), 
        ('x', 'z', 'X-Z Plane', axes[1]),
        ('y', 'z', 'Y-Z Plane', axes[2])
    ]
    
    for x_col, y_col, title, ax in projections:
        for cat in categories:
            idx = labels == cat
            ax.scatter(df.loc[idx, x_col], df.loc[idx, y_col], s=point_size, alpha=alpha, 
                       c=palette.get(cat, '#808080'), label=cat, edgecolors='none')
            
        ax.set_xlabel(x_col.upper())
        ax.set_ylabel(y_col.upper())
        ax.set_title(title)
        ax.set_aspect('equal', adjustable='datalim')

    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc='center right', bbox_to_anchor=(1.12, 0.5), 
               markerscale=10, fontsize='small',
               facecolor=bg_color, edgecolor='none', 
               labelcolor='white' if bg_color != 'white' else 'black')
    
    plt.tight_layout()
    if save_png:
        plt.savefig(save_png, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
        
    if show:
        plt.show()
        return None
    return fig


# ==============================================================================
# Interactive Widgets & Plots
# ==============================================================================

def interactive_3d_labels(
    adata: ad.AnnData, 
    color_col: str = 'cell_class',
    focus_categories: Optional[List[str]] = None,
    palette: Optional[Dict[str, str]] = None,
    spatial_key: str = 'spatial',
    z_key: str = 'z_coord',
    point_size: float = 1.5,
    opacity: float = 0.8,
    bg_color: str = "white",
    max_points: int = 250000,
    title: str = "3D Cell Type Distribution",
    width: Optional[int] = None,
    height: Optional[int] = None,
    save_html: Optional[str] = None
) -> go.Figure:
    """
    Generate an interactive Plotly 3D scatter plot for categorical metadata.
    
    Parameters
    ----------
    adata : ad.AnnData
        Annotated data matrix.
    color_col : str
        Column in `adata.obs` representing categories.
    focus_categories : list of str, optional
        Specific categories to highlight. Non-highlighted cells become faint.
    palette : dict, optional
        Mapping of categories to colors.
    spatial_key : str
        Key in `adata.obsm` for XY coordinates.
    z_key : str
        Key in `adata.obs` for Z coordinate.
    point_size : float
        Marker size.
    opacity : float
        Marker opacity (0.0 to 1.0).
    bg_color : str
        Background color.
    max_points : int
        Maximum rendering limit for browser performance.
    title : str
        Plot title.
    width, height : int, optional
        Dimensions of the rendering canvas in pixels.
    save_html : str, optional
        Path to save as a standalone interactive HTML file.
        
    Returns
    -------
    plotly.graph_objects.Figure
    """
    df, mask = _extract_coords(adata, spatial_key, z_key, max_points)
    df[color_col] = adata.obs[color_col].astype(str).values[mask]

    if focus_categories is not None:
        df['Display'] = df[color_col].where(df[color_col].isin(focus_categories), 'Other')
        active_palette = {cat: palette.get(cat, px.colors.qualitative.Plotly[i % 10]) 
                          for i, cat in enumerate(focus_categories)} if palette else {}
        active_palette['Other'] = 'rgba(200, 200, 200, 0.1)' if bg_color == 'white' else 'rgba(50, 50, 50, 0.1)'
        color_col, palette = 'Display', active_palette

    fig = px.scatter_3d(
        df, x='x', y='y', z='z', 
        color=color_col, color_discrete_map=palette, opacity=opacity
    )
    
    fig.update_traces(marker=dict(size=point_size, line=dict(width=0)))
    _apply_plotly_layout(fig, title, bg_color, width=width, height=height)
    
    if save_html:
        fig.write_html(save_html)
        
    return fig


def interactive_3d_expression(
    adata: ad.AnnData, 
    gene_name: str,
    spatial_key: str = 'spatial',
    z_key: str = 'z_coord',
    vmin_pct: float = 1.0,
    vmax_pct: float = 99.0,
    point_size: float = 2.0,
    opacity: float = 0.8,
    colorscale: str = 'Viridis',
    bg_color: str = "white",
    max_points: int = 250000,
    title: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    save_html: Optional[str] = None
) -> go.Figure:
    """
    Generate an interactive Plotly 3D scatter plot for continuous gene expression.
    
    Parameters
    ----------
    adata : ad.AnnData
        Annotated data matrix.
    gene_name : str
        Feature name present in `adata.var_names`.
    spatial_key : str
        Key in `adata.obsm` for XY coordinates.
    z_key : str
        Key in `adata.obs` for Z coordinate.
    vmin_pct, vmax_pct : float
        Lower and upper percentile bounds for clipping expression values.
    point_size : float
        Marker size.
    opacity : float
        Marker opacity.
    colorscale : str
        Plotly continuous color scale name (e.g., 'Viridis', 'Plasma').
    bg_color : str
        Background color.
    max_points : int
        Maximum rendering limit.
    title : str, optional
        Plot title. Defaults to feature name.
    width, height : int, optional
        Dimensions of the rendering canvas in pixels.
    save_html : str, optional
        Path to save as a standalone interactive HTML file.
        
    Returns
    -------
    plotly.graph_objects.Figure
    """
    if gene_name not in adata.var_names:
        raise ValueError(f"Feature '{gene_name}' not found in adata.var_names.")
        
    df, mask = _extract_coords(adata, spatial_key, z_key, max_points)
    
    expr = adata[:, gene_name].X
    if scipy.sparse.issparse(expr):
        expr = expr.toarray().flatten()[mask]
    else:
        expr = expr[mask]
    
    vmin, vmax = np.percentile(expr[expr > 0], [vmin_pct, vmax_pct]) if (expr > 0).any() else (0, 1)
    df['expression'] = np.clip(expr, vmin, vmax)

    fig = px.scatter_3d(
        df, x='x', y='y', z='z', 
        color='expression', color_continuous_scale=colorscale,
        range_color=[vmin, vmax], opacity=opacity
    )
    
    fig.update_traces(marker=dict(size=point_size, line=dict(width=0)))
    _apply_plotly_layout(fig, title or f"Expression: {gene_name}", bg_color, width=width, height=height)
    
    if save_html:
        fig.write_html(save_html)
        
    return fig


def interactive_spatial_range_widget(
    adata: ad.AnnData,
    color_col: str = 'cell_class',
    palette: Optional[Dict[str, str]] = None,
    spatial_key: str = 'spatial',
    z_key: str = 'z_coord',
    point_size: float = 2.0,
    opacity: float = 0.8,
    bg_color: str = "black",
    width: Optional[int] = None,
    height: Optional[int] = 800,
    show: bool = True
) -> Optional[widgets.VBox]:
    """
    Generate a Jupyter widget for dynamically slicing and projecting 3D data onto a 2D plane.
    
    Parameters
    ----------
    adata : ad.AnnData
        Annotated data matrix.
    color_col : str
        Column in `adata.obs` for coloring points.
    palette : dict, optional
        Mapping of categories to colors.
    spatial_key : str
        Key in `adata.obsm` for XY coordinates.
    z_key : str
        Key in `adata.obs` for Z coordinate.
    point_size : float
        Marker size.
    opacity : float
        Marker opacity.
    bg_color : str
        Canvas background color.
    width, height : int, optional
        Dimensions of the widget rendering area.
    show : bool
        If True, renders the widget inline via IPython.display.
        
    Returns
    -------
    ipywidgets.VBox or None
        Container widget if `show=False`, else None.
    """
    df, _ = _extract_coords(adata, spatial_key, z_key, max_points=None)
    df['label'] = adata.obs[color_col].astype(str).values
    categories = np.unique(df['label'])
    
    if palette is None:
        palette = {cat: px.colors.qualitative.Plotly[i % 10] for i, cat in enumerate(categories)}

    axis_map = {
        'Z-axis (XY Plane)': ('z', 'x', 'y'),
        'Y-axis (XZ Plane)': ('y', 'x', 'z'),
        'X-axis (YZ Plane)': ('x', 'y', 'z')
    }

    component_width = f"{width}px" if width is not None else '100%'

    fig = go.FigureWidget()
    filter_ax, h_ax, v_ax = axis_map['Z-axis (XY Plane)']
    
    for cat in categories:
        cat_df = df[df['label'] == cat]
        fig.add_trace(go.Scatter(
            x=cat_df[h_ax], y=cat_df[v_ax], 
            mode='markers', name=cat,
            marker=dict(color=palette.get(cat, 'gray'), size=point_size, opacity=opacity),
            text=cat_df[filter_ax].round(3),
            hovertemplate=f"<b>%{{name}}</b><br>{filter_ax.upper()}: %{{text}}<extra></extra>"
        ))
        
    fig.update_layout(
        title="Dynamic Spatial Range Projection", 
        width=width, height=height,
        plot_bgcolor=bg_color, paper_bgcolor=bg_color,
        font=dict(color='white' if bg_color == 'black' else 'black'),
        xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
        margin=dict(l=0, r=0, t=40, b=0), 
        legend=dict(itemsizing='constant', title=color_col, font=dict(size=14))
    )

    axis_dropdown = widgets.Dropdown(
        options=list(axis_map.keys()),
        value='Z-axis (XY Plane)',
        description='Project Axis:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='300px')
    )

    range_slider = widgets.FloatRangeSlider(
        value=[df[filter_ax].min(), df[filter_ax].max()],
        min=df[filter_ax].min(), max=df[filter_ax].max(),
        step=(df[filter_ax].max() - df[filter_ax].min()) / 500.0,
        description='Range:',
        readout_format='.2f', continuous_update=False,
        layout=widgets.Layout(width=component_width)
    )

    def update_all(change):
        sel_axis = axis_dropdown.value
        f_ax, hor_ax, ver_ax = axis_map[sel_axis]
        
        if change['owner'] == axis_dropdown:
            range_slider.min = df[f_ax].min()
            range_slider.max = df[f_ax].max()
            range_slider.value = [df[f_ax].min(), df[f_ax].max()]
            range_slider.step = (df[f_ax].max() - df[f_ax].min()) / 500.0

        val_lower, val_upper = range_slider.value
        
        with fig.batch_update():
            for i, cat in enumerate(categories):
                mask = (df['label'] == cat) & (df[f_ax] >= val_lower) & (df[f_ax] <= val_upper)
                fig.data[i].x = df.loc[mask, hor_ax]
                fig.data[i].y = df.loc[mask, ver_ax]
                fig.data[i].text = df.loc[mask, f_ax].round(3)
                fig.data[i].hovertemplate = f"<b>%{{name}}</b><br>{f_ax.upper()}: %{{text}}<extra></extra>"
            
            fig.layout.xaxis.title = hor_ax.upper()
            fig.layout.yaxis.title = ver_ax.upper()

    axis_dropdown.observe(update_all, names='value')
    range_slider.observe(update_all, names='value')
    
    container = widgets.VBox(
        [widgets.HBox([axis_dropdown]), range_slider, fig],
        layout=widgets.Layout(
            width=component_width,     
            max_width=component_width, 
            align_items='flex-start',
            margin='0px',             
            padding='0px',              
            overflow='hidden'           
        )
    )
    
    if show:
        display(container)
        return None
    return container