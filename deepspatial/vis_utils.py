import warnings
from typing import Optional, Union, Dict, List, Tuple

import numpy as np
import pandas as pd
import anndata as ad
import plotly.express as px
import plotly.graph_objects as go


def _extract_coords(
    adata: ad.AnnData, 
    spatial_key: str = 'spatial', 
    z_key: str = 'z_coord',
    max_points: Optional[int] = None,
    random_state: int = 42
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Extracts spatial coordinates and handles safe downsampling."""
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


def _apply_plotly_layout(fig: go.Figure, title: str, bg_color: str) -> None:
    """Standardizes layout, camera, and background for 3D scenes."""
    fig.update_layout(
        title=title,
        scene=dict(
            aspectmode='data',
            xaxis=dict(showbackground=False, visible=False),
            yaxis=dict(showbackground=False, visible=False),
            zaxis=dict(showbackground=False, visible=False)
        ),
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        margin=dict(l=0, r=0, b=0, t=40),
        font=dict(color='white' if bg_color == 'black' else 'black')
    )


def plot_3d_cell_labels(
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
    save_html: Optional[str] = None
) -> go.Figure:
    """Interactive 3D scatter plot for categorical metadata."""
    if color_col not in adata.obs:
        raise ValueError(f"Column '{color_col}' not found in adata.obs")
        
    df, mask = _extract_coords(adata, spatial_key, z_key, max_points)
    df[color_col] = adata.obs[color_col].astype(str).values[mask]

    if focus_categories is not None:
        df['Display'] = df[color_col].where(df[color_col].isin(focus_categories), 'Other')
        active_palette = {cat: palette.get(cat, px.colors.qualitative.Plotly[i % 10]) 
                          for i, cat in enumerate(focus_categories)} if palette else {}
        active_palette['Other'] = 'rgba(200, 200, 200, 0.1)' if bg_color == 'white' else 'rgba(50, 50, 50, 0.1)'
        
        color_col = 'Display'
        palette = active_palette

    fig = px.scatter_3d(
        df, x='x', y='y', z='z',
        color=color_col,
        color_discrete_map=palette,
        opacity=opacity
    )
    
    fig.update_traces(marker=dict(size=point_size, line=dict(width=0)))
    _apply_plotly_layout(fig, title, bg_color)
    
    if save_html:
        fig.write_html(save_html)
        
    return fig