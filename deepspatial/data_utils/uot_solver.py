import gc
import numpy as np
import ot

def compute_cost_matrix(
    x0,
    g0,
    c0,
    x1,
    g1,
    c1,
    alpha_spatial=0.5,
    cell_type_penalty=50.0,
):
    """
    Computes a hybrid cost matrix balancing spatial distance, gene expression, and cell types.

    Parameters
    ----------
    x0 : numpy.ndarray or torch.Tensor
        Spatial coordinates for the source slice, shape `(N0, 2)`.
    g0 : numpy.ndarray or torch.Tensor
        Gene expression matrix for the source slice, shape `(N0, G)`.
    c0 : numpy.ndarray or torch.Tensor
        One-hot encoded cell types for the source slice, shape `(N0, C)`.
    x1 : numpy.ndarray or torch.Tensor
        Spatial coordinates for the target slice, shape `(N1, 2)`.
    g1 : numpy.ndarray or torch.Tensor
        Gene expression matrix for the target slice, shape `(N1, G)`.
    c1 : numpy.ndarray or torch.Tensor
        One-hot encoded cell types for the target slice, shape `(N1, C)`.
    alpha_spatial : float, optional
        Weight balancing the spatial vs. gene distance (between 0 and 1). 
        By default 0.5.
    cell_type_penalty : float, optional
        Additive cost for pairing cells with different labels. The default of
        50.0 is the value used for the manuscript experiments.

    Returns
    -------
    numpy.ndarray
        The combined cost matrix of shape `(N0, N1)`.
    """
    eps = 1e-8

    # Spatial distance (Euclidean)
    # x: [N0, 2], y: [N1, 2] -> M: [N0, N1]
    cost_spatial = ot.dist(x0, x1, metric='euclidean')
    cost_spatial = cost_spatial / (cost_spatial.max() + eps) + eps

    # Gene expression distance (Cosine)
    cost_gene = ot.dist(g0, g1, metric='cosine')
    cost_gene = cost_gene / (cost_gene.max() + eps) + eps

    # Cell type cost (One-hot dot product)
    # 1.0 - dot(c0, c1.T) gives 0 for same type, 1 for different types
    c0_np = c0.numpy() if hasattr(c0, 'numpy') else c0
    c1_np = c1.numpy() if hasattr(c1, 'numpy') else c1
    cost_class = np.clip(1.0 - np.dot(c0_np, c1_np.T), eps, 1)

    C = (
        (alpha_spatial * cost_spatial)
        + ((1 - alpha_spatial) * cost_gene)
        + (cell_type_penalty * cost_class)
    )

    # Clean up large intermediate distance matrices
    del cost_spatial, cost_gene, cost_class
    gc.collect()

    return C

def compute_uot_coupling(x0, g0, c0, x1, g1, c1,
                         alpha_spatial=0.5,
                         cell_type_penalty=50.0,
                         uot_reg=0.8,
                         uot_tau=0.05,
                         num_itermax=50,
                         stop_thr=1e-2):
    """
    Computes the Unbalanced Optimal Transport (UOT) coupling matrix between two slices.
    
    Parameters
    ----------
    x0 : numpy.ndarray or torch.Tensor
        Spatial coordinates for the source slice.
    g0 : numpy.ndarray or torch.Tensor
        Gene expression matrix for the source slice.
    c0 : numpy.ndarray or torch.Tensor
        One-hot encoded cell types for the source slice.
    x1 : numpy.ndarray or torch.Tensor
        Spatial coordinates for the target slice.
    g1 : numpy.ndarray or torch.Tensor
        Gene expression matrix for the target slice.
    c1 : numpy.ndarray or torch.Tensor
        One-hot encoded cell types for the target slice.
    alpha_spatial : float, optional
        Weight for spatial distance in the underlying cost matrix. By default 0.5.
    cell_type_penalty : float, optional
        Cost multiplier for cross-cell-type couplings. By default 50.0, as in
        the manuscript experiments.
    uot_reg : float, optional
        Entropy regularization parameter. Higher values lead to smoother, more dispersed 
        couplings; lower values yield sparser, more deterministic matchings. By default 0.8.
    uot_tau : float, optional
        Marginal relaxation weight (KL divergence penalty). Controls how strictly the 
        mass conservation is enforced. Smaller values allow more mass creation/destruction. 
        By default 0.05.
    num_itermax : int, optional
        Maximum Sinkhorn iterations. By default 50.
    stop_thr : float, optional
        Sinkhorn stopping threshold. By default 1e-2.

    Returns
    -------
    numpy.ndarray
        The calculated optimal transport coupling matrix `pi` of shape `(N0, N1)`.
    """
    # Generate the cost matrix
    C = compute_cost_matrix(
        x0,
        g0,
        c0,
        x1,
        g1,
        c1,
        alpha_spatial=alpha_spatial,
        cell_type_penalty=cell_type_penalty,
    )

    # Define marginal weights (uniform distribution)
    n0, n1 = x0.shape[0], x1.shape[0]
    a = np.ones(n0) / n0 + 1e-8
    b = np.ones(n1) / n1 + 1e-8

    # Solve Unbalanced Sinkhorn
    # reg_m (uot_tau) controls how much we allow marginals to be violated
    pi = ot.unbalanced.sinkhorn_unbalanced(
        a, b, C,
        reg=max(uot_reg, 0.01),
        reg_m=uot_tau,
        numItermax=num_itermax,
        stopThr=stop_thr,
        verbose=False
    )

    # 4. Final memory cleanup
    del C, a, b
    gc.collect()

    return pi
