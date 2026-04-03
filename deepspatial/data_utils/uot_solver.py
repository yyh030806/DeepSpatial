import gc
import numpy as np
import ot
import scipy.sparse as sp

def compute_cost_matrix(x0, g0, c0, x1, g1, c1, alpha_spatial=0.5):
    """
    Computes a hybrid cost matrix balancing spatial distance, gene expression, and cell types.
    """
    eps = 1e-9

    # Spatial distance (Euclidean)
    # x: [N0, 2], y: [N1, 2] -> M: [N0, N1]
    cost_spatial = ot.dist(x0, x1, metric='euclidean')
    # Robust normalization to [0, 1]
    s_max = cost_spatial.max()
    cost_spatial = cost_spatial / (s_max + eps) if s_max > 0 else cost_spatial

    # Gene expression distance (Cosine)
    # Cosine distance is naturally bounded, but we ensure robustness
    cost_gene = ot.dist(g0, g1, metric='cosine')
    g_max = cost_gene.max()
    cost_gene = cost_gene / (g_max + eps) if g_max > 0 else cost_gene

    # Cell type cost (One-hot dot product)
    # 1.0 - dot(c0, c1.T) gives 0 for same type, 1 for different types
    c0_np = c0.numpy() if hasattr(c0, 'numpy') else c0
    c1_np = c1.numpy() if hasattr(c1, 'numpy') else c1
    # Higher values mean higher cost to pair different cell types
    cost_class = np.clip(1.0 - np.dot(c0_np, c1_np.T), 0, 1)

    # Combined cost matrix with implicit large penalty for class mismatch
    # We use a large coefficient (e.g., 10.0) for class mismatch to act as a hard constraint
    C = (alpha_spatial * cost_spatial) + ((1 - alpha_spatial) * cost_gene) + (10.0 * cost_class)

    # Clean up large intermediate distance matrices
    del cost_spatial, cost_gene, cost_class
    gc.collect()

    return C

def compute_uot_coupling(x0, g0, c0, x1, g1, c1,
                         alpha_spatial=0.5,
                         uot_reg=0.8,
                         uot_tau=0.05):
    """
    Computes the Unbalanced Optimal Transport (UOT) coupling matrix between two slices.
    
    Args:
        x0, g0, c0: Spatial, Gene, and One-hot labels for slice 0.
        x1, g1, c1: Spatial, Gene, and One-hot labels for slice 1.
        alpha_spatial: Weight for spatial vs gene distance.
        uot_reg: Entropy regularization (higher = smoother, lower = sparser).
        uot_tau: Marginal relaxation (KL divergence penalty weight).
    """
    # Generate the cost matrix
    C = compute_cost_matrix(x0, g0, c0, x1, g1, c1, alpha_spatial)

    # Define marginal weights (uniform distribution)
    n0, n1 = x0.shape[0], x1.shape[0]
    a = np.ones(n0) / n0
    b = np.ones(n1) / n1

    # Solve Unbalanced Sinkhorn
    # reg_m (uot_tau) controls how much we allow marginals to be violated
    pi = ot.unbalanced.sinkhorn_unbalanced(
        a, b, C,
        reg=max(uot_reg, 0.01),
        reg_m=uot_tau,
        numItermax=100,
        stopThr=1e-3,
        verbose=False
    )

    # 4. Final memory cleanup
    del C, a, b
    gc.collect()

    return pi