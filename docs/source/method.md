# Implementation-aligned method

This page describes the algorithm implemented by the current public code. It
is also the reference for terminology used in the revised manuscript.

## 1. UOT coupling between adjacent slices

For cells in two adjacent observed slices, DeepSpatial builds a hybrid cost
from normalized spatial distance, gene-expression cosine distance, and a
cell-type mismatch penalty:

$$
C_{ij} = \alpha d^{x}_{ij} + (1-\alpha)d^{g}_{ij}
       + \lambda_{\mathrm{type}}d^{c}_{ij}.
$$

The first two distance matrices are independently divided by their maximum.
The cell-type term is zero up to a small numerical epsilon for a matched label
and one for a mismatched label. The coupling is obtained with entropy-regularized
unbalanced Sinkhorn transport. Sampled endpoint pairs are drawn from the
normalized coupling with replacement.

`DeepSpatial.setup_data` exposes all solver controls: `alpha_spatial`,
`cell_type_penalty`, `uot_reg`, `uot_tau`, `uot_num_itermax`, and
`uot_stop_thr`. These values are stored in `data_config` when a checkpoint is
saved.

## 2. Joint continuous flow matching

Spatial coordinates $x$, gene expression $g$, and one-hot cell-type vectors
$c$ follow the same linear probability path. For each modality
$m\in\{x,g,c\}$,

$$
m_t=(1-t)m_0+t m_1, \qquad u_m=m_1-m_0.
$$

Cell type is therefore a **continuous relaxation**, not a discrete categorical
process during training. The network predicts three velocity fields and all
three are optimized with MSE velocity matching. The implementation first takes
the per-sample mean-squared error over each modality. This averaging provides
the single feature-dimension normalization used by the total objective:

$$
\mathcal L_m =
  \mathbb E\!\left[\frac{1}{d_m}\lVert v_m-u_m\rVert_2^2\right],
\qquad
\mathcal L=\mathcal L_x+\lambda_g\mathcal L_g+\lambda_c\mathcal L_c.
$$

The defaults are $\lambda_g=0.1$ and $\lambda_c=10$. There is no
cross-entropy objective in the final joint flow. During inference, the
continuous cell-type state is integrated together with $x$ and $g$ and is
converted to a label by `argmax` only when the trajectory output is assembled.

The standard pipeline expects cell-type annotations because they contribute to
the UOT cost, the joint flow state, and the reconstructed labels.

## 3. GiT architecture

GiT uses the following modality-specific projections:

- spatial coordinates: `Linear(2, H)`, producing one spatial token;
- gene expression: non-overlapping gene patches, each projected by a two-layer
  MLP (`Linear`–`SiLU`–`Linear`);
- continuous one-hot cell-type state: `Linear(K, H)`, used as global
  conditioning.

Sinusoidal timestep embeddings are used for flow time, physical depth, and the
slice gap. Their conditioning vector modulates adaLN-Zero Transformer blocks.
Separate output heads predict spatial, gene-expression, and cell-type
velocities. The implementation does not use a two-layer spatial MLP or
`nn.Embedding` for cell type.

## 4. Explicit bidirectional ODE generation

Forward trajectories integrate physical flow time from $t=0$ to $t=1$. For a
terminal cell sampled from the upper slice, reverse generation starts at
physical time $t=1$. The numerical solver still advances an auxiliary time
$u:0\rightarrow1$, using the exact reverse-time reparameterization

$$
q(u)=s(1-u), \qquad
\frac{dq}{du}=-v(q(u),1-u).
$$

The original endpoint ordering $(z_0,z_1)$ and forward-oriented
$\Delta z=z_1-z_0$ are retained. The implementation does **not** swap
endpoints, negate $\Delta z$, run a forward solve, or flip a completed
trajectory to emulate reverse integration.

For a requested target depth $t$, the parent direction is sampled as

$$
P(\mathrm{forward}\mid t)=1-t,
\qquad
P(\mathrm{backward}\mid t)=t.
$$

Thus direction selection is depth dependent rather than
`Bernoulli(0.5)`.

## 5. Density-aware depths and gene sparsification

Let the endpoint slices contain $N_0$ and $N_1$ cells. Virtual depths are
sampled with density proportional to the linear interpolation of endpoint cell
counts,

$$
p(t)=\frac{2[(1-t)N_0+tN_1]}{N_0+N_1}.
$$

The inverse CDF is used directly; when $N_0=N_1$, this reduces to uniform
depth sampling. The number of virtual cells in a gap is

$$
N_{\mathrm{virtual}}=
\max\!\left(1,\left\lfloor
\frac{N_0+N_1}{2}\frac{|z_1-z_0|}{\mathrm{thickness}}
\right\rfloor\right).
$$

After ODE generation, each virtual cell retains the top-$k$ predicted genes,
where $k$ is the number of non-zero genes in its sampled parent cell. This is
the final gene-sparsification step.

There is no oversampled raw population, spatial subset optimization, or
post-hoc spatial pruning. Every generated virtual cell is retained. The
corresponding private helper is therefore named `_generate_virtual_cells`.

## 6. API defaults and reported benchmark settings

Public API defaults provide a general starting point. The reported Fig. 2
benchmark explicitly overrides the parameters shown below; reproducibility
scripts should set them rather than relying on defaults.

| Parameter | Public API default | Reported Fig. 2 setting |
|---|---:|---:|
| `n_samples_base` | 50,000 | total number of cells in the observed input slices |
| `cell_type_penalty` | 50.0 | 50.0 |
| `uot_reg` | 0.8 | 0.1 |
| `uot_tau` | 0.05 | 1.0 |
| `uot_num_itermax` | 50 | 50 |
| `uot_stop_thr` | $10^{-2}$ | $10^{-2}$ |
| training epochs | 100 | 10 |

All checkpoint metadata includes the data, model, and training configuration so
that a saved run can be audited without inferring values from later defaults.
