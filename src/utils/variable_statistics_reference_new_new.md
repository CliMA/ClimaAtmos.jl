# Warped-Space 2D Quadrature for Bivariate Uniform-Normal Distributions

## Overview
When modeling a sub-grid linear gradient subject to local correlated Gaussian turbulence, evaluating non-linear thermodynamic thresholds requires discrete quadrature. A standard Gauss-Hermite tensor grid applied in physical space fails because the off-diagonal covariance terms (correlation) shear the tensor "box," pushing corner nodes into unphysical, zero-density regions.

This algorithm resolves the geometry mismatch by constructing the $O(N^2)$ tensor grid in the **Mahalanobis (Whitened) Space**, where the structural gradient and local turbulence are strictly independent. By applying a Cornish-Fisher expansion and an empirical variance lock along the gradient axis, the physical boundaries are strictly respected for any quadrature order $N \ge 1$. To maximize parallel efficiency, the algorithm is split into a macroscopic cell setup and a strictly pointwise execution phase.

## Phase 1: Cell Pre-Computation (Joint Setup)
This phase evaluates macroscopic cell properties. It is executed exactly once per grid cell to avoid redundant matrix inversions or reduction sums.

**Inputs:**
* $\mathbf{a}, \mathbf{b}$: State vectors at the start and end of the discrete interval.
* $\Sigma_{local}$: $2 \times 2$ local covariance matrix.
* $N$: Quadrature order.
* $z_{std}, w_{std}$: Standard Probabilist Gauss-Hermite nodes and weights.

**Procedure:**
1. **Mahalanobis Whitening:** Compute the Cholesky decomposition $\Sigma_{local} = \mathbf{L}\mathbf{L}^T$.
   Transform the bounds into isotropic space:
   $\tilde{\mathbf{a}} = \mathbf{L}^{-1}\mathbf{a}$
   $\tilde{\mathbf{d}} = \mathbf{L}^{-1}(\mathbf{b}) - \tilde{\mathbf{a}}$
   $\tilde{L} = \|\tilde{\mathbf{d}}\|$

2. **Orthonormal Basis:**
   Define the parallel vector $\tilde{\mathbf{e}}_1 = \tilde{\mathbf{d}} / \tilde{L}$.
   Define the perpendicular vector $\tilde{\mathbf{e}}_2 = [-\tilde{\mathbf{e}}_1^{(y)}, \tilde{\mathbf{e}}_1^{(x)}]^T$.

3. **1D Uniform-Normal Properties (Parallel Axis):**
   Calculate the true variance and kurtosis of the parallel projection:
   $V_\parallel = 1 + \frac{\tilde{L}^2}{12}$
   $K_\parallel = \frac{\tilde{L}^4}{80} + \frac{\tilde{L}^2}{2} + 3$
   $\gamma = \max\left(-1.2, \frac{K_\parallel}{V_\parallel^2} - 3\right)$

4. **Cornish-Fisher Squash & Empirical Lock:**
   Calculate the expansion coefficient $k = \gamma / 24$.
   Apply the transformation to the standard nodes: $z_{raw} = z_{std} + k(z_{std}^3 - 3z_{std})$.
   Compute the empirical variance sum: $V_{emp} = \sum_{n=1}^{N} w_n z_{raw, n}^2$.
   Normalize to lock variance: $z_{cf} = z_{raw} / \sqrt{V_{emp}}$.

**Outputs:** $\mathbf{L}$, $\tilde{\mathbf{a}}$, $\tilde{\mathbf{e}}_1$, $\tilde{\mathbf{e}}_2$, $\tilde{L}$, $V_\parallel$, and the locked 1D array $z_{cf}$.

---

## Phase 2: Pointwise Execution (Hot Loop)
This phase evaluates the physical states. It requires zero loops or reductions, making it fully independent for pointwise kernel execution.

**Inputs for Thread $(i, j)$:**
* Indices $i, j \in [1, N]$.
* The macroscopic outputs from Phase 1.
* Standard nodes and weights: $z_{std, j}$, $w_{std, i}$, $w_{std, j}$.

**Procedure:**
1. **Coordinate Mapping in Warped Space:**
   Compute the parallel coordinate (shifted by the mean of the interval):
   $p = \frac{\tilde{L}}{2} + \sqrt{V_\parallel} z_{cf, i}$
   Compute the perpendicular coordinate (pure Gaussian):
   $q = z_{std, j}$

2. **Re-Warp to Physical Space:**
   Construct the point in warped space: $\tilde{\mathbf{x}} = \tilde{\mathbf{a}} + p\tilde{\mathbf{e}}_1 + q\tilde{\mathbf{e}}_2$.
   Map back to physical state space: $\mathbf{x}_{phys} = \mathbf{L}\tilde{\mathbf{x}}$.

3. **Weight Combination:**
   The pointwise quadrature weight is exactly $W_{i, j} = w_{std, i} w_{std, j}$.