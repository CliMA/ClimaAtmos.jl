# Subgrid thermodynamic sampling (variance adjustments study)

> **Document objective:** Outline the theoretical constraints and practical tradeoffs for subgrid thermodynamic sampling in vertically staggered models, with emphasis on reducing an \(N^3\)-style cubature burden to computationally viable \(N^2\)-style schemes—**and** document what this branch **implements** in `ClimaAtmos` (`sgs_distribution`; see §9).

This file is the **single authoritative** narrative for this experiment. It is written from first-principles reasoning and from inspection of the code paths named below—not by trusting superseded scratch notes that used to live in this folder.

**Local-only material** (notebooks, ad hoc Python) may live under `local_python/` (git-ignored; see §10).

---

## 1. Cell geometry and state formulation

In a vertically staggered column, thermodynamic scalars—here **temperature \(T\)** and **specific humidity \(q\)** in the sense used by the microphysics closure—are defined at **cell centers**. The vertical **gradients** (slopes) of \(T\) and \(q\) therefore differ **above and below** the cell midpoint.

Within any vertical cell, a minimal continuous subgrid description uses **five** cell-centered parameters:

- **Means:** \(T\) and \(q\)
- **Variances:** \(\sigma^2(T)\), \(\sigma^2(q)\)
- **Covariance:** expressed via a **correlation** \(\rho(T,q)\) in \((-1,1)\)

Using a correlation (rather than a free off-diagonal covariance) keeps \(\Sigma\) positive definite when mapped from face-based statistics and matches how the model supplies `T′T′`, `q′q′`, and a scalar correlation. Piecewise slopes **above and below** center define a **trajectory in \((T,q)\) space** over the cell’s \(z\) extent; that path is central to any honest layer integral.

---

## 2. Formulations we avoid (and why)

Cheap substitutes for full layer quadrature break in stratified or coarse grids. This section names two failure modes so it is clear what the implemented path is **not** doing.

### 2.1 Cell-wide moment matching (must not use)

Some formulations fit a **single** “equivalent” bivariate Gaussian by matching **bulk** moments over the **entire** cell. **That is a critical error for this framework.**

- **Artificial dispersion:** The physical object is a **mixture** of local distributions along a **path** in \((T,q)\) space. Moment matching **merges path variance with local SGS variance**, producing a single diffuse ellipse that puts tail mass into **unphysical** \((T,q)\) states (e.g. “inside the V” when slopes form a V-shape).
- **Spurious microphysics:** Expensive tails trigger saturation / phase changes where the **path-integrated** measure would not.

We do **not** expose a user-facing “moment-matched layer Gaussian” SGS mode.

**Scope (do not conflate with §7):** §2.1 is about the **joint \((T,q)\)** layer draw: one ellipse replacing a **path mixture** in physical space. The default vertical-profile inner rule in §7 does **not** do that; it discretizes the **scalar** inner marginal as a **composite** of the two half-cell laws (½ DN, ½ UP), each sampled with the **same** Gauss–Legendre nodes on \(p\in(0,1)\), inverting the **single-law** CDF per half (Brent, one-step Halley, or Chebyshev on the centered `uniform⊛Gaussian` for that half). That is a different quadrature design than inverting the mixture CDF \(F_{\mathrm{mix}}(u)=p\) at one \(u\) per node; it avoids a fragile one-shot inverse of \(F_{\mathrm{mix}}\) in production.

### 2.2 Constant variance in \(z\) while means vary

Another shortcut lets **means** vary with height while **holding variances and covariances fixed** to cell-center values over the whole layer—i.e. **no \(z\)-dependence of second moments** (“constant-\(\sigma\) in \(z\)”).

- **Coarse-grid pathology:** At a sharp inversion, high variance at the **center** can be **extrapolated** to cell faces that are dynamically **quiet**, inflating variance at boundaries and biasing cloud onset.

This document treats that shortcut as **physically suspect**; the **implemented** vertical-profile path (§8) instead builds layer means with **half-cell structure** and **\(z\)-aware** second moments where the code path supplies them.

---

## 3. Reference quadrature frameworks

There is **no** closed-form CDF that simultaneously encodes arbitrary **linear-in-\(z\)** mean and covariance tracks with full generality at negligible cost. The practical question is **which numerical sampling strategy** approximates the \(T\times q\times z\) tensor quadrature.

**How this relates to §7 (read this before inferring “we threw away all analytics”).** The impossibility above is about the **global** layer object—unrestricted linear-in-\(z\) **mean and covariance** in the original \((T,q)\) formulation. That is exactly why §3.1–§3.5 discuss **different \(N^2\) layouts** for **how** height is paired with the inner \((T,q)\) quadrature (`SubgridColumnTensor`, LHS-\(z\), Voronoi, barycentric, principal axis). Those modes are **orthogonal** to the default **`SubgridProfileRosenblatt`** path in §7: they are selected by different YAML strings (§7.5). The default path **does not** solve the §3 “full generality” problem in one formula; it uses a **narrower** model (two vertical half cells, Rosenblatt inner \(u\)). Along \(u\), each half uses a **face-anchored** conditional standard deviation (constant on that half in \(u\)). The **forward** inner density along \(u\) is the closed-form **two-component uniform⊛Gaussian mixture** in `subgrid_layer_profile_quadrature.jl`; expectations use the **composite split** inner quadrature (§7). Slopes of turbulent variances still enter through the **face** values that set those `σ_{u|v}`—there is **no** separate analytic branch that treats \(\sigma^2_{u|v}(z)\) as linear inside the half’s inner marginal (an earlier `use_linvar` experiment was removed as incorrect). That is **not** a contradiction with the §3 headline: §3 is about **which cubature** approximates the layer; §7 is about **one** inner discretization for the **default** vertical-profile SGS when you pick `gaussian_vertical_profile` / `lognormal_vertical_profile`.

**Implemented YAML modes** for the five layouts in §3 are listed in §9 (`*_vertical_profile_*` suffixes): full column tensor / LHS-\(z\) pairing / Voronoi anchors / barycentric seeds / principal-axis sweep (plus inner marginal variants from §7).

### 3.1 Full cubature (\(N^3\) baseline truth)

- **Mechanism:** \(N\) nodes in \(T\), \(N\) in \(q\), \(N_z\) in \(z\) (often summarized as \(N\times N\times N\)).
- **Pros:** Pointwise in depth (no vertical cache), excellent coverage of core and tails.
- **Cons:** Prohibitive when expensive physics runs at **every** quadrature point.

### 3.2 Latin-hypercube-style staggering in \(z\) (\(N^2\) stochastic / structured)

- **Mechanism:** Build an \(N\times N\) grid in \((T,q)\) but **permute / scramble** which \(z\) level pairs with each node to reduce redundant overlap.
- **Pros:** Strict \(N^2\) budget, remains pointwise.
- **Cons:** Randomness or rigid cycling can induce **temporal noise** when thresholds are sharp.

### 3.3 Voronoi subsampling (\(N^2\) from a dense pool)

- **Mechanism:** Generate a high-resolution \(N^3\) pool, cluster with Voronoi into \(N^2\) representatives.
- **Pros:** Shape-aware reduction toward the truth distribution.
- **Cons:** Needs **storage** for the dense pool; heavier distance work.

### 3.4 Barycentric / seed-based lumping (\(N^2\) deterministic)

- **Mechanism:** Seed \(N^2\) nodes along the \(z\)-path; stream generated \(N^3\) candidates and **accumulate mass** into nearest seeds (barycentric update).
- **Pros:** Deterministic; can be GPU-friendly with modest accumulators.
- **Cons:** Coverage depends on seeding; needs accumulator state.

### 3.5 Local principal-axis sweep (cheap 1D inner reduction)

- **Mechanism:** At each \(z\), whiten by \(\rho\) and place 1D quadrature along the dominant axis, then map back with local \(\sigma_T,\sigma_q\).
- **Pros:** Pointwise, minimal machinery.
- **Cons:** **Under-resolves** 2D saturation geometry; **axis snapping** when \(\rho\) crosses zero between levels injects discontinuities.

---

## 4. Framework comparison summary

| Framework | Complexity | Memory | Deterministic | Primary strength | Primary weakness |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Full cubature** | \(O(N^3)\) | Pointwise | Yes | Exact reference | Too expensive operationally |
| **Latin hypercube** | \(O(N^2)\) | Pointwise | Often no | Very fast | Stochastic / cyclic artifacts |
| **Voronoi subsampling** | Iterative | High (\(N^3\) pool) | Yes | Near-optimal shape | Pool storage + cost |
| **Barycentric lumping** | \(O(N^3)\!\to\!N^2)\) | Low (\(N^2\) accumulators) | Yes | Deterministic \(N^2\) | Seed sensitivity |
| **Principal-axis sweep** | \(O(N)\) | Pointwise | Yes | Extremely cheap | Under-dispersed; axis snaps |

---

## 5. Target: a faithful \(N^2\) reduction

The long-run goal is a method that can take the **information content** of an \(N^3\) cubature and **deterministically** reduce it to an optimal \(N^2\) subset **without** global moment matching and **without** inventing variance at the wrong heights—ideally with minimal vertical cache.

Until such a reduction exists in closed form, **barycentric-style** and **LHS-style** families remain the most plausible operational compromises among the table above.

---

## 6. Implementation status (`ClimaAtmos`)

| Topic | Status |
| :--- | :--- |
| **Baseline SGS** at cell center (`gaussian`, `lognormal`, `mean`) | Implemented (`sgs_distribution`; §9). |
| **Vertical layer-mean profile SGS** (YAML `*_vertical_profile*`) | Implemented: default `SubgridProfileRosenblatt` (`gaussian_vertical_profile`), column tensor and alternate \(N^2\) layouts (§9). |
| **Column tensor** (§3.1; full \(N_z\times N^2\) cubature) | Implemented as `*_vertical_profile_full_cubature`. |
| **LHS-style \(z\) pairing** (§3.2) | Implemented as `*_vertical_profile_lhs_z`. |
| **Voronoi-style anchors** (§3.3) | Implemented (`*_vertical_profile_voronoi`). |
| **Barycentric buckets** (§3.4) | Implemented (`*_vertical_profile_barycentric`; matching `quadrature_order` required). |
| **Principal-axis sweep** (§3.5) | Implemented (`*_vertical_profile_principal_axis`). |
| **Regression / inner marginal** | `test/parameterized_tendencies/microphysics/sgs_quadrature.jl` exercises `SubgridProfileRosenblatt` with `Bracketed` / `Halley` / `ChebyshevLogEta` inner rules, split-`p` inner bookkeeping, unit mass for `quadrature_order` 1–5, full-profile integrals, `Float32`, NamedTuple 1M outputs, and scalar mixture CDF algebra separately. |

**Note:** Voronoi/barycentric kernels use small allocations for index clustering; principal-axis replaces the inner \(N\times N\) fluctuation tensor by a correlation-axis approximation (limitations in §3.5).

---

## 7. Default vertical-profile path (`SubgridProfileRosenblatt`): two halves, split-`p` inner quadrature

YAML `gaussian_vertical_profile` / `lognormal_vertical_profile` select **`SubgridProfileRosenblatt`** in `src/parameterized_tendencies/microphysics/sgs_distribution_types.jl`. The implementation is **two vertical half cells**—below-center (DN) and above-center (UP)—each built from face-anchored slopes of the **cell means**, then rotated into `(u,v)` using a chosen **mean-gradient axis** (which half’s ``(\partial_z\bar T,\partial_z\bar q)`` defines the inner ``u`` direction; not advection).

### 7.1 What matches the “README-first” analytic story

- **Means:** piecewise linear in \(z\) within each half (different slopes DN vs UP).
- **Inner marginal along \(u\)** (after rotating fluctuations): the **density** is still the closed-form **two-component mixture** of uniform×Gaussian convolutions (one convolution per half-cell segment). **Quadrature** uses **one** Gauss–Legendre rule of order \(N\) on \(p\in(0,1)\) with **split-`p`**: when both **Rosenblatt** axes (below-center / above-center means) are valid, nodes with \(p_i<\tfrac12\) use only the **`p_{\mathrm{dn}}`** pack (below-center mean slopes for `M_inv` / outer \(v\)) and remapped leg quantile \(2p_i\in(0,1)\); nodes with \(p_i>\tfrac12\) use only **`p_{\mathrm{up}}`** and \(2p_i-1\); odd \(N\) can use a one-node `p\approx\tfrac12` **mid** bridge. That is **\(N^2\)** microphysics samples per column at order \(N\), not **\(2N^2\)**. When only one **axis** is valid, a **single** mean-gradient pass may still split \(p\) if both half-widths \(L_{dn},L_{up}\) are positive on that pass.
- **Second moments on the inner \(u\) marginal:** each half uses **one** scalar \(\sigma_{u|v}\) from the **half-center** reconstruction (DN at \(-H/4\), UP at \(+H/4\) in \(z\)), held **constant on that half** in \(u\). Turbulent **variance slopes** (\(\partial_z \sigma_T^2\), \(\partial_z \sigma_q^2\)) enter through those half-center values (and through alternate §3 layouts if selected). The correlation scalar \(\rho\) is still **cell-center**; there is **no** \(\partial_z\rho\) model.

### 7.2 Physical \((T,q)\) mapping

With **two valid axes**, the implementation uses **separate** `M_inv` and half packs for the inner nodes assigned to the DN vs UP legs via split-`p` (see `subgrid_layer_profile_quadrature.jl`), not a **second** full \(N^2\) tensor and not an extra `\tfrac12` sum over two identical inner rules.

### 7.3 Per-leg quantile solvers (YAML `_inner_*`)

These select **only** how each **single** shifted half-cell `uniform⊛Gaussian` quantile is evaluated when that leg is active (`subgrid_layer_profile_quadrature.jl`). **Bracketed** and **Halley** use the remapped leg probability \(p_\mathrm{leg}\in(0,1)\) directly. **Chebyshev** tables are defined on the Gauss–Legendre abscissas for order `N`; under split-`p`, \(p_\mathrm{leg}\) is generally **not** one of those nodes, so the implementation **linearly interpolates in \(p\)** between the two tabulated Chebyshev quantiles that bracket \(p_\mathrm{leg}\)—still only `ConvolutionQuantilesChebyshevLogEta` table evaluations, not a silent Brent/Halley substitution.

| YAML suffix | Julia type | Role |
| :--- | :--- | :--- |
| *(default)* | `ConvolutionQuantilesHalley` | One Halley step on the **centered** `uniform[-L/2,L/2]⊛N(0,s²)` law at the **leg** remapped \(p_\mathrm{leg}\), then shift to DN/UP \(u\). |
| `_inner_bracketed` | `ConvolutionQuantilesBracketed` | Brent (exact bracketed root) on that same centered single law at \(p_\mathrm{leg}\). |
| `_inner_chebyshev` | `ConvolutionQuantilesChebyshevLogEta` | Chebyshev tabulation in \(\tau(\log_{10}\eta)\), \(\eta=s/L\), at tabulated \(p\)-nodes; under split-`p`, linear **\(p\)**-interpolation between the two bracketing tabulated nodes (per leg, own \((L,s)\)). |

**Mixture CDF:** A scalar \(F_{\mathrm{mix}}^{-1}(p)\) is **not** used in this profile integrator; mixture CDF/PDF helpers exist for analysis and tests. Per-leg inverses are composed by split-`p` quadrature as above.

### 7.4 Tensor split (`N/2 × N` vs odd \(N\))

Separately from the vertical-profile default, `tensor_gh_row_split` in `sgs_quadrature.jl` partitions the first Hermite index for row bookkeeping: `n1 = N ÷ 2`, `n2 = N - n1`. For **odd** \(N\), the remainder lands in the second block (one extra node on the “upper” partition); calibration Python mirrors this split.

### 7.5 Other vertical-profile modes

Column tensor, LHS-\(z\), Voronoi, barycentric, and principal-axis variants are orthogonal sampling layouts (§3 / §9); they do not replace the half-cell mixture unless you select those YAML strings.

---

## 8. Code map (ClimaAtmos)

| Concern | Location |
| :--- | :--- |
| Scalar half-cell variance / covariance kernels | `src/utils/variance_statistics.jl` |
| YAML \(\to\) distribution object | `get_sgs_distribution` in `src/solver/model_getters.jl` |
| Layer-mean profile quadrature | `src/parameterized_tendencies/microphysics/subgrid_layer_profile_quadrature.jl` |
| SGS quadrature driver / `get_physical_point` | `src/parameterized_tendencies/microphysics/sgs_quadrature.jl` |
| Distribution types / inner quadrature selectors | `src/parameterized_tendencies/microphysics/sgs_distribution_types.jl` |
| Cache / saturation coupling | `src/cache/microphysics_cache.jl` |

---

## 9. `sgs_distribution` names (this branch)

**Baseline (no vertical layer-profile integral inside the SGS PDF):**

| Value | Meaning |
| :--- | :--- |
| `gaussian` | Bivariate Gaussian \((T,q)\) fluctuations (with physical clamping where applicable). |
| `lognormal` | Log-normal \(q\), Gaussian \(T\), copula (common default). |
| `mean` | Grid-mean only (`GridMeanSGS`). |

**Vertical profile (layer-mean path with half-cell / \(z\)-structured quadrature inside the layer):**

| Value | Meaning |
| :--- | :--- |
| `gaussian_vertical_profile` | Default vertical-profile Gaussian SGS: `SubgridProfileRosenblatt` + per-leg `ConvolutionQuantilesHalley` with split-`p` inner quadrature (§7). |
| `gaussian_vertical_profile_full_cubature` | Column-tensor vertical quadrature (`SubgridColumnTensor`; dense \(N_z\times N^2\) path, §3.1). |
| `gaussian_vertical_profile_lhs_z` | Latin-square style pairing of Hermite `(i,j)` with one GL \(z\) level (`SubgridLatinHypercubeZ`). |
| `gaussian_vertical_profile_principal_axis` | Dominant-correlation-axis 1D Hermite inner rule (`SubgridPrincipalAxisLayer`). |
| `gaussian_vertical_profile_voronoi` | Index-space Voronoi anchors + pooled weights (`SubgridVoronoiRepresentatives`). |
| `gaussian_vertical_profile_barycentric` | Barycentric accumulation on `(k,i)` seeds (`SubgridBarycentricSeeds`; requires `quadrature_order` parity). |
| `gaussian_vertical_profile_inner_bracketed` | `ConvolutionQuantilesBracketed`: per-leg Brent on each half-cell law (§7.3). |
| `gaussian_vertical_profile_inner_halley` | `ConvolutionQuantilesHalley`: explicit name for the default per-leg one-step Halley (§7.3). |
| `gaussian_vertical_profile_inner_chebyshev` | `ConvolutionQuantilesChebyshevLogEta`: per-leg Chebyshev tables (§7.3). |
| `lognormal_vertical_profile` | Default vertical-profile lognormal \(q\) / Gaussian \(T\). |
| `lognormal_vertical_profile_full_cubature` | Column tensor; lognormal \(q\). |
| `lognormal_vertical_profile_lhs_z` | LHS-\(z\) pairing; lognormal \(q\). |
| `lognormal_vertical_profile_principal_axis` | Principal-axis inner rule; lognormal \(q\). |
| `lognormal_vertical_profile_voronoi` | Voronoi anchors; lognormal \(q\). |
| `lognormal_vertical_profile_barycentric` | Barycentric seeds; lognormal \(q\). |
| `lognormal_vertical_profile_inner_bracketed` | Same as Gaussian row; lognormal \(q\). |
| `lognormal_vertical_profile_inner_halley` | Same as Gaussian row; lognormal \(q\). |
| `lognormal_vertical_profile_inner_chebyshev` | Same as `gaussian_vertical_profile_inner_chebyshev`; lognormal \(q\). |

Pair with **`quadrature_order`** as in the main ClimaAtmos configuration.

---

## 10. Calibration / EKI layout (this experiment)

Paths (`simulation_output/...`), **`varfix_on` / `varfix_off`**, and sweep drivers live under `lib/` and `scripts/`.

- **Varfix “off”:** effective `sgs_distribution` is a **baseline** name from §9 (first table).
- **Varfix “on”:** effective `sgs_distribution` is a **`(gaussian|lognormal)_vertical_profile*`** name.

Typical steps: set `VA_EXPERIMENT_CONFIG`, build observations when LES stats exist, run `scripts/run_calibration.jl` or sweep helpers. `expected_git_branch` in the experiment YAML should match the branch you run on.

---

## 11. `local_python/` (ignored)

Optional notebooks and exploratory scripts may live under:

`calibration/experiments/variance_adjustments/local_python/`

listed in this folder’s `.gitignore`. Not required to build or test ClimaAtmos.
