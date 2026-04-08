# math_sanity

Scalar and plotting checks for GaussianSGS quadrature, subcell **(1/12)Δz²** geometry (same structure as `materialize_sgs_quadrature_moments!`), and optional resolution transfer.

## Drivers

- `run_all.jl` — summary figures + optional Gaussian mosaics (default **64** panels: nondimensional **ρ × R_q × R_T** with **`μ_T` fixed**). Mosaics use **one** reference thickness **`geo.dz`** only; degenerate “no subcolumn” physics is explored through the **`(R_q, R_T)`** grid (e.g. large **R** ⇒ small **σ**), not a separate **Δz = 0** figure batch. **Per-cell PNGs are opt-in** (`write_cell_pngs=true`); file count is **`2 × n_panels`** (unclamped/clamped) under `figures/sweep/cells/` (default **128** files).
- **`figures/mosaic/`** — regenerated PNGs with stems **`mosaic_gaussian_{unclamped|clamped}_subcellRef_p{n}`** where **`n = length(ρ)·length(μ_T)·n_inner²`**. Each has optional **`_physical.png`** for `(q,T)`. Run `run_all.jl` / `run_sweep.jl` to repopulate. For **legacy 4⁴ (256)** mosaics, use **`mathsanity_legacy_gaussian_grid_ranges_sigma()`** when calling the plotter.
- `run_sweep.jl` — mosaics only (no numeric self-checks).

Default mosaics use a **dimensionless inner grid** in **`R_q`, `R_T`** (`defaults.jl`); recovered turbulent **`σ_q`, `σ_T`** satisfy `σ_q = Δz·(∂q/∂z)/R_q`, `σ_T = Δz·(∂T/∂z)/R_T` with the same reference gradients as the geometry closure. Legacy **`(σ_q, σ_T)`** grids (including zeros on an endpoint) live in `mathsanity_legacy_gaussian_grid_ranges_sigma()`.

**Primary** mega-mosaic axes are **`(z_q, z_T)`** with `z_q = (q - μ_q)/σ_q`, `z_T = (T - μ_T)/σ_T` using the **panel turbulent** `σ_q`, `σ_T` (the same σ as in the `R` definitions). **`(q, T)`** mosaics are written as sibling files `*_physical.png`.

Each large mosaic includes a **REFERENCE** caption block (`mathsanity_mosaic_reference_debug_caption` in `defaults.jl`): mosaic `Δz` in metres, gradients, `μ_q`, outer `ρ`/`μ_T` span, inner `R_q`/`R_T` span (or legacy σ note), `geo` knob values, and standardized `σ_floor`. Regenerate after changing defaults.

**Clamped** maps are a **secondary** diagnostic (piecewise boundaries); unclamped figures show the core Gaussian + geometry algebra most clearly.

## Standardized bin / column segment

For a linear profile in height, displacement **`δz`** from the column center maps to

`(z_q, z_T) = (δz/Δz) · (R_q, R_T)` with `R_q = (Δz·∂q/∂z)/σ_q`, `R_T = (Δz·∂T/∂z)/σ_T`.

If the subcolumn spans **`δz ∈ [-Δz/2, +Δz/2]`**, the segment in **`(z_q, z_T)`** runs from **`−½(R_q, R_T)`** to **`+½(R_q, R_T)`** through the origin (mean at `(0,0)`). Euclidean chord length is **`√(R_q² + R_T²)`** (half-extent **`½√(R_q² + R_T²)`**).

In physical **`(q, T)`**, the same structure is the line through **`(μ_q, μ_T)`** along **`(∂q/∂z, ∂T/∂z)`** over **`|δz| ≤ Δz/2`**.
