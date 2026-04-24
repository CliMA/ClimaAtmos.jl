# Central defaults for `math_sanity` drivers — override via kwargs in plot/grid functions.

using Printf

"""
Recover turbulent `(σ_q, σ_T)` from dimensionless **R**-ratios and reference column geometry.

`R_q = (Δz·∂q/∂z)/σ_q`, `R_T = (Δz·∂T/∂z)/σ_T` with `∂T/∂z = (∂T/∂θ_li)(∂θ_li/∂z)` (same as
`mathsanity_column_grad_dq_dT`). Use the **same** `Δz` and gradients as in
`mathsanity_sgs_quad_moments_with_geometry`.
"""
function mathsanity_sigma_from_R_q_R_T(
    dz::FT,
    dq_dz::FT,
    dtheta_dz::FT,
    dT_dθ::FT,
    R_q::FT,
    R_T::FT,
) where {FT <: Real}
    dT_dz = dT_dθ * dtheta_dz
    σ_q = dz * dq_dz / R_q
    σ_T = dz * dT_dz / R_T
    return σ_q, σ_T
end

"""
If `ranges` carries `R_q` / `R_T` (dimensionless grid), return `(σ_q_vec, σ_T_vec)` from
`σ_q = Δz·(∂q/∂z)/R_q`, `σ_T = Δz·(∂T/∂z)/R_T` using **`geo`** gradients and `dz_use` (defaults to `geo.dz`).

If `ranges` already stores `σ_q` / `σ_T` vectors (no `R_q` / `R_T`), returns those vectors unchanged.
"""
function mathsanity_inner_sigmas_for_gaussian_grid(ranges, geo; dz_use = nothing)
    if hasproperty(ranges, :R_q) && hasproperty(ranges, :R_T)
        FT = typeof(geo.dz)
        dz = something(dz_use, FT(geo.dz))
        dq_dz = FT(geo.dq_dz)
        dtheta_dz = FT(geo.dtheta_dz)
        dT_dθ = FT(geo.dT_dθ)
        dT_dz = dT_dθ * dtheta_dz
        Rq = collect(ranges.R_q)
        RT = collect(ranges.R_T)
        σq = map(R -> dz * dq_dz / R, Rq)
        σT = map(R -> dz * dT_dz / R, RT)
        return σq, σT
    else
        return collect(ranges.σ_q), collect(ranges.σ_T)
    end
end

"""Defaults for GaussianSGS scatter / σ-scan (same order of magnitude as column SCM toys)."""
function mathsanity_default_scatter_knobs(FT::Type{<:Real} = Float32)
    (; μ_q = FT(0.012), μ_T = FT(288.0), σ_q = FT(0.002), σ_T = FT(0.35), ρ = FT(0.65), N_quad = 5, n_mc = 8000, lim_z = FT(3.2))
end

"""Defaults for `materialize_sgs_quadrature_moments!`-style scalar geometry."""
function mathsanity_default_geometric_knobs(FT::Type{<:Real} = Float32)
    (;
        qq = FT(4e-6),
        TT = FT(0.04),
        ρ_param = FT(0.62),
        dz = FT(400.0),
        dq_dz = FT(8e-6),
        dtheta_dz = FT(0.015),
        dT_dθ = FT(0.48),
    )
end

"""
Floor for **standardized** `(q,T) → (z_q,z_T)` maps when `σ_q` or `σ_T` is zero on the grid.
Physical `(q,T)` mosaic panels do not divide by `σ`; standardized `(z_q,z_T)` panels use this floor when `σ` hits zero on the grid.
"""
mathsanity_sigma_floor_standardized(::Type{FT}) where {FT} = FT(1e-12)
mathsanity_sigma_floor_standardized() = mathsanity_sigma_floor_standardized(Float64)

"""Δz samples for **variance vs thickness** curves (`plots_geometric.jl`); unrelated to mosaic panel count."""
function mathsanity_default_dz_scan_list(FT::Type{<:Real} = Float32)
    dz_nom = mathsanity_default_geometric_knobs(FT).dz
    (FT(0), FT(200), dz_nom, FT(800))
end

"""Panel count for a Gaussian mosaic: `length(ρ)·length(μ_T)·nᵢ²` with `nᵢ = length(R_q)` or `length(σ_q)`."""
function mathsanity_gaussian_mosaic_panel_count(ranges)
    n_i = if hasproperty(ranges, :R_q)
        length(collect(ranges.R_q))
    else
        length(collect(ranges.σ_q))
    end
    n_ρ = length(collect(ranges.ρ))
    n_μ = length(collect(ranges.μ_T))
    return n_ρ * n_μ * n_i * n_i
end

"""
Stem (no extension) for mosaic PNGs. Drivers use **`dz == geo.dz`** only: tag **`subcellRef`**.

Suffix includes panel count as **`_p{panel_count}`** (from `mathsanity_gaussian_mosaic_panel_count(ranges)`). Any other `dz` (REPL) uses **`subcell{metres}m`**
(`Δz = 0` → `subcell0m`).
"""
function mathsanity_mosaic_output_basename_stem(clamped::Bool, dz::FT, geo, n_panels::Int) where {FT <: Real}
    c = clamped ? "clamped" : "unclamped"
    pref = "mosaic_gaussian_$(c)_"
    mid = if dz == FT(geo.dz)
        "subcellRef"
    else
        dz_tag = isinteger(dz) ? string(Int(dz)) : @sprintf("%.1f", dz)
        "subcell$(dz_tag)m"
    end
    return pref * mid * "_p$(n_panels)"
end

"""
Default **4³ = 64** Gaussian mosaic grid: nondimensional inner **`R_q`, `R_T`** (length 4 each) and outer **`ρ`** (length 4);
**`μ_T` fixed** to `mathsanity_default_scatter_knobs().μ_T` (one value) so the swept controls are **`ρ`, `R_q`, `R_T`** only.

`R_q = (Δz_ref·∂q/∂z)/σ_q`, `R_T = (Δz_ref·∂T/∂z)/σ_T` at `dz_ref` from `mathsanity_default_geometric_knobs`.

For **4⁴** sweeps with a **μ_T** axis and inner axes given directly as **`(σ_q, σ_T)`** (including zeros on an endpoint), use `mathsanity_gaussian_grid_ranges_sigma_axes()` or widen `μ_T` yourself.
"""
function mathsanity_default_gaussian_grid_ranges(FT::Type{<:Real} = Float32)
    g = mathsanity_default_geometric_knobs(FT)
    sc = mathsanity_default_scatter_knobs(FT)
    dz = g.dz
    dq_dz = g.dq_dz
    dtheta_dz = g.dtheta_dz
    dT_dθ = g.dT_dθ
    dT_dz = dT_dθ * dtheta_dz
    # R_q = range(dz * dq / FT(0.004), dz * dq_dz / FT(0.00133); length = 4)
    # R_T = range(dz * dTdz / FT(0.55), dz * dT_dz / FT(0.183); length = 4)


    R_q = FT[Inf, range(dz * dq_dz / FT(0.00133), dz * dq_dz / FT(0.004); length = 3)...] # Inf maps to no initial variance. There will be a correction if dz_ref is not 0
    R_T = FT[Inf, range(dz * dT_dz / FT(0.18300), dz * dT_dz / FT(0.550); length = 3)...] # Inf maps to no initial variance. There will be a correction if dz_ref is not 0

    return (;
        R_q = R_q,
        R_T = R_T,
        ρ = range(FT(-0.5), FT(1); length = 4),
        μ_T = range(sc.μ_T, sc.μ_T; length = 1),
        dz_ref = dz,
        dq_dz_ref = dq_dz,
        dtheta_dz_ref = dtheta_dz,
        dT_dθ_ref = dT_dθ,
    )
end

"""`(σ_q, σ_T, ρ, μ_T)` tensor grid with inner axes in **physical σ** (`σ` may be zero on an endpoint)."""
function mathsanity_gaussian_grid_ranges_sigma_axes(FT::Type{<:Real} = Float32)
    (;
        σ_q = range(FT(0.0), FT(0.004); length = 4),
        σ_T = range(FT(0.0), FT(0.55); length = 4),
        ρ = range(FT(-0.5), FT(1); length = 4),
        μ_T = range(FT(286.5), FT(289.5); length = 4),
    )
end

"""
Multiline caption for figure headers / footers: **SI reference values** used to recover turbulent `σ` from
dimensionless `(R_q,R_T)` (or to state **`σ_q,σ_T`** grids when `show_R` is false), plus outer-range summaries—so a user can reproduce
or debug without reading source.

`dz_layer` is the **mosaic layer thickness** passed into geometry (`mathsanity_sgs_quad_moments_with_geometry`);
`geo` supplies the **gradients** (`dq_dz`, `dtheta_dz`, `dT_dθ`). Recovery:
`σ_q = Δz_layer·(∂q/∂z)/R_q`, `σ_T = Δz_layer·(∂T/∂z)/R_T` when `show_R` is true.
"""
function mathsanity_mosaic_reference_debug_caption(
    geo,
    dz_layer::FT,
    ranges,
    μ_q::Real;
    show_R::Bool,
    Rq = nothing,
    RT = nothing,
) where {FT <: Real}
    dq = FT(geo.dq_dz)
    dth = FT(geo.dtheta_dz)
    dTdth = FT(geo.dT_dθ)
    dTdz = dTdth * dth
    dz_l = isinteger(dz_layer) ? string(Int(dz_layer)) : @sprintf("%.1f", dz_layer)
    σf = mathsanity_sigma_floor_standardized(FT)
    s1 = @sprintf(
        "REFERENCE (nondim↔dim / repro): μ_q=%.6f | mosaic Δz=%s m | ∂q/∂z=%.4e | ∂θ_li/∂z=%.5f | ∂T/∂θ_li=%.4f ⇒ ∂T/∂z=%.6e",
        μ_q,
        dz_l,
        dq,
        dth,
        dTdth,
        dTdz,
    )
    rho_rng = collect(ranges.ρ)
    muT_rng = collect(ranges.μ_T)
    s2 = @sprintf(
        "Outer axes: ρ ∈ [%.4g, %.4g], μ_T ∈ [%.5g, %.5g] K",
        minimum(rho_rng),
        maximum(rho_rng),
        minimum(muT_rng),
        maximum(muT_rng),
    )
    s3 = if show_R && Rq !== nothing && RT !== nothing
        @sprintf(
            "Inner R: R_q ∈ [%.5g, %.5g], R_T ∈ [%.5g, %.5g]  →  σ_q=Δz·(∂q/∂z)/R_q, σ_T=Δz·(∂T/∂z)/R_T (Δz = mosaic value above)",
            minimum(Rq),
            maximum(Rq),
            minimum(RT),
            maximum(RT),
        )
    else
        "Inner grid: (σ_q, σ_T) taken directly from ranges (no R→σ recovery)."
    end
    s4 = @sprintf(
        "geo defaults (turbulent variances in knobs; not panel σ): default dz_knob=%.5g m, qq=%.3g, TT=%.3g, ρ_param=%.4g",
        geo.dz,
        geo.qq,
        geo.TT,
        geo.ρ_param,
    )
    if show_R && hasproperty(ranges, :dz_ref)
        s4 = s4 * @sprintf(" | ranges.dz_ref=%.5g (equals geo.dz when using default ranges)", ranges.dz_ref)
    end
    s5 = @sprintf("Standardized maps: σ_floor = %.3g.", σf)
    return join([s1, s2, s3, s4, s5], "\n")
end

"""Four 1D ranges for geometry sweep pages (dq·dθ × qq × TT; fourth axis pages)."""
function mathsanity_default_geometry_grid_ranges(FT::Type{<:Real} = Float32)
    (;
        dq_dz = range(FT(0.0), FT(2.0e-5); length = 4),
        dtheta_dz = range(FT(0.0), FT(0.03); length = 4),
        qq = range(FT(1e-6), FT(8e-6); length = 4),
        TT = range(FT(0.015), FT(0.07); length = 4),
    )
end
