# Requires `defaults.jl`, `geometric_moments.jl`, `moment_recovery.jl`, `scatter_series.jl`, and CairoMakie (see `run_all.jl` include order).

using CairoMakie
using LinearAlgebra
using Printf
using Random

# CairoMakie 2D draw order: sort by model **z translation**, then parent scene, then insertion order
# (see https://docs.makie.org/stable/explanations/backends/cairomakie.html#Z-Order — use `translate!(plot, 0, 0, z)`).
#
# Makie's `depth_shift` attribute is documented as **GLMakie and WGLMakie only** (clip-space tweak), not Cairo —
# see e.g. the `depth_shift` section under scatter: https://docs.makie.org/stable/reference/plots/scatter.html
#
# Reserve spaced `z` values so new layers can slot in without renumbering. **Larger z draws on top** (CairoMakie).
# Column gradient is a background guide → lowest z; GH nodes on top for visibility.
const MATHSANITY_Z_LAYER_COLUMN = 0.0f0
const MATHSANITY_Z_LAYER_MC = 0.02f0
const MATHSANITY_Z_LAYER_ELLIPSE_NAIVE = 0.04f0
const MATHSANITY_Z_LAYER_ELLIPSE_GEOM = 0.06f0
const MATHSANITY_Z_LAYER_GH_QUAD = 0.08f0
const MATHSANITY_Z_LAYER_SINGULAR_MARK = 0.11f0

"""True when turbulent σ is zero / non-finite; standardized plots use `σ_floor` and need separate axis limits."""
function _mathsanity_sigma_axis_singular_turb(σ_raw::FT) where {FT <: AbstractFloat}
    return !isfinite(σ_raw) || iszero(σ_raw)
end

function _mathsanity_darken_rgbf(c::RGBf, factor::Float32)
    RGBf(Float32(c.r) * factor, Float32(c.g) * factor, Float32(c.b) * factor)
end

"""Pad and optionally clip standardized `(z_q,z_T)` limits for one pool (nonsingular vs singular axis)."""
function _mathsanity_finalize_mosaic_standard_z_limits(z_lo::FT, z_hi::FT, clamped::Bool, Lz::FT) where {FT <: AbstractFloat}
    if !(z_lo < z_hi)
        return (-Lz, Lz)
    end
    span = max(z_hi - z_lo, FT(0.35))
    pad = FT(0.1) * span
    z_lo = z_lo - pad
    z_hi = z_hi + pad
    if !clamped
        z_lo = min(z_lo, -Lz)
        z_hi = max(z_hi, Lz)
    end
    return (z_lo, z_hi)
end

"""Apply CairoMakie z-layering for `(q,T)` physical overlays (`translate!` is the supported mechanism)."""
function _mathsanity_cairo_layer!(plot, z::Float32)
    translate!(plot, 0.0f0, 0.0f0, z)
    return plot
end

"""Draw quadrature nodes + MC cloud + analytic Σ_z ellipses on one axis (standardized z-space)."""
function mathsanity_draw_mc_quad_overlay!(
    ax,
    dat;
    quad_stroke = 0.15,
    mc_ms::Float32 = 3.0f0,
    mc_color = (:gray, 0.28),
)
    scatter!(ax, dat.zq_mc, dat.zT_mc; markersize = mc_ms, color = mc_color, strokewidth = 0)
    scatter!(
        ax,
        dat.zq_quad,
        dat.zT_quad;
        markersize = dat.ms,
        color = (:dodgerblue, 0.82),
        strokewidth = quad_stroke,
        strokecolor = (:black, 0.35),
    )
    lines!(ax, dat.xe1, dat.ye1; color = (:red, 0.75), linewidth = 1.8)
    lines!(ax, dat.xe2, dat.ye2; color = (:red, 0.55), linestyle = :dash, linewidth = 1.2)
    xlims!(ax, dat.xlo, dat.xhi)
    ylims!(ax, dat.ylo, dat.yhi)
    return ax
end

"""Light **opaque** background for each `(ρ, μ_T)` outer block. Do not use a `Box` in the same layout cell as `Axis`."""
function _mathsanity_mosaic_block_axis_bg(iρ::Int, iμ::Int, n_ρ::Int, n_μ::Int)
    u = Float32((iρ - 1) / max(n_ρ - 1, 1))
    v = Float32((iμ - 1) / max(n_μ - 1, 1))
    return RGBf(
        0.94f0 + 0.04f0 * u,
        0.96f0 - 0.03f0 * v,
        0.99f0 - 0.02f0 * (u + v) / 2,
    )
end

"""
Draw column structure in `(q,T)` through `(μ_q, μ_T)` as a **finite** `lines!` segment.

Pass `column_dz = Δz` so the segment spans one vertical layer, `(μ_q,μ_T) ± (Δz/2)(dq/dz, dT/dz)`, then is clipped to the
axis box. Pass explicit `column_dz ≤ 0` (e.g. `Δz = 0`) to draw a **black dot** at `(μ_q, μ_T)` only — no extended `dT/dq` line
(there is no layer thickness to span). Use `column_dz = nothing` to draw the full column line through the view box (callers that
omit `Δz`).

Avoids `ablines!`, which often does not render in CairoMakie multi-layer figures.

Singular cases: exact `iszero(dq)` → vertical segment; exact `iszero(dT)` → horizontal.
"""
function mathsanity_draw_column_grad_makie!(
    ax,
    column_grad,
    qlo,
    qhi,
    Tlo,
    Thi;
    linewidth::Float32 = 2.4f0,
    column_dz = nothing,
)
    column_grad === nothing && return
    μq, μT, dq_dz, dθ, dTθ = column_grad
    FT = typeof(μq)
    qloF = FT(qlo)
    qhiF = FT(qhi)
    TloF = FT(Tlo)
    ThiF = FT(Thi)
    if column_dz !== nothing && FT(column_dz) <= zero(FT)
        if (qloF <= μq <= qhiF) && (TloF <= μT <= ThiF)
            pt_ms = max(5.0f0, linewidth * 2.0f0)
            _mathsanity_cairo_layer!(
                scatter!(ax, [μq], [μT]; marker = :circle, color = :black, markersize = pt_ms, strokewidth = 0),
                MATHSANITY_Z_LAYER_COLUMN,
            )
        end
        return
    end
    seg = mathsanity_column_line_segment_dz_or_box(
        qloF,
        qhiF,
        TloF,
        ThiF,
        μq,
        μT,
        dq_dz,
        dθ,
        dTθ,
        column_dz,
    )
    seg === nothing && return
    q1, T1, q2, T2 = seg
    halo = linewidth + 5.0f0
    _mathsanity_cairo_layer!(
        lines!(ax, [q1, q2], [T1, T2]; color = (:white, 0.94), linewidth = halo),
        MATHSANITY_Z_LAYER_COLUMN,
    )
    _mathsanity_cairo_layer!(
        lines!(ax, [q1, q2], [T1, T2]; color = :black, linewidth = linewidth),
        MATHSANITY_Z_LAYER_COLUMN,
    )
    return nothing
end

function mathsanity_draw_physical_mc_overlay!(
    ax,
    dat::MathsanityPhysicalMosaicPanel;
    mc_ms::Float32 = 5.0f0,
    naive_color = (:gray, 0.28),
    corr_color = (:coral, 0.45),
    marker = :circle,
    quad_stroke = 0.15f0,
    column_grad_linewidth::Float32 = 2.4f0,
    qlo = nothing,
    qhi = nothing,
    Tlo = nothing,
    Thi = nothing,
    reference_gaussian = nothing,
    draw_geometry_ellipses::Bool = true,
    show_legend::Bool = false,
    column_dz = nothing,
)
    xlo = qlo === nothing ? dat.qlo : qlo
    xhi = qhi === nothing ? dat.qhi : qhi
    ylo = Tlo === nothing ? dat.Tlo : Tlo
    yhi = Thi === nothing ? dat.Thi : Thi
    xlims!(ax, xlo, xhi)
    ylims!(ax, ylo, yhi)
    FT = typeof(dat.qlo)
    # Z-order: explicit `translate!` (CairoMakie). Column lowest; MC / Σ / GH above so the line stays readable as backdrop.
    mathsanity_draw_column_grad_makie!(
        ax,
        dat.column_grad,
        xlo,
        xhi,
        ylo,
        yhi;
        linewidth = column_grad_linewidth,
        column_dz = column_dz,
    )
    if dat.q_naive !== nothing
        _mathsanity_cairo_layer!(
            scatter!(
                ax,
                dat.q_naive,
                dat.T_naive;
                marker = marker,
                markersize = mc_ms,
                color = naive_color,
                strokewidth = 0,
            ),
            MATHSANITY_Z_LAYER_MC,
        )
        _mathsanity_cairo_layer!(
            scatter!(
                ax,
                dat.q_corr,
                dat.T_corr;
                marker = marker,
                markersize = mc_ms,
                color = corr_color,
                strokewidth = 0,
            ),
            MATHSANITY_Z_LAYER_MC,
        )
    end
    # Red = turbulent / “naive” SGS Gaussian in (q,T) (same σ_q, σ_T, ρ as the parameter grid).
    if reference_gaussian !== nothing
        μqr, μTr, σqr, σTr, ρr = reference_gaussian
        qe1, Te1, qe2, Te2 =
            mathsanity_naive_gaussian_ellipse_polylines_physical(FT(μqr), FT(μTr), FT(σqr), FT(σTr), FT(ρr))
        if !isempty(qe1)
            _mathsanity_cairo_layer!(
                lines!(ax, qe1, Te1; color = (:red, 0.82), linewidth = 2.2),
                MATHSANITY_Z_LAYER_ELLIPSE_NAIVE,
            )
            _mathsanity_cairo_layer!(
                lines!(ax, qe2, Te2; color = (:red, 0.68), linestyle = :dash, linewidth = 2.0),
                MATHSANITY_Z_LAYER_ELLIPSE_NAIVE,
            )
        end
    end
    # Teal = geometry-inclusive Gaussian matching the **coral** MC map (σ_tot, ρ_eff after (1/12)Δz² terms).
    had_teal = false
    if draw_geometry_ellipses && reference_gaussian !== nothing
        μqr, μTr, σqr, σTr, ρr = reference_gaussian
        σqg, σTg, ρg = dat.sigma_q_tot, dat.sigma_T_tot, dat.rho_eff_tot
        if mathsanity_sigma_geometry_differs_from_turb(FT(σqr), FT(σTr), FT(ρr), σqg, σTg, ρg)
            ga1, Ta1, ga2, Ta2 =
                mathsanity_naive_gaussian_ellipse_polylines_physical(FT(μqr), FT(μTr), FT(σqg), FT(σTg), FT(ρg))
            if !isempty(ga1)
                had_teal = true
                _mathsanity_cairo_layer!(
                    lines!(ax, ga1, Ta1; color = (:teal, 0.88), linewidth = 2.0),
                    MATHSANITY_Z_LAYER_ELLIPSE_GEOM,
                )
                _mathsanity_cairo_layer!(
                    lines!(ax, ga2, Ta2; color = (:teal, 0.62), linestyle = :dash, linewidth = 1.75),
                    MATHSANITY_Z_LAYER_ELLIPSE_GEOM,
                )
            end
        end
    end
    if dat.quad_q_geom !== nothing
        _mathsanity_cairo_layer!(
            scatter!(
                ax,
                dat.quad_q_geom,
                dat.quad_T_geom;
                markersize = dat.quad_ms_geom,
                marker = :circle,
                color = (:darkorchid, 0.78),
                strokewidth = 0.12f0,
                strokecolor = (:black, 0.25),
            ),
            MATHSANITY_Z_LAYER_GH_QUAD,
        )
    end
    if dat.quad_q !== nothing
        _mathsanity_cairo_layer!(
            scatter!(
                ax,
                dat.quad_q,
                dat.quad_T;
                markersize = dat.quad_ms,
                marker = :circle,
                color = (:dodgerblue, 0.82),
                strokewidth = quad_stroke,
                strokecolor = (:black, 0.35),
            ),
            MATHSANITY_Z_LAYER_GH_QUAD,
        )
    end
    if show_legend
        elems = Any[]
        labels = String[]
        if dat.column_grad !== nothing
            if column_dz !== nothing && FT(column_dz) <= zero(FT)
                push!(elems, MarkerElement(; marker = :circle, color = :black, markersize = 11, strokewidth = 0))
                push!(
                    labels,
                    "column Δz=0: " *
                    mathsanity_column_slope_phrase(dat.column_grad[3], dat.column_grad[4], dat.column_grad[5]) *
                    " → dot at (μ_q, μ_T)",
                )
            else
                push!(elems, LineElement(; color = :black, linewidth = 3))
                push!(
                    labels,
                    "column: " * mathsanity_column_slope_phrase(dat.column_grad[3], dat.column_grad[4], dat.column_grad[5]),
                )
            end
        end
        if reference_gaussian !== nothing
            push!(elems, LineElement(; color = (:red, 0.82), linewidth = 2.5))
            push!(labels, "1σ naive Σ (turb σ_q, σ_T, ρ)")
            push!(elems, LineElement(; color = (:red, 0.68), linewidth = 2, linestyle = :dash))
            push!(labels, "2σ naive Σ")
        end
        if had_teal
            push!(elems, LineElement(; color = (:teal, 0.88), linewidth = 2.2))
            push!(labels, "1σ geometry Σ (σ_tot, ρ_eff)")
            push!(elems, LineElement(; color = (:teal, 0.62), linewidth = 2, linestyle = :dash))
            push!(labels, "2σ geometry Σ")
        end
        if dat.quad_q_geom !== nothing
            push!(elems, MarkerElement(; marker = :circle, color = (:darkorchid, 0.78), markersize = 9, strokewidth = 1))
            push!(labels, "GH at σ_tot, ρ_eff (purple)")
        end
        if dat.quad_q !== nothing
            push!(elems, MarkerElement(; marker = :circle, color = (:dodgerblue, 0.82), markersize = 10, strokewidth = 1))
            push!(labels, "GH at grid σ_q, σ_T, ρ (blue)")
        end
        if dat.q_naive !== nothing
            push!(elems, MarkerElement(; marker = marker, color = naive_color, markersize = 8, strokewidth = 0))
            push!(labels, "MC same χ, turb map")
            push!(elems, MarkerElement(; marker = marker, color = corr_color, markersize = 8, strokewidth = 0))
            push!(labels, "MC same χ, geometry map (coral)")
        end
        if !isempty(elems)
            axislegend(ax, elems, labels; position = :rt, framevisible = true, labelsize = 9, patchsize = (18, 10))
        end
    end
    return ax
end

"""
Same overlays as `mathsanity_draw_physical_mc_overlay!`, but axes are **`(z_q,z_T)`** with
`z_q=(q-μ_q)/σ_q`, `z_T=(T-μ_T)/σ_T` using the **panel turbulent** `σ_q,σ_T` (same σ as the `R_q,R_T`
grid convention). Column segment is the chord from **`−½(R_q,R_T)`** to **`+½(R_q,R_T)`** in this plane
(see `math_sanity/README.md`).
"""
function mathsanity_draw_standardized_mosaic_overlay!(
    ax,
    dat::MathsanityPhysicalMosaicPanel,
    μ_q,
    μ_T,
    σ_q,
    σ_T,
    ρ_turb::FT;
    dq_dz::FT,
    dtheta_dz::FT,
    dT_dθ::FT,
    column_dz,
    zq_lo,
    zq_hi,
    zT_lo,
    zT_hi,
    mc_ms::Float32 = 5.0f0,
    naive_color = (:gray, 0.28),
    corr_color = (:coral, 0.45),
    marker = :circle,
    quad_stroke = 0.15f0,
    column_grad_linewidth::Float32 = 2.4f0,
    reference_gaussian = nothing,
    draw_geometry_ellipses::Bool = true,
    show_legend::Bool = false,
) where {FT <: AbstractFloat}
    xlims!(ax, zq_lo, zq_hi)
    ylims!(ax, zT_lo, zT_hi)
    σf = mathsanity_sigma_floor_standardized(FT)
    σqi = max(FT(σ_q), σf)
    σTj = max(FT(σ_T), σf)
    dTdz = dT_dθ * dtheta_dz

    if dat.column_grad !== nothing
        if column_dz !== nothing && FT(column_dz) <= zero(FT)
            pt_ms = max(5.0f0, column_grad_linewidth * 2.0f0)
            _mathsanity_cairo_layer!(
                scatter!(ax, [zero(FT)], [zero(FT)]; marker = :circle, color = :black, markersize = pt_ms, strokewidth = 0),
                MATHSANITY_Z_LAYER_COLUMN,
            )
        else
            Rq = FT(column_dz) * dq_dz / σqi
            RT = FT(column_dz) * dTdz / σTj
            halo = column_grad_linewidth + 5.0f0
            _mathsanity_cairo_layer!(
                lines!(
                    ax,
                    [Rq * FT(-0.5), Rq * FT(0.5)],
                    [RT * FT(-0.5), RT * FT(0.5)];
                    color = (:white, 0.94),
                    linewidth = halo,
                ),
                MATHSANITY_Z_LAYER_COLUMN,
            )
            _mathsanity_cairo_layer!(
                lines!(
                    ax,
                    [Rq * FT(-0.5), Rq * FT(0.5)],
                    [RT * FT(-0.5), RT * FT(0.5)];
                    color = :black,
                    linewidth = column_grad_linewidth,
                ),
                MATHSANITY_Z_LAYER_COLUMN,
            )
        end
    end

    if dat.q_naive !== nothing
        zq_n = (dat.q_naive .- μ_q) ./ σqi
        zT_n = (dat.T_naive .- μ_T) ./ σTj
        zq_c = (dat.q_corr .- μ_q) ./ σqi
        zT_c = (dat.T_corr .- μ_T) ./ σTj
        _mathsanity_cairo_layer!(
            scatter!(ax, zq_n, zT_n; marker = marker, markersize = mc_ms, color = naive_color, strokewidth = 0),
            MATHSANITY_Z_LAYER_MC,
        )
        _mathsanity_cairo_layer!(
            scatter!(ax, zq_c, zT_c; marker = marker, markersize = mc_ms, color = corr_color, strokewidth = 0),
            MATHSANITY_Z_LAYER_MC,
        )
    end

    μz = FT[zero(FT), zero(FT)]
    Σz = Symmetric(FT[one(FT) ρ_turb; ρ_turb one(FT)])
    xe1, ye1 = mathsanity_cov_ellipse_xy(μz, Σz, one(FT))
    xe2, ye2 = mathsanity_cov_ellipse_xy(μz, Σz, FT(2))
    if !isempty(xe1)
        _mathsanity_cairo_layer!(
            lines!(ax, xe1, ye1; color = (:red, 0.82), linewidth = 2.2),
            MATHSANITY_Z_LAYER_ELLIPSE_NAIVE,
        )
        _mathsanity_cairo_layer!(
            lines!(ax, xe2, ye2; color = (:red, 0.68), linestyle = :dash, linewidth = 2.0),
            MATHSANITY_Z_LAYER_ELLIPSE_NAIVE,
        )
    end

    had_teal = false
    σqg = dat.sigma_q_tot
    σTg = dat.sigma_T_tot
    ρg = dat.rho_eff_tot
    if draw_geometry_ellipses && reference_gaussian !== nothing
        μqr, μTr, σqr, σTr, ρr = reference_gaussian
        if mathsanity_sigma_geometry_differs_from_turb(FT(σqr), FT(σTr), FT(ρr), σqg, σTg, ρg)
            ga1, Ta1, ga2, Ta2 =
                mathsanity_naive_gaussian_ellipse_polylines_physical(FT(μqr), FT(μTr), σqg, σTg, ρg)
            if !isempty(ga1)
                had_teal = true
                zq1 = (ga1 .- μ_q) ./ σqi
                zT1 = (Ta1 .- μ_T) ./ σTj
                zq2 = (ga2 .- μ_q) ./ σqi
                zT2 = (Ta2 .- μ_T) ./ σTj
                _mathsanity_cairo_layer!(
                    lines!(ax, zq1, zT1; color = (:teal, 0.88), linewidth = 2.0),
                    MATHSANITY_Z_LAYER_ELLIPSE_GEOM,
                )
                _mathsanity_cairo_layer!(
                    lines!(ax, zq2, zT2; color = (:teal, 0.62), linestyle = :dash, linewidth = 1.75),
                    MATHSANITY_Z_LAYER_ELLIPSE_GEOM,
                )
            end
        end
    end

    if dat.quad_q_geom !== nothing
        zq_g = (dat.quad_q_geom .- μ_q) ./ σqi
        zT_g = (dat.quad_T_geom .- μ_T) ./ σTj
        _mathsanity_cairo_layer!(
            scatter!(
                ax,
                zq_g,
                zT_g;
                markersize = dat.quad_ms_geom,
                marker = :circle,
                color = (:darkorchid, 0.78),
                strokewidth = 0.12f0,
                strokecolor = (:black, 0.25),
            ),
            MATHSANITY_Z_LAYER_GH_QUAD,
        )
    end
    if dat.quad_q !== nothing
        zq_b = (dat.quad_q .- μ_q) ./ σqi
        zT_b = (dat.quad_T .- μ_T) ./ σTj
        _mathsanity_cairo_layer!(
            scatter!(
                ax,
                zq_b,
                zT_b;
                markersize = dat.quad_ms,
                marker = :circle,
                color = (:dodgerblue, 0.82),
                strokewidth = quad_stroke,
                strokecolor = (:black, 0.35),
            ),
            MATHSANITY_Z_LAYER_GH_QUAD,
        )
    end

    if show_legend
        elems = Any[]
        labels = String[]
        push!(elems, LineElement(; color = :black, linewidth = 3))
        push!(labels, "column in (z_q,z_T): ±Δz/2 along chord ±½(R_q,R_T)")
        push!(elems, LineElement(; color = (:red, 0.82), linewidth = 2.5))
        push!(labels, "1σ naive Σ_z (unit variances, ρ)")
        if had_teal
            push!(elems, LineElement(; color = (:teal, 0.88), linewidth = 2.2))
            push!(labels, "geometry Σ mapped with turb σ")
        end
        if !isempty(elems)
            axislegend(ax, elems, labels; position = :rt, framevisible = true, labelsize = 9, patchsize = (18, 10))
        end
    end
    return ax
end

"""
**Fast preview:** four outer blocks × `n_inner²` standardized scatter panels.

- If **`length(μ_T) ≥ 2`** and **`length(ρ) ≥ 2`**, blocks sit at the **corners** of the `(ρ, μ_T)` grid.
- If **`μ_T` is fixed** (`length(μ_T)==1`) and **`length(ρ) ≥ 4`**, blocks use **four ρ indices** spaced along the ρ axis (same fixed `μ_T`).

Inner axes sweep `(R_q, R_T)` when `ranges` carries `R_q`/`R_T`, or **`(σ_q, σ_T)`** when those fields are set instead. For the **full** mosaic, use
`mathsanity_plot_gaussian_parameter_mosaic` (panel count from `ranges`, e.g. default `mathsanity_default_gaussian_grid_ranges`).
"""
function mathsanity_plot_gaussian_parameter_mosaic_preview(
    path::AbstractString;
    ranges = mathsanity_default_gaussian_grid_ranges(),
    μ_q = nothing,
    geo = mathsanity_default_geometric_knobs(),
    N_quad::Int = 5,
    n_mc::Int = 2000,
    clamped::Bool = false,
    lim_z = nothing,
    rng_seed::UInt64 = UInt64(42),
    figsize = (2800, 2800),
)
    σq, σT = mathsanity_inner_sigmas_for_gaussian_grid(ranges, geo)
    ρv = collect(ranges.ρ)
    μTv = collect(ranges.μ_T)
    FT = eltype(σq)
    d = mathsanity_default_scatter_knobs(FT)
    μ_q = something(μ_q, d.μ_q)
    lim_z = something(lim_z, d.lim_z)
    n_i = length(σq)
    n_ρ = length(ρv)
    n_μ = length(μTv)
    if length(σT) != n_i
        throw(ArgumentError("σ_q and σ_T inner grids must match length (got $(n_i) vs $(length(σT)))"))
    end
    outer_mode::Symbol = if n_ρ >= 2 && n_μ >= 2
        :corners
    elseif n_μ == 1 && n_ρ >= 4
        :rho_spaced
    else
        throw(
            ArgumentError(
                "Preview needs either (n_ρ≥2 and n_μ≥2) or (n_μ==1 and n_ρ≥4); got n_ρ=$n_ρ, n_μ=$n_μ.",
            ),
        )
    end
    show_R = hasproperty(ranges, :R_q)
    Rq = show_R ? collect(ranges.R_q) : nothing
    RT = show_R ? collect(ranges.R_T) : nothing

    if outer_mode === :corners
        ρ_ix = (1, n_ρ, 1, n_ρ)
        μ_ix = (1, 1, n_μ, n_μ)
        cap_outer = "corners in (ρ, μ_T)"
    else
        ρ_ix = (
            1,
            max(2, round(Int, 1 + (n_ρ - 1) / 3)),
            max(2, round(Int, 1 + 2 * (n_ρ - 1) / 3)),
            n_ρ,
        )
        μ_ix = (1, 1, 1, 1)
        cap_outer = "four ρ samples at fixed μ_T"
    end
    block_bg = (
        RGBf(0.85, 0.96, 0.86),
        RGBf(0.99, 0.96, 0.78),
        RGBf(1.0, 0.86, 0.86),
        RGBf(0.92, 0.86, 0.98),
    )

    fig = Figure(; size = figsize, fontsize = 11, figure_padding = (24, 24, 20, 20))
    inner_lbl = show_R ? "inner grid (R_q, R_T)" : "inner grid (σ_q, σ_T)"
    Label(
        fig[1, 1],
        (clamped ? "GaussianSGS (clamped): preview — " : "GaussianSGS (unclamped): preview — ") * cap_outer * "; " * inner_lbl;
        fontsize = 16,
        tellwidth = false,
        halign = :center,
    )
    outer = GridLayout(fig[2, 1])
    rowsize!(fig.layout, 2, Auto(1))

    for b in 1:4
        bi, bj = if outer_mode === :corners
            (b <= 2 ? 1 : 2, b % 2 == 1 ? 1 : 2)
        else
            (1, b)
        end
        ρ_i = ρ_ix[b]
        μ_i = μ_ix[b]
        ρc = ρv[ρ_i]
        μTc = μTv[μ_i]
        gl = GridLayout(outer[bi, bj])
        Label(
            gl[1, 1:n_i],
            @sprintf("ρ = %.3f   μ_T = %.2f K   (μ_q = %.5f)", ρc, μTc, μ_q);
            fontsize = 12,
            color = (:black, 0.75),
            tellwidth = false,
            halign = :center,
            valign = :bottom,
            padding = (0, 0, 4, 0),
        )
        bg = block_bg[b]
        for i in 1:n_i, j in 1:n_i
            ax = Axis(
                gl[i + 1, j];
                aspect = DataAspect(),
                backgroundcolor = bg,
                xlabel = i == n_i ? "z_q" : "",
                ylabel = j == 1 ? "z_T" : "",
                xlabelsize = 8,
                ylabelsize = 8,
                titlesize = 8,
                title =
                    show_R ? @sprintf("R_q=%.2f R_T=%.1f", Rq[i], RT[j]) :
                    @sprintf("σ_q=%.2e σ_T=%.3f", σq[i], σT[j]),
                titlegap = 2,
            )
            rng = Random.Xoshiro(
                rng_seed ⊻ (UInt64(i) << 8) ⊻ (UInt64(j) << 16) ⊻ (UInt64(ρ_i) << 24) ⊻ (UInt64(μ_i) << 32) ⊻
                (UInt64(b) << 40),
            )
            dat = mathsanity_standardized_mc_quad_data(
                FT(μ_q),
                FT(μTc),
                FT(σq[i]),
                FT(σT[j]),
                FT(ρc);
                N_quad = N_quad,
                n_mc = n_mc,
                clamped = clamped,
                rng = rng,
                lim_z = FT(lim_z),
            )
            mathsanity_draw_mc_quad_overlay!(ax, dat)
        end
        colgap!(gl, 4)
        rowgap!(gl, 4)
    end
    colgap!(outer, 12)
    rowgap!(outer, 12)

    mkpath(dirname(path))
    save(path, fig; px_per_unit = 2)
    return fig
end

"""Build one physical `(q,T)` mosaic panel’s data: turbulent `σ` from the inner grid, geometry from `dz` and `geo` gradients."""
function _mathsanity_mosaic_panel_pdat(
    ::Type{FT},
    iρ::Int,
    iμ::Int,
    i::Int,
    j::Int,
    rng_seed::UInt64,
    μ_q,
    σq,
    σT,
    ρv,
    μTv,
    dz,
    dq_dz,
    dtheta_dz,
    dT_dθ,
    n_mc::Int,
    N_quad::Int,
    clamped::Bool,
) where {FT <: AbstractFloat}
    rng = Random.Xoshiro(
        rng_seed ⊻ (UInt64(i) << 8) ⊻ (UInt64(j) << 16) ⊻ (UInt64(iρ) << 24) ⊻ (UInt64(iμ) << 32) ⊻
        (UInt64(1) << 40),
    )
    ρc = ρv[iρ]
    μTc = μTv[iμ]
    qq_t = max(zero(FT), FT(σq[i])^2)
    TT_t = max(zero(FT), FT(σT[j])^2)
    var_q, var_T, ρ_eff, _ = mathsanity_sgs_quad_moments_with_geometry(
        qq_t,
        TT_t,
        FT(ρc),
        dz,
        dq_dz,
        dtheta_dz,
        dT_dθ,
    )
    return mathsanity_physical_mosaic_panel_data(
        FT(μ_q),
        FT(μTc),
        FT(σq[i]),
        FT(σT[j]),
        FT(ρc),
        var_q,
        var_T,
        ρ_eff,
        dq_dz,
        dtheta_dz,
        dT_dθ;
        n_mc = n_mc,
        N_quad = N_quad,
        clamped = clamped,
        rng = rng,
    )
end

"""Linear index for panel `(iρ, iμ, i, j)` stored in `ρ`-major, `μ_T`, then inner `(R_q,R_T)` order."""
function _mathsanity_mosaic_panel_linear_index(iρ::Int, iμ::Int, i::Int, j::Int, n_μ::Int, n_i::Int)
    return (iρ - 1) * n_μ * n_i * n_i + (iμ - 1) * n_i * n_i + (i - 1) * n_i + j
end

"""
**`n_ρ × n_μ × n_i × n_i`** panels (e.g. `4×1×4×4` for default `mathsanity_default_gaussian_grid_ranges`).

- **`axes = :standardized` (default):** each inner axis is **`(z_q,z_T)`** with the same turbulent `σ_q,σ_T` used for
  `R_q,R_T` on the dimensionless grid; black column = chord **`±½(R_q,R_T)`** (see `math_sanity/README.md`).
- **`axes = :physical`:** **`(q,T)`** as before.

Inner axes sweep **`(R_q, R_T)`** by default (`mathsanity_default_gaussian_grid_ranges`); use
`mathsanity_gaussian_grid_ranges_sigma_axes()` for a **`(σ_q,σ_T)`** inner tensor (including zeros).

**Clamping:** `clamped=true` is a **secondary** diagnostic (piecewise maps); core geometry is clearest in the unclamped case.

With `n_mc=0` (default) there is no MC scatter. `N_quad > 0` draws Gauss–Hermite nodes (blue = turb, purple = σ_tot when it differs).

`path` should end in `.png`; `save_pdf=true` also writes `.pdf` (slow).
"""
function mathsanity_plot_gaussian_parameter_mosaic(
    path::AbstractString;
    ranges = mathsanity_default_gaussian_grid_ranges(),
    μ_q = nothing,
    geo = mathsanity_default_geometric_knobs(),
    dz = nothing,
    axes::Symbol = :standardized,
    n_mc::Int = 0,
    N_quad = nothing,
    clamped::Bool = false,
    rng_seed::UInt64 = UInt64(42),
    figsize = nothing,
    layout_units_per_inner_axis::Float64 = 280.0,
    save_pdf::Bool = false,
    px_per_unit::Int = 1,
    mc_ms::Float32 = 5.0f0,
    mc_marker = :pixel,
    column_grad_linewidth::Float32 = 4.5f0,
)
    axes === :standardized || axes === :physical ||
        throw(ArgumentError("axes must be :standardized or :physical (got $(repr(axes)))"))
    lu = Float64(layout_units_per_inner_axis)
    dz0 = something(dz, geo.dz)
    FT = typeof(dz0)
    σq, σT = mathsanity_inner_sigmas_for_gaussian_grid(ranges, geo; dz_use = FT(dz0))
    d = mathsanity_default_scatter_knobs(FT)
    μ_q = something(μ_q, d.μ_q)
    N_quad = something(N_quad, d.N_quad)
    dz = FT(dz0)
    dq_dz = FT(geo.dq_dz)
    dtheta_dz = FT(geo.dtheta_dz)
    dT_dθ = FT(geo.dT_dθ)
    dT_dz_col = dT_dθ * dtheta_dz

    ρv = collect(ranges.ρ)
    μTv = collect(ranges.μ_T)
    n_i = length(σq)
    n_ρ = length(ρv)
    n_μ = length(μTv)
    if length(σT) != n_i
        throw(ArgumentError("σ_q and σ_T inner grids must match length (got $(n_i) vs $(length(σT)))"))
    end
    show_R = hasproperty(ranges, :R_q)
    Rq = show_R ? collect(ranges.R_q) : nothing
    RT = show_R ? collect(ranges.R_T) : nothing

    dz_lbl = isinteger(dz) ? string(Int(dz)) : @sprintf("%.1f", dz)
    col_title = iszero(dz) ? "column dot @ origin (Δz=0)" : "black column / chord"
    title_mid =
        if n_mc > 0 && N_quad > 0
            " MC + GH nodes + naive Σ ellipses + column"
        elseif n_mc > 0
            " naive vs geometry-corrected MC + column"
        elseif N_quad > 0
            " GH blue=turb purple=σ_tot + red/teal Σ + $(col_title) (no MC)"
        else
            " $(col_title) + red/teal Σ (no MC, no GH)"
        end
    plane = axes === :standardized ? "(z_q,z_T)" : "(q,T)"
    ref_short = @sprintf(
        " | ref: ∂q/∂z=%.1e ∂T/∂z=%.2e (see block below)",
        dq_dz,
        dT_dz_col,
    )
    np = n_ρ * n_μ * n_i * n_i
    title_top =
        (clamped ? "GaussianSGS (clamped): $(np) panels — " : "GaussianSGS (unclamped): $(np) panels — ") *
        plane * title_mid * "  |  mosaic Δz = $(dz_lbl) m" * ref_short

    slope_mosaic = mathsanity_column_slope_phrase(dq_dz, dtheta_dz, dT_dθ)
    black_col_txt =
        axes === :standardized ?
        (
            iszero(dz) ?
            "Δz=0: black dot at (0,0) in (z_q,z_T)." :
            "Black = chord from −½(R_q,R_T) to +½(R_q,R_T); same σ_q,σ_T as axis normalization."
        ) :
        (
            iszero(dz) ?
            "Δz=0: black dot = (μ_q, μ_T) only (no segment along z; caption still reports dT/dq from SI gradients)." :
            "Black = column segment over ±Δz/2 along z in (q,T)."
        )
    mosaic_sub =
        axes === :standardized ? @sprintf(
            "Standardized axes: z_q=(q-μ_q)/σ_q, z_T=(T-μ_T)/σ_T with **panel turb σ** (matches R_q,R_T). Column chord ±½(R_q,R_T). SI gradients: dq/dz=%.2e  dθ_li/dz=%.4f  ∂T/∂θ_li=%.4f  ⇒ dT/dz=%.5f  |  %s\n%s Teal = geometry Σ mapped with turb σ. Red = unit Σ_z (ρ). Blue/purple GH.",
            dq_dz,
            dtheta_dz,
            dT_dθ,
            dT_dz_col,
            slope_mosaic,
            black_col_txt,
        ) : @sprintf(
            "Column (SI): dq/dz=%.2e  dθ_li/dz=%.4f  ∂T/∂θ_li=%.4f  ⇒ dT/dz=%.5f  |  %s\nθ_li only builds dT/dz = (∂T/∂θ_li)(∂θ_li/∂z); axes are (q,T). %s Teal solid/dashed = σ_tot, ρ_eff Gaussian (not the column). Red = turb Σ. Blue/purple GH = turb vs σ_tot nodes.",
            dq_dz,
            dtheta_dz,
            dT_dθ,
            dT_dz_col,
            slope_mosaic,
            black_col_txt,
        )
    singular_z_caption =
        " Singular turb σ (R→∞): red ! (one axis) or !! (both); darker panel. Nonsingular axes share one z-scale; singular axes share a separate pool (σ_floor normalization)."
    mosaic_sub = mosaic_sub * (axes === :standardized ? singular_z_caption : "")

    pdats = Vector{MathsanityPhysicalMosaicPanel{FT}}(undef, np)
    for iρ in 1:n_ρ, iμ in 1:n_μ, i in 1:n_i, j in 1:n_i
        idx = _mathsanity_mosaic_panel_linear_index(iρ, iμ, i, j, n_μ, n_i)
        pdats[idx] = _mathsanity_mosaic_panel_pdat(
            FT,
            iρ,
            iμ,
            i,
            j,
            rng_seed,
            μ_q,
            σq,
            σT,
            ρv,
            μTv,
            dz,
            dq_dz,
            dtheta_dz,
            dT_dθ,
            n_mc,
            N_quad,
            clamped,
        )
    end

    qlo_g = typemax(FT)
    qhi_g = typemin(FT)
    Tlo_g = typemax(FT)
    Thi_g = typemin(FT)
    for k in 1:np
        p = pdats[k]
        qlo_g = min(qlo_g, p.qlo)
        qhi_g = max(qhi_g, p.qhi)
        Tlo_g = min(Tlo_g, p.Tlo)
        Thi_g = max(Thi_g, p.Thi)
    end
    ϵq = max(sqrt(eps(FT)), FT(1e-12))
    ϵT = max(sqrt(eps(FT)), FT(1e-6))
    if !(qlo_g < qhi_g)
        μq = FT(μ_q)
        qlo_g = μq - ϵq
        qhi_g = μq + ϵq
    end
    if !(Tlo_g < Thi_g)
        μT = FT(sum(μTv) / length(μTv))
        Tlo_g = μT - ϵT
        Thi_g = μT + ϵT
    end

    σf = mathsanity_sigma_floor_standardized(FT)
    Lz = FT(d.lim_z)
    ns_zq_lo = typemax(FT)
    ns_zq_hi = typemin(FT)
    sq_zq_lo = typemax(FT)
    sq_zq_hi = typemin(FT)
    ns_zT_lo = typemax(FT)
    ns_zT_hi = typemin(FT)
    st_zT_lo = typemax(FT)
    st_zT_hi = typemin(FT)
    if axes === :standardized
        μz = FT[zero(FT), zero(FT)]
        for iρ in 1:n_ρ, iμ in 1:n_μ, i in 1:n_i, j in 1:n_i
            idxz = _mathsanity_mosaic_panel_linear_index(iρ, iμ, i, j, n_μ, n_i)
            p = pdats[idxz]
            μTc = μTv[iμ]
            ρc = FT(ρv[iρ])
            σqi = max(FT(σq[i]), σf)
            σTj = max(FT(σT[j]), σf)
            sq_axis = _mathsanity_sigma_axis_singular_turb(FT(σq[i]))
            st_axis = _mathsanity_sigma_axis_singular_turb(FT(σT[j]))
            if p.quad_q !== nothing
                for k in eachindex(p.quad_q)
                    zq = (p.quad_q[k] - μ_q) / σqi
                    zT = (p.quad_T[k] - μTc) / σTj
                    if !sq_axis
                        ns_zq_lo = min(ns_zq_lo, zq)
                        ns_zq_hi = max(ns_zq_hi, zq)
                    else
                        sq_zq_lo = min(sq_zq_lo, zq)
                        sq_zq_hi = max(sq_zq_hi, zq)
                    end
                    if !st_axis
                        ns_zT_lo = min(ns_zT_lo, zT)
                        ns_zT_hi = max(ns_zT_hi, zT)
                    else
                        st_zT_lo = min(st_zT_lo, zT)
                        st_zT_hi = max(st_zT_hi, zT)
                    end
                end
            end
            if p.quad_q_geom !== nothing
                for k in eachindex(p.quad_q_geom)
                    zq = (p.quad_q_geom[k] - μ_q) / σqi
                    zT = (p.quad_T_geom[k] - μTc) / σTj
                    if !sq_axis
                        ns_zq_lo = min(ns_zq_lo, zq)
                        ns_zq_hi = max(ns_zq_hi, zq)
                    else
                        sq_zq_lo = min(sq_zq_lo, zq)
                        sq_zq_hi = max(sq_zq_hi, zq)
                    end
                    if !st_axis
                        ns_zT_lo = min(ns_zT_lo, zT)
                        ns_zT_hi = max(ns_zT_hi, zT)
                    else
                        st_zT_lo = min(st_zT_lo, zT)
                        st_zT_hi = max(st_zT_hi, zT)
                    end
                end
            end
            Σz = Symmetric(FT[one(FT) ρc; ρc one(FT)])
            xe1, ye1 = mathsanity_cov_ellipse_xy(μz, Σz, one(FT))
            xe2, ye2 = mathsanity_cov_ellipse_xy(μz, Σz, FT(2))
            for zq in (minimum(xe1), minimum(xe2), maximum(xe1), maximum(xe2))
                if !sq_axis
                    ns_zq_lo = min(ns_zq_lo, zq)
                    ns_zq_hi = max(ns_zq_hi, zq)
                else
                    sq_zq_lo = min(sq_zq_lo, zq)
                    sq_zq_hi = max(sq_zq_hi, zq)
                end
            end
            for zT in (minimum(ye1), minimum(ye2), maximum(ye1), maximum(ye2))
                if !st_axis
                    ns_zT_lo = min(ns_zT_lo, zT)
                    ns_zT_hi = max(ns_zT_hi, zT)
                else
                    st_zT_lo = min(st_zT_lo, zT)
                    st_zT_hi = max(st_zT_hi, zT)
                end
            end
            σqg, σTg, ρg = p.sigma_q_tot, p.sigma_T_tot, p.rho_eff_tot
            if mathsanity_sigma_geometry_differs_from_turb(FT(σq[i]), FT(σT[j]), ρc, σqg, σTg, ρg)
                ga1, Ta1, ga2, Ta2 =
                    mathsanity_naive_gaussian_ellipse_polylines_physical(FT(μ_q), μTc, σqg, σTg, ρg)
                if !isempty(ga1)
                    zq1 = (ga1 .- μ_q) ./ σqi
                    zT1 = (Ta1 .- μTc) ./ σTj
                    zq2 = (ga2 .- μ_q) ./ σqi
                    zT2 = (Ta2 .- μTc) ./ σTj
                    for zq in (minimum(zq1), minimum(zq2), maximum(zq1), maximum(zq2))
                        if !sq_axis
                            ns_zq_lo = min(ns_zq_lo, zq)
                            ns_zq_hi = max(ns_zq_hi, zq)
                        else
                            sq_zq_lo = min(sq_zq_lo, zq)
                            sq_zq_hi = max(sq_zq_hi, zq)
                        end
                    end
                    for zT in (minimum(zT1), minimum(zT2), maximum(zT1), maximum(zT2))
                        if !st_axis
                            ns_zT_lo = min(ns_zT_lo, zT)
                            ns_zT_hi = max(ns_zT_hi, zT)
                        else
                            st_zT_lo = min(st_zT_lo, zT)
                            st_zT_hi = max(st_zT_hi, zT)
                        end
                    end
                end
            end
            if !iszero(dz)
                Rq_c = dz * dq_dz
                RT_c = dz * dT_dz_col
                for zq in (Rq_c * FT(-0.5) / σqi, Rq_c * FT(0.5) / σqi)
                    if !sq_axis
                        ns_zq_lo = min(ns_zq_lo, zq)
                        ns_zq_hi = max(ns_zq_hi, zq)
                    else
                        sq_zq_lo = min(sq_zq_lo, zq)
                        sq_zq_hi = max(sq_zq_hi, zq)
                    end
                end
                for zT in (RT_c * FT(-0.5) / σTj, RT_c * FT(0.5) / σTj)
                    if !st_axis
                        ns_zT_lo = min(ns_zT_lo, zT)
                        ns_zT_hi = max(ns_zT_hi, zT)
                    else
                        st_zT_lo = min(st_zT_lo, zT)
                        st_zT_hi = max(st_zT_hi, zT)
                    end
                end
            end
            if p.q_naive !== nothing
                for k in eachindex(p.q_naive)
                    zq = (p.q_naive[k] - μ_q) / σqi
                    zT = (p.T_naive[k] - μTc) / σTj
                    if !sq_axis
                        ns_zq_lo = min(ns_zq_lo, zq)
                        ns_zq_hi = max(ns_zq_hi, zq)
                    else
                        sq_zq_lo = min(sq_zq_lo, zq)
                        sq_zq_hi = max(sq_zq_hi, zq)
                    end
                    if !st_axis
                        ns_zT_lo = min(ns_zT_lo, zT)
                        ns_zT_hi = max(ns_zT_hi, zT)
                    else
                        st_zT_lo = min(st_zT_lo, zT)
                        st_zT_hi = max(st_zT_hi, zT)
                    end
                end
                for k in eachindex(p.q_corr)
                    zq = (p.q_corr[k] - μ_q) / σqi
                    zT = (p.T_corr[k] - μTc) / σTj
                    if !sq_axis
                        ns_zq_lo = min(ns_zq_lo, zq)
                        ns_zq_hi = max(ns_zq_hi, zq)
                    else
                        sq_zq_lo = min(sq_zq_lo, zq)
                        sq_zq_hi = max(sq_zq_hi, zq)
                    end
                    if !st_axis
                        ns_zT_lo = min(ns_zT_lo, zT)
                        ns_zT_hi = max(ns_zT_hi, zT)
                    else
                        st_zT_lo = min(st_zT_lo, zT)
                        st_zT_hi = max(st_zT_hi, zT)
                    end
                end
            end
        end
        ns_zq_lo, ns_zq_hi = _mathsanity_finalize_mosaic_standard_z_limits(ns_zq_lo, ns_zq_hi, clamped, Lz)
        sq_zq_lo, sq_zq_hi = _mathsanity_finalize_mosaic_standard_z_limits(sq_zq_lo, sq_zq_hi, clamped, Lz)
        ns_zT_lo, ns_zT_hi = _mathsanity_finalize_mosaic_standard_z_limits(ns_zT_lo, ns_zT_hi, clamped, Lz)
        st_zT_lo, st_zT_hi = _mathsanity_finalize_mosaic_standard_z_limits(st_zT_lo, st_zT_hi, clamped, Lz)
    end

    slots_w = n_ρ * n_i
    slots_h = n_μ * n_i
    margin_w = 520
    margin_h = 880
    fig_w, fig_h = if figsize === nothing
        w = round(Int, lu * slots_w + margin_w)
        h = round(Int, lu * slots_h + margin_h)
        (w, h)
    else
        (figsize[1], figsize[2])
    end

    fig = Figure(;
        size = (fig_w, fig_h),
        fontsize = 16,
        figure_padding = (32, 32, 28, 28),
    )
    Label(
        fig[1, 1],
        title_top;
        fontsize = 26,
        tellwidth = false,
        halign = :center,
    )
    Label(
        fig[2, 1],
        mosaic_sub;
        fontsize = 14,
        tellwidth = false,
        halign = :center,
        color = (:black, 0.78),
    )
    ref_caption = mathsanity_mosaic_reference_debug_caption(
        geo,
        dz,
        ranges,
        μ_q;
        show_R = show_R,
        Rq = Rq,
        RT = RT,
    )
    Label(
        fig[3, 1],
        ref_caption;
        fontsize = 11,
        tellwidth = false,
        halign = :left,
        justification = :left,
        color = (:black, 0.82),
    )
    outer = GridLayout(fig[4, 1])
    rowsize!(fig.layout, 4, Auto(1))

    for iμ in 1:n_μ, iρ in 1:n_ρ
        ρc = ρv[iρ]
        μTc = μTv[iμ]
        sub = GridLayout(outer[iμ, iρ])
        block_bg = _mathsanity_mosaic_block_axis_bg(iρ, iμ, n_ρ, n_μ)
        block_sub = show_R ? "  |  inner (R_q,R_T); σ from Δz·grad/R" : "  |  inner (σ_q,σ_T)"
        Label(
            sub[1, 1:n_i],
            @sprintf("ρ = %.3f   μ_T = %.2f K   (μ_q = %.5f)%s", ρc, μTc, μ_q, block_sub);
            fontsize = 17,
            color = (:black, 0.78),
            tellwidth = false,
            halign = :center,
            valign = :bottom,
            padding = (0, 0, 5, 0),
        )
        for i in 1:n_i, j in 1:n_i
            idx = _mathsanity_mosaic_panel_linear_index(iρ, iμ, i, j, n_μ, n_i)
            x_lbl = axes === :standardized ? (i == n_i ? "z_q" : "") : (i == n_i ? "q" : "")
            y_lbl = axes === :standardized ? (j == 1 ? "z_T" : "") : (j == 1 ? "T" : "")
            ti =
                show_R ? @sprintf("Rq=%.2f RT=%.1f", Rq[i], RT[j]) :
                @sprintf("σq=%.1e σT=%.2f", σq[i], σT[j])
            sq_ax = _mathsanity_sigma_axis_singular_turb(FT(σq[i]))
            st_ax = _mathsanity_sigma_axis_singular_turb(FT(σT[j]))
            n_sing_axes = (sq_ax ? 1 : 0) + (st_ax ? 1 : 0)
            panel_bg =
                axes === :standardized ? (n_sing_axes == 0 ? block_bg :
                n_sing_axes >= 2 ? _mathsanity_darken_rgbf(block_bg, 0.72f0) :
                _mathsanity_darken_rgbf(block_bg, 0.86f0)) : block_bg
            ax = Axis(
                sub[i + 1, j];
                backgroundcolor = panel_bg,
                xlabel = x_lbl,
                ylabel = y_lbl,
                xlabelsize = 12,
                ylabelsize = 12,
                xticklabelsize = 9,
                yticklabelsize = 9,
                titlesize = 11,
                title = ti,
                titlegap = 2,
            )
            i < n_i && hidexdecorations!(ax; grid = false, minorgrid = false)
            j > 1 && hideydecorations!(ax; grid = false, minorgrid = false)
            pdat = pdats[idx]
            if axes === :physical
                mathsanity_draw_physical_mc_overlay!(
                    ax,
                    pdat;
                    mc_ms = mc_ms,
                    marker = mc_marker,
                    column_grad_linewidth = column_grad_linewidth,
                    qlo = qlo_g,
                    qhi = qhi_g,
                    Tlo = Tlo_g,
                    Thi = Thi_g,
                    reference_gaussian = (FT(μ_q), FT(μTc), FT(σq[i]), FT(σT[j]), FT(ρc)),
                    column_dz = dz,
                )
            else
                zq_lo_p = sq_ax ? sq_zq_lo : ns_zq_lo
                zq_hi_p = sq_ax ? sq_zq_hi : ns_zq_hi
                zT_lo_p = st_ax ? st_zT_lo : ns_zT_lo
                zT_hi_p = st_ax ? st_zT_hi : ns_zT_hi
                mathsanity_draw_standardized_mosaic_overlay!(
                    ax,
                    pdat,
                    FT(μ_q),
                    FT(μTc),
                    FT(σq[i]),
                    FT(σT[j]),
                    FT(ρc);
                    dq_dz = dq_dz,
                    dtheta_dz = dtheta_dz,
                    dT_dθ = dT_dθ,
                    column_dz = dz,
                    zq_lo = zq_lo_p,
                    zq_hi = zq_hi_p,
                    zT_lo = zT_lo_p,
                    zT_hi = zT_hi_p,
                    mc_ms = mc_ms,
                    marker = mc_marker,
                    column_grad_linewidth = column_grad_linewidth,
                    reference_gaussian = (FT(μ_q), FT(μTc), FT(σq[i]), FT(σT[j]), FT(ρc)),
                )
                if sq_ax || st_ax
                    mark = (sq_ax && st_ax) ? "!!" : "!"
                    _mathsanity_cairo_layer!(
                        text!(
                            ax,
                            0.97,
                            0.97;
                            text = mark,
                            space = :relative,
                            align = (:right, :top),
                            fontsize = 26,
                            font = :bold,
                            color = :red,
                        ),
                        MATHSANITY_Z_LAYER_SINGULAR_MARK,
                    )
                end
            end
        end
        colgap!(sub, 4)
        rowgap!(sub, 4)
    end
    colgap!(outer, 8)
    rowgap!(outer, 8)

    mkpath(dirname(path))
    save(path, fig; px_per_unit = px_per_unit)
    @info "Wrote PNG figure to $path"
    if save_pdf
        base, ext = splitext(path)
        if lowercase(ext) == ".png"
            pdf_path = base * ".pdf"
        else
            pdf_path = path * ".pdf"
        end
        save(pdf_path, fig)
        @info "Wrote PDF figure to $pdf_path"
    end
    return fig
end

"""
Write one `(q,T)` panel PNG for every mosaic panel (`n_ρ·n_μ·n_i²` files per variant; depends on `ranges`).

Each figure matches the mosaic drivers: **blue** GH = turbulent `(σ_q,σ_T,ρ)`; **purple** GH = `(σ_tot,ρ_eff)` when subcell geometry changes Σ;
**grey** / **coral** MC = same `χ`, turb vs geometry map; **black** column (under overlays, white halo); **red** / **teal** ellipses.
Legend + caption give numeric gradients. Prefer `mathsanity_run_all(write_cell_pngs=true)` from `run_all.jl`.
"""
function mathsanity_write_gaussian_mosaic_cell_pngs(
    out_dir::AbstractString;
    ranges = mathsanity_default_gaussian_grid_ranges(),
    μ_q = nothing,
    geo = mathsanity_default_geometric_knobs(),
    dz = nothing,
    n_mc::Int = 4000,
    N_quad = nothing,
    clamped::Bool = false,
    rng_seed::UInt64 = UInt64(42),
)
    dz0 = something(dz, geo.dz)
    FT = typeof(dz0)
    σq, σT = mathsanity_inner_sigmas_for_gaussian_grid(ranges, geo; dz_use = FT(dz0))
    d = mathsanity_default_scatter_knobs(FT)
    N_quad = something(N_quad, d.N_quad)
    μ_q = FT(something(μ_q, d.μ_q))
    dz = FT(dz0)
    dq_dz = FT(geo.dq_dz)
    dtheta_dz = FT(geo.dtheta_dz)
    dT_dθ = FT(geo.dT_dθ)
    ρv = collect(ranges.ρ)
    μTv = collect(ranges.μ_T)
    n_i = length(σq)
    n_ρ = length(ρv)
    n_μ = length(μTv)
    if length(σT) != n_i
        throw(ArgumentError("σ_q and σ_T inner grids must match length (got $(n_i) vs $(length(σT)))"))
    end
    show_R = hasproperty(ranges, :R_q)
    Rq = show_R ? collect(ranges.R_q) : nothing
    RT = show_R ? collect(ranges.R_T) : nothing
    mkpath(out_dir)
    clamp_suf = clamped ? "clamped" : "unclamped"
    dz_lbl = isinteger(dz) ? string(Int(dz)) : @sprintf("%.1f", dz)
    for iρ in 1:n_ρ, iμ in 1:n_μ, i in 1:n_i, j in 1:n_i
        ρc = FT(ρv[iρ])
        μTc = FT(μTv[iμ])
        fname =
            show_R ? @sprintf(
                "cell_%s_dz%sm_Rq_%.3f_RT_%.3f_sigq_%.4e_sigT_%.4f_rho_%.3f_muT_%.2f.png",
                clamp_suf,
                dz_lbl,
                Rq[i],
                RT[j],
                σq[i],
                σT[j],
                ρc,
                μTc,
            ) : @sprintf(
                "cell_%s_dz%sm_sigq_%.4e_sigT_%.4f_rho_%.3f_muT_%.2f.png",
                clamp_suf,
                dz_lbl,
                σq[i],
                σT[j],
                ρc,
                μTc,
            )
        path = joinpath(out_dir, fname)
        fig = Figure(; size = (720, 980), fontsize = 13, figure_padding = (18, 18, 14, 14))
        ax = Axis(
            fig[1, 1];
            backgroundcolor = :white,
            xlabel = "q (kg/kg)",
            ylabel = "T (K)",
            title = @sprintf("ρ=%.3f μ_T=%.2f | σ_q=%.2e σ_T=%.3f | Δz=%s m", ρc, μTc, σq[i], σT[j], dz_lbl),
        )
        rng = Random.Xoshiro(
            rng_seed ⊻ (UInt64(i) << 8) ⊻ (UInt64(j) << 16) ⊻ (UInt64(iρ) << 24) ⊻ (UInt64(iμ) << 32),
        )
        qq_t = max(zero(FT), FT(σq[i])^2)
        TT_t = max(zero(FT), FT(σT[j])^2)
        var_q, var_T, ρ_eff, _ =
            mathsanity_sgs_quad_moments_with_geometry(qq_t, TT_t, ρc, dz, dq_dz, dtheta_dz, dT_dθ)
        pdat = mathsanity_physical_mosaic_panel_data(
            μ_q,
            μTc,
            FT(σq[i]),
            FT(σT[j]),
            ρc,
            var_q,
            var_T,
            ρ_eff,
            dq_dz,
            dtheta_dz,
            dT_dθ;
            n_mc = n_mc,
            N_quad = N_quad,
            clamped = clamped,
            rng = rng,
        )
        mathsanity_draw_physical_mc_overlay!(
            ax,
            pdat;
            reference_gaussian = (μ_q, μTc, FT(σq[i]), FT(σT[j]), ρc),
            show_legend = true,
            column_grad_linewidth = 3.2f0,
            column_dz = dz,
        )
        dT_dz_col = dT_dθ * dtheta_dz
        slope_part = mathsanity_column_slope_phrase(dq_dz, dtheta_dz, dT_dθ)
        cap_a = @sprintf(
            "Column: dq/dz=%.3e  dθ_li/dz=%.4f  ∂T/∂θ_li=%.4f  ⇒ dT/dz=%.5f  |  %s",
            dq_dz,
            dtheta_dz,
            dT_dθ,
            dT_dz_col,
            slope_part,
        )
        cap_b = if isapprox(FT(σq[i]), zero(FT); atol = FT(1e-12)) && isapprox(FT(σT[j]), zero(FT); atol = FT(1e-8))
            "σ_q=σ_T=0 on grid: red naive ellipse omitted; MC/GH sit at μ (plus clamps). Teal shows geometry-only σ_tot if Δz>0."
        elseif clamped
            "Clamped: ellipses are Gaussian references only; MC uses physical clamps so points can lie outside red/teal."
        else
            "Unclamped: grey should match red (turb); coral should match teal (σ_tot, ρ_eff)."
        end
        ref_cell = mathsanity_mosaic_reference_debug_caption(
            geo,
            dz,
            ranges,
            μ_q;
            show_R = show_R,
            Rq = Rq,
            RT = RT,
        )
        Label(
            fig[2, 1],
            cap_a * "\n" * cap_b * "\n\n" * ref_cell;
            fontsize = 8,
            color = (:black, 0.72),
            tellwidth = false,
            halign = :left,
            justification = :left,
        )
        save(path, fig; px_per_unit = 2)
    end
    return nothing
end
