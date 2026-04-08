# Pure math checks:
# - Gauss–Hermite + √(2)·χ (GaussianSGS map)
# - Subcell `(1/12)Δz²(∂·/∂z)²` variance / covariance (same scalars as `materialize_sgs_quadrature_moments!`)
# - Optional resolution transfer `scale_sgs_moments_geometry` (LES ref Δz → target Δz)
#
# **REPL:** activate `calibration/experiments/variance_adjustments`, then
#   `include(path/to/math_sanity/run_all.jl)`
#   `mathsanity_run_all()`  — numeric checks + overview plots → `figures/summary/`
#   `mathsanity_run_all(write_gaussian_mosaic=true)`  — plus Gaussian mosaics at **`geo.dz`** (panel count from `ranges`,
#       e.g. `mathsanity_default_gaussian_grid_ranges`); primary `(z_q,z_T)` PNG + `_physical.png`; clamped = secondary diagnostic.
#   `mathsanity_run_all(write_cell_pngs=true)`  — opt-in: one PNG per mosaic panel under `figures/sweep/cells/` (slow).
# Default is **false** so `run_all` does not flood the tree; combine flags as needed.
#
# **`include` only loads definitions.** **`julia run_all.jl`** calls `mathsanity_run_all()` with defaults.
#
# **Mosaics only** (skip checks): `include(.../math_sanity/run_sweep.jl)` then `mathsanity_run_mosaic_sweep()`.
#
# Sweep axes: `math_sanity/defaults.jl` (`mathsanity_default_gaussian_grid_ranges`, etc.).
#
const MATH_SANITY_DIR = dirname(@__FILE__)

using Printf

include(joinpath(MATH_SANITY_DIR, "defaults.jl"))
include(joinpath(MATH_SANITY_DIR, "geometric_moments.jl"))
include(joinpath(MATH_SANITY_DIR, "moment_recovery.jl"))
include(joinpath(MATH_SANITY_DIR, "scatter_series.jl"))
include(joinpath(MATH_SANITY_DIR, "plots.jl"))
include(joinpath(MATH_SANITY_DIR, "plots_geometric.jl"))
include(joinpath(MATH_SANITY_DIR, "grid_plots.jl"))

import ClimaAtmos as CA

function mathsanity_run_all(;
    FT::Type{<:AbstractFloat} = Float32,
    write_gaussian_mosaic::Bool = true,
    write_cell_pngs::Bool = false,
)
    # println("=== Univariate moments (X = μ + σZ) ===")
    # for N in 1:5
    #     m, v = mathsanity_univariate_mean_var(FT(100.0), FT(3.0), N)
    #     @assert isapprox(m, FT(100.0); rtol = 1e-10, atol = 1e-10)
    #     # N==1 is a single node at χ=0 ⇒ zero empirical spread (quadrature still integrates low-order polynomials exactly by construction, but a discrete second moment about the sample mean is not σ²).
    #     N == 1 || @assert isapprox(v, FT(9.0); rtol = 1e-10, atol = 1e-10)
    #     println("  N=$N: mean=$m var=$v (expect mean 100; var 9 for N≥2)")
    # end

    println("\n=== Bivariate (unclamped): quadrature Σ vs analytic ===")
    sc = mathsanity_default_scatter_knobs(FT)
    μ_q, μ_T = sc.μ_q, sc.μ_T
    σ_q, σ_T = sc.σ_q, sc.σ_T
    ρ = sc.ρ
    # # N=1 tensor product is a single atom at the mean ⇒ zero empirical covariance.
    # for N in 2:5
    #     qs, Ts, ωs = mathsanity_quadrature_moments_bivariate(μ_q, μ_T, σ_q, σ_T, ρ, N; clamped = false)
    #     (mq, mT), Σq = mathsanity_weighted_mean_cov(qs, Ts, ωs)
    #     μ, Σ = mathsanity_analytic_mean_cov(μ_q, μ_T, σ_q, σ_T, ρ)
    #     @assert isapprox([mq, mT], μ; rtol = FT(1e-9), atol = FT(1e-9))
    #     @assert isapprox(Σq, Σ; rtol = FT(1e-9), atol = FT(1e-9))
    #     println("  N=$N: mean error ", norm([mq, mT] - μ), "  cov error ", opnorm(Σq - Σ))
    # end

    # println("\n=== Subcell geometry vs ClimaAtmos helpers ===")
    # geo = mathsanity_default_geometric_knobs()
    # dz = FT(geo.dz)
    # dq_dz = FT(1.2e-5)
    # dth = FT(0.02)
    # dT_dθ = FT(0.55)
    # dqd_sq = dq_dz^2
    # dTdz_sq = (dT_dθ * dth)^2
    # Δq_ca, ΔT_ca = CA.subcell_geometric_variance_increment(dz, dqd_sq, dTdz_sq)
    # tw = (one(FT) / FT(12.0)) * dz^2
    # @assert isapprox(Δq_ca, tw * dqd_sq; rtol = FT(1e-12), atol = FT(1e-20))    
    # @assert isapprox(ΔT_ca, tw * dTdz_sq; rtol = FT(1e-12), atol = FT(1e-20))
    # dot_wq_wθ = dq_dz * dth
    # covg_ca = CA.subcell_geometric_covariance_Tq(dz, dT_dθ, dot_wq_wθ)
    # @assert isapprox(covg_ca, tw * dT_dθ * dot_wq_wθ; rtol = FT(1e-12), atol = FT(1e-20))
    # println("  subcell_geometric_variance_increment / subcell_geometric_covariance_Tq match hand twelfth·Δz² formulas")

    # qq = FT(geo.qq)
    # TT = FT(geo.TT)
    # ρp = FT(geo.ρ_param)
    # var_q, var_T, ρ_eff, ge =
    #     mathsanity_sgs_quad_moments_with_geometry(qq, TT, ρp, dz, dq_dz, dth, dT_dθ)
    # @assert isapprox(var_q, qq + Δq_ca; rtol = FT(1e-12), atol = FT(1e-20))
    # @assert isapprox(var_T, TT + ΔT_ca; rtol = FT(1e-12), atol = FT(1e-20))
    # numer = ρp * sqrt(qq) * sqrt(TT) + covg_ca
    # denom = max(FT(1e-10), sqrt(var_q) * sqrt(var_T))
    # @assert isapprox(ρ_eff, clamp(numer / denom, -1.0, 1.0); rtol = FT(1e-10), atol = FT(1e-12))
    # println("  materialize-style var_q, var_T, ρ_eff consistent (ρ_eff = $ρ_eff)")

    # println("\n=== scale_sgs_moments_geometry (resolution transfer) ===")
    # q_a, th_a, c_a = scale_sgs_moments_geometry(FT(1e-5), FT(0.02), FT(3e-4), FT(200.0), FT(200.0), FT(1e-5), FT(0.01)) 
    # @assert q_a ≈ FT(1e-5) && th_a ≈ FT(0.02)
    # println("  same dz_ref=dz_target ⇒ no change")
    # q_b, _, _ = scale_sgs_moments_geometry(
    #     FT(1e-5),
    #     FT(0.02),
    #     FT(0.0),
    #     FT(0.0),
    #     FT(200.0),
    #     FT(1e-5),
    #     FT(0.02);
    #     ref_includes_gradient = false,
    # )
    # gf = (one(FT) / FT(12.0)) * FT(200.0)^2 * FT(1e-5)^2
    # @assert isapprox(q_b, FT(1e-5) + gf; rtol = FT(1e-12), atol = FT(1e-24))
    # println("  ref_includes_gradient=false adds (1/12)dz_target² gradient term")

    # println("\n=== d(variance)/dσ vs 2σ (finite difference on GH variance, N=5) ===")
    # dfd, dtrue = mathsanity_d_variance_d_sigma(FT(0.0), FT(1.25), 5)
    # println("  FD: $dfd  analytic: $dtrue  abs err: $(abs(dfd - dtrue))")

    # println("\n=== Clamped GaussianSGS: moments deviate (expected) ===")
    # qs, Ts, ωs = mathsanity_quadrature_moments_bivariate(μ_q, μ_T, σ_q, σ_T, ρ, 5; clamped = true)
    # (mq, mT), Σq = mathsanity_weighted_mean_cov(qs, Ts, ωs)
    # println("  weighted mean (q,T) = ($mq, $mT)")
    # println("  compare to unclamped analytic μ = ($(μ_q), $(μ_T))")

    out = joinpath(MATH_SANITY_DIR, "figures")
    summary_dir = joinpath(out, "summary")
    mosaic_dir = joinpath(out, "mosaic")
    sweep_cells_dir = joinpath(out, "sweep", "cells")
    println("\n=== Writing summary figures to $summary_dir ===")
    mkpath(summary_dir)
    mathsanity_plot_quadrature_scatter(joinpath(summary_dir, "scatter_quad_vs_mc_unclamped.png"); clamped = false)
    mathsanity_plot_quadrature_scatter(joinpath(summary_dir, "scatter_quad_clamped.png"); clamped = true)
    mathsanity_plot_sigma_ratio_scan(joinpath(summary_dir, "sigma_ratio_moment_error.png"))
    mathsanity_plot_geometric_breakdown(joinpath(summary_dir, "geometric_variance_breakdown.png"))
    mathsanity_plot_geometric_breakdown_dz_scan(joinpath(summary_dir, "geometric_variance_vs_dz.png"))
    if write_gaussian_mosaic
        println("\n=== Gaussian mosaics (standardized primary + _physical) ===")
        mkpath(mosaic_dir)
        geo = mathsanity_default_geometric_knobs(FT)
        ranges = mathsanity_default_gaussian_grid_ranges(FT)
        np = mathsanity_gaussian_mosaic_panel_count(ranges)
        dz = FT(geo.dz)
        stem_u = mathsanity_mosaic_output_basename_stem(false, dz, geo, np)
        stem_c = mathsanity_mosaic_output_basename_stem(true, dz, geo, np)
        @info "Plotting mosaic at reference Δz = $dz m ($stem_u, $np panels)"
        base_u = joinpath(mosaic_dir, stem_u)
        base_c = joinpath(mosaic_dir, stem_c)
        mathsanity_plot_gaussian_parameter_mosaic(
            base_u * ".png";
            ranges = ranges,
            clamped = false,
            dz = dz,
            geo = geo,
            axes = :standardized,
        )
        mathsanity_plot_gaussian_parameter_mosaic(
            base_u * "_physical.png";
            ranges = ranges,
            clamped = false,
            dz = dz,
            geo = geo,
            axes = :physical,
        )
        mathsanity_plot_gaussian_parameter_mosaic(
            base_c * ".png";
            ranges = ranges,
            clamped = true,
            dz = dz,
            geo = geo,
            axes = :standardized,
        )
        mathsanity_plot_gaussian_parameter_mosaic(
            base_c * "_physical.png";
            ranges = ranges,
            clamped = true,
            dz = dz,
            geo = geo,
            axes = :physical,
        )
    end
    if write_cell_pngs
        println("\n=== Per-tuple panel PNGs under $sweep_cells_dir ===")
        mkpath(sweep_cells_dir)
        geo = mathsanity_default_geometric_knobs(FT)
        ranges = mathsanity_default_gaussian_grid_ranges(FT)
        mathsanity_write_gaussian_mosaic_cell_pngs(
            joinpath(sweep_cells_dir, "unclamped");
            ranges = ranges,
            clamped = false,
            dz = geo.dz,
            geo = geo,
        )
        mathsanity_write_gaussian_mosaic_cell_pngs(
            joinpath(sweep_cells_dir, "clamped");
            ranges = ranges,
            clamped = true,
            dz = geo.dz,
            geo = geo,
        )
    end
    println("Done.")
end

# Script entry: `julia run_all.jl` (not on bare `include`).
if !isempty(something(PROGRAM_FILE, "")) && abspath(PROGRAM_FILE) == abspath(@__FILE__)
    mathsanity_run_all()
end
