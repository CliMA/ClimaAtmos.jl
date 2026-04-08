# Optional driver: Gaussian parameter mosaics without re-running numeric self-checks.
#
# **REPL:** activate `calibration/experiments/variance_adjustments`, then
#   `include(path/to/math_sanity/run_sweep.jl)`  `mathsanity_run_mosaic_sweep()` — PNG under `figures/mosaic/` (PDF off by default; pass kwargs to `mathsanity_plot_gaussian_parameter_mosaic` if needed).
#   (`julia run_sweep.jl` calls that automatically.)
#
# Customize ranges in `defaults.jl` (`mathsanity_default_gaussian_grid_ranges`) or call
# `mathsanity_plot_gaussian_parameter_mosaic(...)` with kwargs from the REPL.

const MATH_SANITY_DIR = dirname(@__FILE__)

using Printf

include(joinpath(MATH_SANITY_DIR, "defaults.jl"))
include(joinpath(MATH_SANITY_DIR, "geometric_moments.jl"))
include(joinpath(MATH_SANITY_DIR, "moment_recovery.jl"))
include(joinpath(MATH_SANITY_DIR, "scatter_series.jl"))
include(joinpath(MATH_SANITY_DIR, "plots.jl"))
include(joinpath(MATH_SANITY_DIR, "grid_plots.jl"))

function mathsanity_run_mosaic_sweep()
    out = joinpath(MATH_SANITY_DIR, "figures")
    mosaic_dir = joinpath(out, "mosaic")
    mkpath(mosaic_dir)
    FT = typeof(mathsanity_default_geometric_knobs().dz)
    geo = mathsanity_default_geometric_knobs(FT)
    ranges = mathsanity_default_gaussian_grid_ranges(FT)
    np = mathsanity_gaussian_mosaic_panel_count(ranges)
    dz = FT(geo.dz)
    stem_u = mathsanity_mosaic_output_basename_stem(false, dz, geo, np)
    stem_c = mathsanity_mosaic_output_basename_stem(true, dz, geo, np)
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
    println("Wrote $np-panel mosaic PNGs under $mosaic_dir (standardized + _physical; use save_pdf=true for PDFs)")
end

if !isempty(something(PROGRAM_FILE, "")) && abspath(PROGRAM_FILE) == abspath(@__FILE__)
    mathsanity_run_mosaic_sweep()
end
