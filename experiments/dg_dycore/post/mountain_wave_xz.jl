#=
Agnesi mountain-wave x–z cross-sections from the DG dycore's NetCDF output
(MountainWaveDG, §10 of docs/vi_kep_face_terms.md). Stacked filled contours
of the horizontal (ua) and vertical (wa) velocity at the final snapshot,
sliced at the slab mid-plane (y = 0), over the full horizontal domain and
up to `z_top_km`, with the analytic Agnesi ridge h(x) = h₀/(1 + (x/a)²)
overlaid on each panel.

Two-stage, matching the wave_train_maps.jl convention (the sim writes
NetCDF under the run project; this reads it under the post project). There
is no YAML/driver path for MountainWaveDG, and diagnostics are written ONLY
when `output_dir` is set, so the sim is a one-liner that must set it.

STAGE 1 — write NetCDF (run project). Set `output_dir` + `diag_period`:

    julia --project=experiments/dg_dycore -e '
      include("experiments/dg_dycore/src/DGDycore.jl"); import .DGDycore as DG
      DG.run!(DG.DGSimulation(DG.MountainWaveDG(;
          helem = 20, zelem = 20, zmax = 30e3, xmax = 600e3,
          a = 25e3, U₀ = 20.0, T₀ = 250.0, h₀ = 250.0, face_set = :es,
          t_end = 3600.0, diag_period = 3600.0,
          output_dir = "output/mw_agnesi_h250")))'

The §10 figures use the fuller setup (helem 40 × zelem 40, κ₄_frac = 0.1,
ν_vert = 100, sponge_τ = 1, sponge_depth = 15e3, t_end = 4 * 3600, and
h₀ = 25.0 for the linear-verification panel).

STAGE 2 — plot (this script, post project):

    julia --project=experiments/dg_dycore/post \
        experiments/dg_dycore/post/mountain_wave_xz.jl \
        <output_dir> <out.png> [z_top_km] [h0_m] [a_m]

`z_top_km` (default 10) caps the vertical window; `h0_m`/`a_m` (default
250 / 25e3) only position the overlaid ridge. The full horizontal domain is
always shown (no x-window). Example:

    ... mountain_wave_xz.jl output/mw_agnesi_h250 \
        output/mw_agnesi_h250/uw_xz.png 10 250 25e3
=#

import ClimaAnalysis
import CairoMakie

const VELOCITY_PANELS =
    (short_name = "ua", label = "u [m/s]"), (short_name = "wa", label = "w [m/s]")

"""
    mountain_wave_xz(output_dir, out_png; z_top_km, h₀, a, panels)

Stacked `ua`/`wa` x–z contours at the final snapshot (mid-plane `y = 0`),
over the full horizontal domain and `z ≤ z_top_km`, ridge overlaid.
"""
function mountain_wave_xz(
    output_dir,
    out_png;
    z_top_km = 10.0,
    h₀ = 250.0,
    a = 25e3,
    panels = VELOCITY_PANELS,
)
    simdir = ClimaAnalysis.SimDir(output_dir)
    fig = CairoMakie.Figure(; size = (1000, 360 * length(panels)))
    for (row, p) in enumerate(panels)
        var = ClimaAnalysis.get(simdir; short_name = p.short_name)
        t_h = round(ClimaAnalysis.times(var)[end] / 3600.0; digits = 2)
        # slab is quasi-2D (one y element) — collapse y, take the last time
        haskey(var.dims, "y") && (var = ClimaAnalysis.slice(var; y = 0.0))
        var = ClimaAnalysis.slice(var; time = Inf)
        var = ClimaAnalysis.window(var, "z"; right = 1e3 * z_top_km)

        x = var.dims["x"]          # [m], full domain (no x-window)
        z = var.dims["z"]          # [m]
        d = var.data               # (nx, nz)
        amp = max(maximum(abs, d), eps())

        ax = CairoMakie.Axis(
            fig[row, 1];
            xlabel = "x [km]",
            ylabel = "z [km]",
            title = "$(p.label), h₀ = $(round(Int, h₀)) m, t = $(t_h) h " *
                    "(max|·| = $(round(amp; sigdigits = 2)))",
        )
        plt = CairoMakie.contourf!(
            ax,
            x ./ 1e3,
            z ./ 1e3,
            d;
            levels = range(-amp, amp; length = 21),
            colormap = :balance,
        )
        CairoMakie.Colorbar(fig[row, 2], plt)
        # analytic Agnesi ridge h(x) = h₀ / (1 + (x/a)²)
        CairoMakie.lines!(
            ax,
            x ./ 1e3,
            (h₀ ./ (1 .+ (x ./ a) .^ 2)) ./ 1e3;
            color = :black,
            linewidth = 1.5,
        )
    end
    CairoMakie.save(out_png, fig)
    @info "wrote $out_png"
    return out_png
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) ≥ 2 || error(
        "usage: mountain_wave_xz.jl <output_dir> <out.png> [z_top_km] [h0_m] [a_m]",
    )
    z_top_km = length(ARGS) ≥ 3 ? parse(Float64, ARGS[3]) : 10.0
    h₀ = length(ARGS) ≥ 4 ? parse(Float64, ARGS[4]) : 250.0
    a = length(ARGS) ≥ 5 ? parse(Float64, ARGS[5]) : 25e3
    mountain_wave_xz(ARGS[1], ARGS[2]; z_top_km, h₀, a)
end
