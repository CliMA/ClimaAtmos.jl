#=
Canonical Held–Suarez diagnostics from the DG dycore's NetCDF output:
time & zonal mean u(φ, z) and T(φ, z) over the post-spinup window.
Replaces the ClimaCore examples' hand-rolled WJ-weighted latitude binning
with ClimaAnalysis (remap-then-average; differences vs binning are ~1 m/s
/ ~1 K and documented).

Usage:
    julia --project=experiments/dg_dycore/post \
        experiments/dg_dycore/post/zonal_means.jl <output_dir> [spinup_days]
=#

import ClimaAnalysis
import CairoMakie

function zonal_mean_panels(output_dir; spinup_days = nothing)
    simdir = ClimaAnalysis.SimDir(output_dir)
    for (short_name, cmap, label) in (
        ("ua", :balance, "zonal-mean u [m/s]"),
        ("ta", :thermal, "zonal-mean T [K]"),
    )
        var = ClimaAnalysis.get(simdir; short_name)
        times = ClimaAnalysis.times(var)
        t0 = isnothing(spinup_days) ? times[end ÷ 2 + 1] :
             spinup_days * 86400.0
        var = ClimaAnalysis.window(var, "time"; left = t0)
        zm = ClimaAnalysis.average_time(ClimaAnalysis.average_lon(var))
        # remaining dims: (lat, z)
        lats = zm.dims["lat"]
        zs = zm.dims["z_reference"] ./ 1e3
        fig = CairoMakie.Figure()
        ax = CairoMakie.Axis(
            fig[1, 1];
            xlabel = "latitude [deg]",
            ylabel = "z_reference [km]",
            title = "$label, mean over t ≥ $(round(t0 / 86400; digits = 1)) d",
        )
        plt = CairoMakie.contourf!(
            ax,
            lats,
            zs,
            zm.data;
            colormap = cmap,
            levels = 20,
        )
        CairoMakie.Colorbar(fig[1, 2], plt)
        out = joinpath(output_dir, "$(short_name)_zonal_mean.png")
        CairoMakie.save(out, fig)
        @info "wrote $out"
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    isempty(ARGS) && error("usage: zonal_means.jl <output_dir> [spinup_days]")
    spinup = length(ARGS) ≥ 2 ? parse(Float64, ARGS[2]) : nothing
    zonal_mean_panels(ARGS[1]; spinup_days = spinup)
end
