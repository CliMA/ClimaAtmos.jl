#=
Baroclinic-wave-style lat–lon maps from the DG dycore's NetCDF output,
sliced at a fixed altitude (default z = 2.5 km — above the Hughes &
Jablonowski 2 km peaks, so fields are shown on a level that clears the
terrain everywhere). One 2×2 panel figure per snapshot:

    ta  (temperature)         va (meridional wind)
    rv  (relative vorticity)  pfull (pressure)

va and rv are the wave-train signature fields (Hughes & Jablonowski 2023);
ta and pfull are the classic JW06 baroclinic-wave panels.

Usage:
    julia --project=experiments/dg_dycore/post \
        experiments/dg_dycore/post/wave_train_maps.jl <output_dir> [z_km] [days]

    z_km:  slice altitude in km (default 2.5)
    days:  comma-separated list of days to plot (default: every snapshot)

Writes wave_maps_z<z>km_day<d>.png into <output_dir>.
=#

import ClimaAnalysis
import CairoMakie

const PANELS = (
    (short_name = "ta", cmap = :thermal, sym = false, label = "T [K]"),
    (short_name = "va", cmap = :balance, sym = true, label = "v [m/s]"),
    (short_name = "rv", cmap = :balance, sym = true, label = "ζ [1/s]"),
    (short_name = "pfull", cmap = :viridis, sym = false, label = "p [Pa]"),
)

function wave_train_maps(output_dir; z_km = 2.5, days = nothing)
    simdir = ClimaAnalysis.SimDir(output_dir)
    z = 1e3 * z_km
    ref = ClimaAnalysis.get(simdir; short_name = "ta")
    ts = ClimaAnalysis.times(ref)
    sel = if isnothing(days)
        ts
    else
        [ts[argmin(abs.(ts .- d * 86400.0))] for d in days]
    end
    for t in unique(sel)
        day = round(t / 86400.0; digits = 2)
        fig = CairoMakie.Figure(; size = (1100, 750))
        for (k, p) in enumerate(PANELS)
            var = ClimaAnalysis.get(simdir; short_name = p.short_name)
            sl = ClimaAnalysis.slice(var; time = t, z = z)
            lons = sl.dims["lon"]
            lats = sl.dims["lat"]
            data = sl.data
            row, col = fldmod1(k, 2)
            ax = CairoMakie.Axis(
                fig[row, 2 * col - 1];
                xlabel = "longitude [deg]",
                ylabel = "latitude [deg]",
                title = "$(p.label), z = $(z_km) km, day $(day)",
            )
            levels = if p.sym
                a = max(maximum(abs, data), eps())
                range(-a, a; length = 21)
            else
                20
            end
            plt = CairoMakie.contourf!(
                ax,
                lons,
                lats,
                data;
                levels,
                colormap = p.cmap,
            )
            CairoMakie.Colorbar(fig[row, 2 * col], plt)
        end
        out = joinpath(output_dir, "wave_maps_z$(z_km)km_day$(day).png")
        CairoMakie.save(out, fig)
        @info "wrote $out"
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    isempty(ARGS) && error(
        "usage: wave_train_maps.jl <output_dir> [z_km] [days,comma,separated]",
    )
    z_km = length(ARGS) ≥ 2 ? parse(Float64, ARGS[2]) : 2.5
    days =
        length(ARGS) ≥ 3 ? parse.(Float64, split(ARGS[3], ",")) : nothing
    wave_train_maps(ARGS[1]; z_km, days)
end
