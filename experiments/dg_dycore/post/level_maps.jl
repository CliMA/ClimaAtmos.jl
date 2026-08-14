#=
Single-level lat–lon maps of ua and ta from the DG dycore's NetCDF output,
at a fixed z_REFERENCE altitude (default 1 km).

COORDINATE NOTE: the sandbox's NetCDFWriter uses ClimaDiagnostics'
`LevelsMethod` — no vertical interpolation, so the file's `z` dimension is
the grid's REFERENCE levels. Over terrain (LinearAdaption/SLEVE) a slice at
z_ref = 1 km is therefore a terrain-following coordinate surface whose true
altitude varies with the surface elevation underneath (up to z_sfc + ~1 km
over the highest topography), NOT a constant-altitude surface. That is
usually what you want for near-surface fields — a constant-z surface would
intersect the terrain. The plot titles say z_ref to keep this explicit.

Usage:
    julia --project=experiments/dg_dycore/post \
        experiments/dg_dycore/post/level_maps.jl <output_dir> [z_ref_km] [days]

`days` is a comma-separated list (default: last snapshot only); writes
level_maps_zref<z>km_day<d>.png into <output_dir>.
=#

import ClimaAnalysis
import CairoMakie

const LEVEL_PANELS = (
    (short_name = "ua", cmap = :balance, sym = true, label = "u [m/s]"),
    (short_name = "ta", cmap = :thermal, sym = false, label = "T [K]"),
)

function level_maps(output_dir; z_ref_km = 1.0, days = nothing)
    simdir = ClimaAnalysis.SimDir(output_dir)
    z_ref = 1e3 * z_ref_km
    ref = ClimaAnalysis.get(simdir; short_name = "ta")
    ts = ClimaAnalysis.times(ref)
    isempty(ts) && error("$output_dir has no written snapshots (empty time \
                          axis) — the run likely died before diag_period")
    sel = if isnothing(days)
        [ts[end]]
    else
        [ts[argmin(abs.(ts .- d * 86400.0))] for d in days]
    end
    # the vertical dim is "z_reference" over hypsography ("z" on flat
    # grids); slice() picks the nearest stored level — report which
    zname = ClimaAnalysis.altitude_name(ref)
    zs = ref.dims[zname]
    z_near = zs[argmin(abs.(zs .- z_ref))]
    z_near == z_ref ||
        @info "nearest stored reference level" requested = z_ref actual =
            z_near
    for t in unique(sel)
        day = round(t / 86400.0; digits = 2)
        fig = CairoMakie.Figure(; size = (1100, 380))
        for (k, p) in enumerate(LEVEL_PANELS)
            var = ClimaAnalysis.get(simdir; short_name = p.short_name)
            if t ∉ ClimaAnalysis.times(var)
                @warn "$(p.short_name) has no snapshot at day $day — skipped \
                       (partially written output?)"
                continue
            end
            sl = ClimaAnalysis.slice(var; time = t, Symbol(zname) => z_ref)
            lons = sl.dims["lon"]
            lats = sl.dims["lat"]
            data = sl.data
            ax = CairoMakie.Axis(
                fig[1, 2 * k - 1];
                xlabel = "longitude [deg]",
                ylabel = "latitude [deg]",
                title = "$(p.label), z_ref = $(round(z_near / 1e3; digits = 2)) km, day $(day)",
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
            CairoMakie.Colorbar(fig[1, 2 * k], plt)
        end
        out = joinpath(output_dir, "level_maps_zref$(z_ref_km)km_day$(day).png")
        CairoMakie.save(out, fig)
        @info "wrote $out"
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    isempty(ARGS) && error(
        "usage: level_maps.jl <output_dir> [z_ref_km] [days,comma,separated]",
    )
    z_ref_km = length(ARGS) ≥ 2 ? parse(Float64, ARGS[2]) : 1.0
    days = length(ARGS) ≥ 3 ? parse.(Float64, split(ARGS[3], ",")) : nothing
    level_maps(ARGS[1]; z_ref_km, days)
end
