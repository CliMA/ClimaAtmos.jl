# Visualization companion — ClimaAtmos.jl 1-hour hands-on tutorial
#
# Run this in a *default* Julia session, not `julia --project` in the
# ClimaAtmos clone (do not add plotting packages to the model environment):
#
#   julia +1.11
#   import Pkg; Pkg.add(["ClimaAnalysis", "CairoMakie"])
#
# Then, from the ClimaAtmos.jl repository root:
#
#   include("talks/tutorial_1h_plots.jl")
#
# Output figures land in the current working directory.

using ClimaAnalysis
using ClimaAnalysis: SimDir, times, average_lon
import ClimaAnalysis.Visualize as viz
import CairoMakie

function _latest(job_id)
    path = joinpath("output", job_id, "output_active")
    isdir(path) || error("No output at $path — run talks/tutorial_1h.jl first.")
    return path
end

# ---------------------------------------------------------------------------
# Column: vertical profiles
# ---------------------------------------------------------------------------
bomex = SimDir(_latest("workshop_bomex"))
@show bomex

ta = get(bomex, "ta")     # air temperature
hus = get(bomex, "hus")   # specific humidity

t0 = first(times(ta))
t1 = last(times(ta))

fig = CairoMakie.Figure(; size = (900, 400))
viz.plot!(fig[1, 1], ta; time = t0, lon = 0, lat = 0)
viz.plot!(fig[1, 2], hus; time = t0, lon = 0, lat = 0)
CairoMakie.save("workshop_bomex_ic.png", fig)

fig = CairoMakie.Figure(; size = (900, 400))
viz.plot!(fig[1, 1], ta; lon = 0, lat = 0)
viz.plot!(fig[1, 2], hus; lon = 0, lat = 0)
CairoMakie.save("workshop_bomex_evolution.png", fig)

# ---------------------------------------------------------------------------
# Sphere: zonal mean and a horizontal slice
# ---------------------------------------------------------------------------
bcw = SimDir(_latest("workshop_bcw"))
@show bcw

ua = get(bcw, "ua")
ta = get(bcw, "ta")
t_end = last(times(ua))

fig = CairoMakie.Figure(; size = (700, 450))
viz.plot!(fig, average_lon(ua); time = t_end)
CairoMakie.save("workshop_bcw_ua_zonal_mean.png", fig)

fig = CairoMakie.Figure(; size = (700, 450))
viz.plot!(fig, ta; z = 0, time = t_end)
CairoMakie.save("workshop_bcw_ta_surface.png", fig)

println("Wrote workshop_bomex_ic.png, workshop_bomex_evolution.png,")
println("      workshop_bcw_ua_zonal_mean.png, workshop_bcw_ta_surface.png")
