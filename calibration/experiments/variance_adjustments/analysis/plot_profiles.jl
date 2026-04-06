# Plot vertical profiles from a completed run (`output_active` under simulation_output).
# Usage (from experiment directory):
#   julia --project=. analysis/plot_profiles.jl <path_to_output_active>
#
import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

import CairoMakie as M
import ClimaAnalysis: SimDir, get, slice, average_lat, average_lon

path = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "simulation_output", "_reference_truth", "output_active")
!isdir(path) && error("Not a directory: $path")

simdir = SimDir(path)
theta = get(simdir; short_name = "thetaa", reduction = "average", period = "10mins")
t_last = ClimaAnalysis.times(theta)[end]
th = slice(theta, time = t_last)
prof = average_lat(average_lon(th))
z = ClimaAnalysis.coordinates(prof).z
fig = M.Figure()
ax = M.Axis(fig[1, 1]; xlabel = "θ (K)", ylabel = "z (m)")
M.lines!(ax, vec(prof.data), vec(z))
outpng = joinpath(@__DIR__, "profile_theta.png")
M.save(outpng, fig)
@info "Saved" outpng
