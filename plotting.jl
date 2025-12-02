using ClimaAnalysis
using CairoMakie
simdir = SimDir("/home/kphan/Desktop/work/ClimaAtmos.jl/pfull-coords/output/42/output_0047/pfull_coords")

var = get(simdir; short_name = "ta", reduction = "average")
var = slice(var, t = 86400)
var = slice(var, pressure_level = 55000)

fig = CairoMakie.Figure();
ClimaAnalysis.Visualize.heatmap2D!(fig, var)
CairoMakie.save("yo.png", fig, more_kwargs = Dict(:plot => Dict(), :cb => Dict(:limits => (230, 270)), :axis => Dict()))




var = OutputVar("/home/kphan/Desktop/work/ClimaAtmos.jl/pfull-coords/output/42/output_0047/ta_1d_average.nc")
var = slice(var, t = 86400)
var = slice(var, z = 5000)

fig = CairoMakie.Figure();
wads = ClimaAnalysis.Visualize.heatmap2D!(fig, var)
CairoMakie.save("yo1.png", fig, more_kwargs = Dict(:plot => Dict(), :cb => Dict(:limits => (230, 270)), :axis => Dict()))
