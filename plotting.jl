using ClimaAnalysis
using CairoMakie

var = OutputVar("output/42/output_active/pfull_coords/ta_1d_average.nc")
var = slice(var, t = 1186400)
var = slice(var, pressure_level = 55000)

fig = CairoMakie.Figure();
ClimaAnalysis.Visualize.heatmap2D!(fig, var)
CairoMakie.save("ta_from_pfull_coords.png", fig, more_kwargs = Dict(:plot => Dict(), :cb => Dict(:limits => (230, 270)), :axis => Dict()))






var = OutputVar("output/42/output_active/ta_1d_average.nc")
var = slice(var, t = 1186400)
var = slice(var, z = 5000)

fig = CairoMakie.Figure();
wads = ClimaAnalysis.Visualize.heatmap2D!(fig, var)
CairoMakie.save("ta_from_z_coords.png", fig, more_kwargs = Dict(:plot => Dict(), :cb => Dict(:limits => (230, 270)), :axis => Dict()))
