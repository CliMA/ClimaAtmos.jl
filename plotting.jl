using ClimaAnalysis
using CairoMakie
using NCDatasets

var = OutputVar("output/42/output_active/pfull_coords/ta_1d_average.nc")
var = slice(var, t = 1186400)
var = slice(var, pressure_level = 55000)

fig = CairoMakie.Figure();
ClimaAnalysis.Visualize.heatmap2D!(fig, var)
CairoMakie.save(
    "ta_from_pfull_coords.png",
    fig,
    more_kwargs = Dict(
        :plot => Dict(),
        :cb => Dict(:limits => (230, 270)),
        :axis => Dict(),
    ),
)






var = OutputVar("output/42/output_active/ta_1d_average.nc")
var = slice(var, t = 1186400)
var = slice(var, z = 5000)

fig = CairoMakie.Figure();
wads = ClimaAnalysis.Visualize.heatmap2D!(fig, var)
CairoMakie.save(
    "ta_from_z_coords.png",
    fig,
    more_kwargs = Dict(
        :plot => Dict(),
        :cb => Dict(:limits => (230, 270)),
        :axis => Dict(),
    ),
)

# Used to be 0040
reference = NCDataset(
    "/home/kphan/Desktop/work_tree/mainRepos/ClimaAtmos.jl/pfull-coords/output/42/output_0212/pfull_coords/pfull_1d_inst.nc",
)["pfull"][
    :,
    :,
    :,
    :,
]
reference = permutedims(reference, (1, 3, 4, 2))

current = NCDataset(
    "/home/kphan/Desktop/work_tree/mainRepos/ClimaAtmos.jl/pfull-coords/output/42/output_active/pfull_coords/pfull_1d_inst.nc",
)["pfull"][
    :,
    :,
    :,
    :,
]

squared_error = (reference .- current) .^ 2 |> sum |> sqrt
@info squared_error
