using Plots
using Plots.PlotMeasures
using JLD2

include("cli_options.jl")
if !(@isdefined parsed_args)
    (s, parsed_args) = parse_commandline()
end
parse_arg(pa, key, default) = isnothing(pa[key]) ? default : pa[key]

job_id = if isnothing(parsed_args["job_id"])
    job_id_from_parsed_args(s, parsed_args)
else
    parsed_args["job_id"]
end

output_dir = parse_arg(parsed_args, "output_dir", job_id)
# Note
# Low-resolution simulation is integrated for 10 days
# High-resolution simulation is integrated for 1 hour

# tempest scaling data
if occursin("low", output_dir)
    # low-resolution (integration time = 10 days, on Caltech central cluster)
    nprocs_tempest = [1, 2, 3, 6, 24]
    walltime_tempest = [74.41848, 34.43472, 23.22432, 11.83896, 3.60936]
    resolution = "low-resolution"
    t_int = "10 days"
    sypd_tempest = (60 * 60 * 24) ./ walltime_tempest ./ (8760 / 10 / 24)
else
    # high-resolution (integration time = 1 day, on Caltech central cluster)
    nprocs_tempest = [1, 2, 3, 6, 24, 54, 96, 216]
    walltime_tempest = [
        4673.969568,
        2338.740864,
        1505.628864,
        755.51184,
        245.157408,
        130.145184,
        76.487328,
        32.298912,
    ]
    walltime_tempest .*= (1.0 / 1 / 24) # walltime for 1 hour of integration time
    # cut-off at 54
    nprocs_tempest = nprocs_tempest[1:6]
    walltime_tempest = walltime_tempest[1:6]
    resolution = "high-resolution"
    t_int = "1 hour"
    sypd_tempest = (60 * 60 * 24) ./ walltime_tempest ./ 8760
end
cpu_hours_tempest = nprocs_tempest .* walltime_tempest / (60 * 60)
scaling_efficiency_tempest =
    trunc.(
        100 * (walltime_tempest[1] ./ nprocs_tempest) ./ walltime_tempest,
        digits = 1,
    )
#--------------------
I = Int
FT = Float64

nprocs = I[]
walltime = FT[]

for filename in readdir(output_dir)
    if occursin("scaling_data_", filename)
        dict = load(joinpath(output_dir, filename))
        push!(nprocs, I(dict["nprocs"]))
        push!(walltime, FT(dict["walltime"]))
    end
end

order = sortperm(nprocs)
nprocs, walltime = nprocs[order], walltime[order]
cpu_hours = nprocs .* walltime / (60 * 60)
if t_int == "1 hour"
    sypd = (60 * 60 * 24) ./ walltime ./ 8760
else
    sypd = (60 * 60 * 24) ./ walltime ./ (8760 / 10 / 24)
end
scaling_efficiency =
    trunc.(100 * (walltime[1] ./ nprocs) ./ walltime, digits = 1)

function linkfig(figpath, alt = "")
    # buildkite-agent upload figpath
    # link figure in logs if we are running on CI
    if get(ENV, "BUILDKITE", "") == "true"
        artifact_url = "artifact://$figpath"
        print("\033]1338;url='$(artifact_url)';alt='$(alt)'\a\n")
    end
end

figpath = joinpath(output_dir, resolution * "_" * "sypd.png")

Plots.GRBackend()
Plots.png(
    plot(
        [nprocs nprocs_tempest],
        [sypd sypd_tempest],
        markershape = [:circle :square],
        xticks = (nprocs, [string(i) for i in nprocs]),
        xlabel = "nprocs",
        ylabel = "SYPD",
        title = "Simulated years per day",
        label = ["ClimaAtmos" "Tempest"],
        legend = :right,
        grid = :true,
        left_margin = 10mm,
        bottom_margin = 10mm,
        top_margin = 10mm,
    ),
    figpath,
)
linkfig(relpath(figpath, joinpath(@__DIR__, "../..")), "SYPD")

figpath = joinpath(output_dir, resolution * "_" * "Scaling.png")

Plots.GRBackend()
Plots.png(
    plot(
        [nprocs nprocs_tempest],
        [cpu_hours cpu_hours_tempest],
        markershape = [:circle :square],
        xticks = (nprocs, [string(i) for i in nprocs]),
        xlabel = "nprocs",
        ylabel = "CPU hours",
        title = "Scaling data (integration time = " * t_int * ")",
        label = ["ClimaAtmos" "Tempest"],
        legend = :topleft,
        grid = :true,
        left_margin = 10mm,
        bottom_margin = 10mm,
        top_margin = 10mm,
    ),
    figpath,
)

linkfig(relpath(figpath, joinpath(@__DIR__, "../..")), "Scaling Data")

figpath = joinpath(output_dir, resolution * "_" * "Scaling_efficiency.png")

Plots.GRBackend()
Plots.png(
    plot(
        [nprocs nprocs_tempest],
        [scaling_efficiency scaling_efficiency_tempest],
        markershape = [:circle :square],
        xticks = (nprocs, [string(i) for i in nprocs]),
        ylims = ((0, 110)),
        xlabel = "nprocs",
        ylabel = "Efficiency",
        title = "Scaling efficiency",
        label = ["ClimaAtmos" "Tempest"],
        legend = :bottom,
        grid = :true,
        left_margin = 10mm,
        bottom_margin = 10mm,
        top_margin = 10mm,
        annotations = [
            (
                nprocs,
                scaling_efficiency .- 5,
                [string(i) * "%" for i in scaling_efficiency],
                4,
            ),
            (
                nprocs_tempest,
                scaling_efficiency_tempest .- 5,
                [string(i) * "%" for i in scaling_efficiency_tempest],
                4,
            ),
        ],
    ),
    figpath,
)

linkfig(relpath(figpath, joinpath(@__DIR__, "../..")), "Scaling Efficiency")
