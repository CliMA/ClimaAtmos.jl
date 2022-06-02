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

figpath = joinpath(output_dir, "Scaling.png")

Plots.GRBackend()
Plots.png(
    plot(
        log2.(nprocs),
        cpu_hours,
        markershape = :circle,
        xticks = (log2.(nprocs), [string(i) for i in nprocs]),
        xlabel = "nprocs",
        ylabel = "CPU hours",
        title = "Scaling data",
        label = "simulation time = 1 hour",
        legend = :top,
        grid = :true,
        left_margin = 10mm,
        bottom_margin = 10mm,
        top_margin = 10mm,
    ),
    figpath,
)

linkfig(relpath(figpath, joinpath(@__DIR__, "../..")), "Scaling Data")

figpath = joinpath(output_dir, "Scaling_efficiency.png")

Plots.GRBackend()
Plots.png(
    plot(
        log2.(nprocs),
        scaling_efficiency,
        markershape = :circle,
        xticks = (log2.(nprocs), [string(i) for i in nprocs]),
        ylims = ((0, 110)),
        xlabel = "nprocs",
        ylabel = "Efficiency",
        title = "Scaling efficiency",
        label = "simulation time = 1 hour",
        legend = :bottomleft,
        grid = :true,
        left_margin = 10mm,
        bottom_margin = 10mm,
        top_margin = 10mm,
        annotations = (
            log2.(nprocs),
            scaling_efficiency .- 5,
            [string(i) * "%" for i in scaling_efficiency],
            10,
        ),
    ),
    figpath,
)

linkfig(relpath(figpath, joinpath(@__DIR__, "../..")), "Scaling Efficiency")
