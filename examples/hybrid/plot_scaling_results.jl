using Plots
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
@show output_dir
I = Int
FT = Float64

nprocs = I[]
walltime = FT[]

for filename in readdir(output_dir)
    if occursin("scaling_data_", filename)
        @show filename
        dict = load(joinpath(output_dir, filename))
        push!(nprocs, I(dict["nprocs"]))
        push!(walltime, FT(dict["walltime"]))
    end
end
@show nprocs
@show walltime

order = sortperm(nprocs)
nprocs, walltime = nprocs[order], walltime[order]
nprocs_string = [string(i) for i in nprocs]

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
        walltime,
        markershape = :circle,
        xticks = (log2.(nprocs), nprocs_string),
        xlabel = "nprocs",
        ylabel = "wall time (sec)",
        title = "Scaling data",
        label = "simulation time = 1 hour",
        legend = :topright,
        grid = :true,
    ),
    figpath,
)

linkfig(relpath(figpath, joinpath(@__DIR__, "../..")), "Scaling Data")
