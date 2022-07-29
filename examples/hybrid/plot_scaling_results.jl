using Plots
using Plots.PlotMeasures
using JLD2

include("cli_options.jl")
if !(@isdefined parsed_args)
    (s, parsed_args) = parse_commandline()
end

job_id =
    isnothing(parsed_args["job_id"]) ? job_id_from_parsed_args(s, parsed_args) :
    parsed_args["job_id"]
output_dir = "./"

secs_per_hour = 60 * 60
secs_per_day = 60 * 60 * 24
days_per_year = 8760 / 24

if occursin("low", job_id)
    resolution = "low-resolution"
    t_int_days = 10 # integration time
else
    resolution = "high-resolution"
    t_int_days = 1 # integration time
end
t_int = string(t_int_days) * " days"

# read ClimaAtmos scaling data
I, FT = Int, Float64
nprocs_clima_atmos = I[]
walltime_clima_atmos = FT[]

for foldername in readdir(output_dir)
    if occursin(job_id, foldername) && !occursin("comparison", foldername)
        nprocs_string = split(foldername, "_")[end]
        dict = load(
            joinpath(
                output_dir,
                foldername,
                "scaling_data_$(nprocs_string)_processes.jld2",
            ),
        )
        push!(nprocs_clima_atmos, I(dict["nprocs"]))
        push!(walltime_clima_atmos, FT(dict["walltime"]))
    end
end
order = sortperm(nprocs_clima_atmos)
nprocs_clima_atmos, walltime_clima_atmos =
    nprocs_clima_atmos[order], walltime_clima_atmos[order]
# simulated years per day
sypd_clima_atmos =
    (secs_per_day ./ walltime_clima_atmos) * t_int_days ./ days_per_year
# CPU hours
cpu_hours_clima_atmos =
    nprocs_clima_atmos .* walltime_clima_atmos / secs_per_hour
# scaling efficiency
single_proc_time_clima_atmos = walltime_clima_atmos[1] * nprocs_clima_atmos[1]
scaling_efficiency_clima_atmos =
    trunc.(
        100 * (single_proc_time_clima_atmos ./ nprocs_clima_atmos) ./
        walltime_clima_atmos,
        digits = 1,
    )
ENV["GKSwstype"] = "100"
Plots.GRBackend()
plt1 = plot(
    log2.(nprocs_clima_atmos),
    sypd_clima_atmos,
    markershape = :circle,
    markercolor = :blue,
    xticks = (
        log2.(nprocs_clima_atmos),
        [string(i) for i in nprocs_clima_atmos],
    ),
    xlabel = "# of MPI processes",
    ylabel = "SYPD",
    title = "Simulated years per day",
    label = "ClimaAtmos (Float32)",
    legend = :topleft,
    grid = :true,
    left_margin = 10mm,
    bottom_margin = 10mm,
    top_margin = 10mm,
)
Plots.png(plt1, joinpath(output_dir, resolution * "_" * "sypd"))
Plots.pdf(plt1, joinpath(output_dir, resolution * "_" * "sypd"))

Plots.GRBackend()
plt2 = plot(
    log2.(nprocs_clima_atmos),
    cpu_hours_clima_atmos,
    markershape = :circle,
    markercolor = :blue,
    xticks = (
        log2.(nprocs_clima_atmos),
        [string(i) for i in nprocs_clima_atmos],
    ),
    xlabel = "# of MPI processes",
    ylabel = "CPU hours",
    title = "Scaling data (T_int = $t_int)",
    label = "ClimaAtmos (Float32)",
    legend = :topleft,
    grid = :true,
    left_margin = 10mm,
    bottom_margin = 10mm,
    top_margin = 10mm,
)
Plots.png(plt2, joinpath(output_dir, resolution * "_" * "Scaling"))
Plots.pdf(plt2, joinpath(output_dir, resolution * "_" * "Scaling"))


Plots.GRBackend()
plt3 = plot(
    log2.(nprocs_clima_atmos),
    scaling_efficiency_clima_atmos,
    markershape = :circle,
    markercolor = :blue,
    xticks = (
        log2.(nprocs_clima_atmos),
        [string(i) for i in nprocs_clima_atmos],
    ),
    xlabel = "# of MPI processes",
    ylabel = "Efficiency (T1/N)/TN",
    title = "Scaling efficiency (T_int = $t_int)",
    label = "ClimaAtmos (Float32)",
    legend = :topright,
    grid = :true,
    left_margin = 10mm,
    bottom_margin = 10mm,
    top_margin = 10mm,
    annotations = [
        (
            log2.(nprocs_clima_atmos),
            scaling_efficiency_clima_atmos .- 5,
            [string(i) * "%" for i in scaling_efficiency_clima_atmos],
            4,
        ),
    ],
)
Plots.png(plt3, joinpath(output_dir, resolution * "_" * "Scaling_efficiency"))
Plots.pdf(plt3, joinpath(output_dir, resolution * "_" * "Scaling_efficiency"))
