using Plots
using Plots.PlotMeasures
using JLD2

job_id = ARGS[1]
output_dir = "./"

secs_per_hour = 60 * 60
secs_per_day = 60 * 60 * 24
days_per_year = 8760 / 24

# tempest scaling data
if occursin("low", job_id)
    resolution = "low-resolution"
    nprocs_tempest = [1, 2, 3, 6, 24]
    walltime_tempest = [102.5892, 49.0212, 33.41088, 17.064, 5.77584]
    t_int_days_tempest = 10 # integration time for Tempest
    t_int_days_climaatmos = 10 # integration time for ClimaAtmos
else
    resolution = "high-resolution"
    nprocs_tempest = [1, 2, 3, 6, 24, 54, 96, 216]
    walltime_tempest = [
        6186.595968,
        3197.915424,
        2160.269568,
        1131.526368,
        385.558272,
        174.662784,
        139.771872,
        45.971712,
        43.602624,
    ]
    t_int_days_tempest = 1 # integration time for Tempest
    t_int_days_climaatmos = 1 # integration time for ClimaAtmos
end
t_int_days = t_int_days_climaatmos
t_int = string(t_int_days_climaatmos) * " days"
# normalalize Tempest walltime for comparison with ClimaAtmos (if necessary)
walltime_tempest .*= (t_int_days / t_int_days_tempest)

# read ClimaAtmos scaling data
I, FT = Int, Float64
nprocs_clima_atmos = I[]
walltime_clima_atmos = FT[]
for foldername in readdir(output_dir)
    if occursin(job_id, foldername)
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

# update data if all data points are not available for ClimaAtmos
minprocs = nprocs_clima_atmos[1]
loc = findfirst(nprocs_tempest .== minprocs)
nprocs_tempest = nprocs_tempest[loc:end]
walltime_tempest = walltime_tempest[loc:end]

@assert nprocs_clima_atmos == nprocs_tempest # needed for comparison plot

# simulated years per day
sypd_tempest = (secs_per_day ./ walltime_tempest) * t_int_days ./ days_per_year
sypd_clima_atmos =
    (secs_per_day ./ walltime_clima_atmos) * t_int_days ./ days_per_year
# CPU hours
cpu_hours_tempest = nprocs_tempest .* walltime_tempest / secs_per_hour
cpu_hours_clima_atmos =
    nprocs_clima_atmos .* walltime_clima_atmos / secs_per_hour
# scaling efficiency
single_proc_time_tempest = walltime_tempest[1] * nprocs_tempest[1]
scaling_efficiency_tempest =
    trunc.(
        100 * (single_proc_time_tempest ./ nprocs_tempest) ./ walltime_tempest,
        digits = 1,
    )
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
    [log2.(nprocs_clima_atmos) log2.(nprocs_tempest)],
    [sypd_clima_atmos sypd_tempest],
    markershape = [:circle :square],
    markercolor = [:blue :orange],
    xticks = (
        log2.(nprocs_clima_atmos),
        [string(i) for i in nprocs_clima_atmos],
    ),
    xlabel = "# of MPI processes",
    ylabel = "SYPD",
    yaxis = :log,
    title = "Simulated years per day",
    label = ["ClimaAtmos (Float64)" "Tempest (Float64)"],
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
    [log2.(nprocs_clima_atmos) log2.(nprocs_tempest)],
    [cpu_hours_clima_atmos cpu_hours_tempest],
    markershape = [:circle :square],
    markercolor = [:blue :orange],
    xticks = (
        log2.(nprocs_clima_atmos),
        [string(i) for i in nprocs_clima_atmos],
    ),
    xlabel = "# of MPI processes",
    ylabel = "CPU hours",
    title = "Scaling data (T_int = $t_int)",
    label = ["ClimaAtmos (Float64)" "Tempest (Float64)"],
    ylims = (0.0, Inf),
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
    cpu_hours_clima_atmos ./ cpu_hours_tempest,
    markershape = [:circle :square],
    markercolor = [:blue :orange],
    xticks = (
        log2.(nprocs_clima_atmos),
        [string(i) for i in nprocs_clima_atmos],
    ),
    xlabel = "# of MPI processes",
    ylabel = "ratio",
    ylims = (0.0, Inf),
    label = "Ratio of CPU hours (ClimaAtmos/Tempest) (Float64)",
    title = "Comparison with Tempest",
    legend = :topleft,
    grid = :true,
    left_margin = 10mm,
    bottom_margin = 10mm,
    top_margin = 10mm,
)
Plots.png(plt3, joinpath(output_dir, resolution * "_" * "comparison"))
Plots.pdf(plt3, joinpath(output_dir, resolution * "_" * "comparison"))


Plots.GRBackend()
plt4 = plot(
    [log2.(nprocs_clima_atmos) log2.(nprocs_tempest)],
    [scaling_efficiency_clima_atmos scaling_efficiency_tempest],
    markershape = [:circle :square],
    markercolor = [:blue :orange],
    xticks = (
        log2.(nprocs_clima_atmos),
        [string(i) for i in nprocs_clima_atmos],
    ),
    xlabel = "# of MPI processes",
    ylabel = "Efficiency (T1/N)/TN",
    title = "Scaling efficiency (T_int = $t_int)",
    label = ["ClimaAtmos (Float64)" "Tempest (Float64)"],
    ylims = (0.0, Inf),
    legend = :bottom,
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
        (
            log2.(nprocs_tempest),
            scaling_efficiency_tempest .- 5,
            [string(i) * "%" for i in scaling_efficiency_tempest],
            4,
        ),
    ],
)
Plots.png(plt4, joinpath(output_dir, resolution * "_" * "Scaling_efficiency"))
Plots.pdf(plt4, joinpath(output_dir, resolution * "_" * "Scaling_efficiency"))
