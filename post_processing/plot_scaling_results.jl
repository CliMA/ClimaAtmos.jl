using Plots
using Plots.PlotMeasures
using JLD2

job_id = ARGS[1]
output_dir = "./"

secs_per_hour = 60 * 60
secs_per_day = 60 * 60 * 24
days_per_year = 8760 / 24

if occursin("low", job_id)
    resolution = "low-resolution"
    z_elem = 10
    t_int_days = 10 # integration time
elseif occursin("mid", job_id)
    z_elem = 45
    resolution = "mid-resolution"
    t_int_days = 4 # integration time
else
    z_elem = 45
    resolution = "high-resolution"
    t_int_days = 1 # integration time
end
t_int = string(t_int_days) * " days"

# read ClimaAtmos scaling data
I, FT = Int, Float64
nprocs_clima_atmos = I[]
ncols_per_process = I[]
walltime_clima_atmos = FT[]

for foldername in filter(isdir, readdir(output_dir))
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
        push!(ncols_per_process, I(dict["ncols_per_process"]))
        push!(walltime_clima_atmos, FT(dict["walltime"]))
    end
end
order = sortperm(nprocs_clima_atmos)
nprocs_clima_atmos, ncols_per_process, walltime_clima_atmos =
    nprocs_clima_atmos[order],
    ncols_per_process[order],
    walltime_clima_atmos[order]
# normalize ncols to columns with 45 levels
ncols_per_process = trunc.(ncols_per_process .* (z_elem / 45), digits = 1)
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
plt1 = Plots.plot(
    nprocs_clima_atmos,
    sypd_clima_atmos,
    markershape = :circle,
    markercolor = :blue,
    xticks = (nprocs_clima_atmos, [string(i) for i in nprocs_clima_atmos]),
    xaxis = :log,
    yaxis = :log,
    minorgrid = true,
    xlabel = "# of MPI processes",
    ylabel = "SYPD",
    title = "Simulated years per day",
    label = "ClimaAtmos (Float32)",
    legend = :topleft,
    grid = :true,
    left_margin = 10mm,
    right_margin = 10mm,
    bottom_margin = 10mm,
    top_margin = 10mm,
)
Plots.png(plt1, joinpath(output_dir, resolution * "_" * "sypd"))
Plots.pdf(plt1, joinpath(output_dir, resolution * "_" * "sypd"))

Plots.GRBackend()
plt2 = Plots.plot(
    nprocs_clima_atmos,
    cpu_hours_clima_atmos,
    markershape = :circle,
    markercolor = :blue,
    xticks = (nprocs_clima_atmos, [string(i) for i in nprocs_clima_atmos]),
    xaxis = :log,
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
plt3 = Plots.plot(
    nprocs_clima_atmos,
    scaling_efficiency_clima_atmos,
    markershape = :circle,
    markercolor = :blue,
    xticks = (nprocs_clima_atmos, [string(i) for i in nprocs_clima_atmos]),
    xaxis = :log,
    xlabel = "# of MPI processes",
    ylabel = "Efficiency (T1/N)/TN",
    title = "Scaling efficiency (T_int = $t_int)",
    label = "ClimaAtmos (Float32)",
    legend = :topright,
    grid = :true,
    left_margin = 10mm,
    bottom_margin = 10mm,
    top_margin = 10mm,
)
Plots.png(plt3, joinpath(output_dir, resolution * "_" * "Scaling_efficiency"))
Plots.pdf(plt3, joinpath(output_dir, resolution * "_" * "Scaling_efficiency"))

Plots.GRBackend()
plt4 = Plots.plot(
    ncols_per_process,
    scaling_efficiency_clima_atmos,
    markershape = :circle,
    markercolor = :blue,
    xaxis = :log,
    minorticks = true,
    xlabel = "# of columns per process (with 45 levels)",
    ylabel = "Efficiency (T1/N)/TN",
    title = "Scaling efficiency (T_int = $t_int)",
    label = "ClimaAtmos (Float32)",
    legend = :topleft,
    grid = :true,
    left_margin = 10mm,
    bottom_margin = 10mm,
    top_margin = 10mm,
)
Plots.png(
    plt4,
    joinpath(output_dir, resolution * "_" * "Scaling_efficiency_vs_ncols"),
)
Plots.pdf(
    plt4,
    joinpath(output_dir, resolution * "_" * "Scaling_efficiency_vs_ncols"),
)
