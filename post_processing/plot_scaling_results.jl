using CairoMakie
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

num_ticks = 4

min_tick, max_tick = extrema(sypd_clima_atmos)
tick_size = (max_tick - min_tick) / num_ticks

fig = Figure(resolution = (1200, 900))
Makie.Label(
    fig[begin - 1, 1:2],
    "$resolution scaling";
    font = :bold,
    fontsize = 20,
)
ax1 = Axis(
    fig[1, 1],
    xlabel = "# of MPI processes",
    ylabel = "SYPD",
    title = "Simulated years per day",
    xticks = (nprocs_clima_atmos, [string(i) for i in nprocs_clima_atmos]),
    xscale = log10,
    yscale = log10,
    xgridvisible = true,
    ygridvisible = false,
)
scatterlines!(nprocs_clima_atmos, sypd_clima_atmos)
# Plot a second axis to display tick labels clearly
ax1 = Axis(
    fig[1, 1],
    yaxisposition = :right,
    yticklabelalign = (:left, :center),
    xticklabelsvisible = false,
    yticklabelsvisible = true,
    xlabelvisible = false,
    xgridvisible = false,
    xticksvisible = true,
    xscale = log10,
    yscale = log10,
    ytickformat = "{:.2f}",
)
scatterlines!(nprocs_clima_atmos, sypd_clima_atmos)

ax2 = Axis(
    fig[2, 1],
    xlabel = "# of MPI processes",
    ylabel = "CPU hours",
    title = "Scaling data (T_int = $t_int)",
    xticks = (nprocs_clima_atmos, [string(i) for i in nprocs_clima_atmos]),
    xscale = log,
    xgridvisible = true,
    ygridvisible = true,
)
scatterlines!(ax2, nprocs_clima_atmos, cpu_hours_clima_atmos)

ax3 = Axis(
    fig[1, 2],
    xlabel = "# of MPI processes",
    ylabel = "Efficiency (T1/N)/TN",
    title = "Scaling efficiency (T_int = $t_int)",
    xticks = (nprocs_clima_atmos, [string(i) for i in nprocs_clima_atmos]),
    xscale = log,
    xgridvisible = true,
    ygridvisible = true,
)
scatterlines!(ax3, nprocs_clima_atmos, scaling_efficiency_clima_atmos)

min_tick, max_tick = extrema(ncols_per_process)
tick_size = (max_tick - min_tick) / num_ticks

ax4 = Axis(
    fig[2, 2],
    xlabel = "# of columns per process (with 45 levels)",
    ylabel = "Efficiency (T1/N)/TN",
    title = "Scaling efficiency (T_int = $t_int)",
    xscale = log10,
    xgridvisible = true,
    ygridvisible = true,
)
scatterlines!(ax4, ncols_per_process, scaling_efficiency_clima_atmos)

save("$resolution.png", fig)
save("$resolution.pdf", fig)
