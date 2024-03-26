using CairoMakie
using JLD2

job_id = "gpu_aquaplanet_dyamond_ss"
output_dir = "./"

secs_per_hour = 60 * 60
secs_per_day = 60 * 60 * 24
days_per_year = 8760 / 24

t_int_days = 12 / 24 # simulation integration time in days
h_elem = 30
z_elem = 63
nlevels = z_elem + 1

t_int = string(t_int_days) * " days"

# read ClimaAtmos scaling data
I, FT = Int, Float64
nprocs_clima_atmos = I[]
ncols_per_process = I[]
walltime_clima_atmos = FT[]
for foldername in readdir(output_dir)
    if occursin(job_id, foldername) && occursin("_ss_", foldername)
        nprocs_string = split(split(foldername, "_ss_")[end], "process")[1]
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

# simulated years per day
sypd_clima_atmos =
    (secs_per_day ./ walltime_clima_atmos) * t_int_days ./ days_per_year
# GPU hours
gpu_hours_clima_atmos =
    nprocs_clima_atmos .* walltime_clima_atmos / secs_per_hour
# scaling efficiency
single_proc_time_clima_atmos = walltime_clima_atmos[1] * nprocs_clima_atmos[1]
scaling_efficiency_clima_atmos =
    trunc.(
        100 * (single_proc_time_clima_atmos ./ nprocs_clima_atmos) ./
        walltime_clima_atmos,
        digits = 1,
    )
num_ticks = length(sypd_clima_atmos)

min_tick, max_tick = extrema(sypd_clima_atmos)
tick_size = (max_tick - min_tick) / num_ticks

fig = Figure(; size = (1200, 900))
Makie.Label(
    fig[begin - 1, 1:2],
    "GPU strong scaling";
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
    xticks = (nprocs_clima_atmos, [string(i) for i in nprocs_clima_atmos]),
    xticksvisible = true,
    xscale = log10,
    yscale = log10,
    ytickformat = "{:.2f}",
)
scatterlines!(nprocs_clima_atmos, sypd_clima_atmos)

ax2 = Axis(
    fig[2, 1],
    xlabel = "# of MPI processes",
    ylabel = "GPU hours",
    title = "Scaling data (T_int = $t_int)",
    xticks = (nprocs_clima_atmos, [string(i) for i in nprocs_clima_atmos]),
    xscale = log,
    xgridvisible = true,
    ygridvisible = true,
)
scatterlines!(ax2, nprocs_clima_atmos, gpu_hours_clima_atmos)

ax3 = Axis(
    fig[1, 2],
    xlabel = "# of MPI processes",
    ylabel = "Efficiency (T1/N)/TN",
    title = "Scaling efficiency (T_int = $t_int)",
    xticks = (nprocs_clima_atmos, string.(nprocs_clima_atmos)),
    xscale = log,
    xgridvisible = true,
    ygridvisible = true,
)
scatterlines!(ax3, nprocs_clima_atmos, scaling_efficiency_clima_atmos)

min_tick, max_tick = extrema(ncols_per_process)
tick_size = (max_tick - min_tick) / num_ticks

ax4 = Axis(
    fig[2, 2],
    xlabel = "# of columns per process",
    ylabel = "Efficiency (T1/N)/TN",
    title = "Scaling efficiency (T_int = $t_int)",
    xticks = (ncols_per_process, string.(ncols_per_process)),
    xscale = log10,
    xgridvisible = true,
    ygridvisible = true,
)
scatterlines!(ax4, ncols_per_process, scaling_efficiency_clima_atmos)

save(joinpath(output_dir, job_id, "GPU_strong_scaling.png"), fig)
save(joinpath(output_dir, job_id, "GPU_strong_scaling.pdf"), fig)
