using CairoMakie
using JLD2

include("plot_gpu_scaling_utils.jl")

job_id = "gpu_aquaplanet_dyamond_ws"
output_dir = "./"

t_int_days = 12 / 24 # simulation integration time in days
h_elem = [30, 42, 60]
z_elem = 63
nlevels = z_elem + 1

t_int = string(t_int_days) * " days"

# read ClimaAtmos scaling data
(;
    nprocs_clima_atmos,
    ncols_per_process,
    walltime_clima_atmos,
    sypd_clima_atmos,
    gpu_hours_clima_atmos,
) = get_jld2data(output_dir, job_id, t_int_days, "_ws_")

# weak scaling efficiency
single_proc_time_clima_atmos = walltime_clima_atmos[1] * nprocs_clima_atmos[1]
weak_scaling_efficiency_clima_atmos =
    trunc.(
        100 * single_proc_time_clima_atmos ./ walltime_clima_atmos,
        digits = 1,
    )

num_ticks = length(sypd_clima_atmos)

min_tick, max_tick = extrema(sypd_clima_atmos)
tick_size = (max_tick - min_tick) / num_ticks

fig = Figure(; size = (1200, 900))
Makie.Label(
    fig[begin - 1, 1:2],
    "GPU weak scaling";
    font = :bold,
    fontsize = 20,
)
ax1 = Axis(
    fig[1, 1],
    xlabel = "(# of MPI processes, h_elem)",
    ylabel = "SYPD",
    title = "Simulated years per day",
    xticks = (
        nprocs_clima_atmos,
        [string(i) for i in zip(nprocs_clima_atmos, h_elem)],
    ),
    xscale = log10,
    xgridvisible = true,
    ygridvisible = false,
)
ylims!(ax1, 0.0, cld(maximum(sypd_clima_atmos), 1.0))
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
    ytickformat = "{:.2f}",
)
ylims!(ax1, 0.0, cld(maximum(sypd_clima_atmos), 1.0))
scatterlines!(nprocs_clima_atmos, sypd_clima_atmos)

ax2 = Axis(
    fig[2, 1],
    xlabel = "(# of MPI processes, h_elem)",
    ylabel = "GPU hours",
    title = "Scaling data (T_int = $t_int)",
    xticks = (
        nprocs_clima_atmos,
        [string(i) for i in zip(nprocs_clima_atmos, h_elem)],
    ),
    xscale = log,
    xgridvisible = true,
    ygridvisible = true,
)
scatterlines!(ax2, nprocs_clima_atmos, gpu_hours_clima_atmos)

ax3 = Axis(
    fig[1, 2],
    xlabel = "(# of MPI processes, h_elem)",
    ylabel = "Efficiency (T1/N)/TN",
    title = "Scaling efficiency (T_int = $t_int)",
    xticks = (
        nprocs_clima_atmos,
        [string(i) for i in zip(nprocs_clima_atmos, h_elem)],
    ),
    xscale = log,
    xgridvisible = true,
    ygridvisible = true,
)
scatterlines!(ax3, nprocs_clima_atmos, weak_scaling_efficiency_clima_atmos)

min_tick, max_tick = extrema(ncols_per_process)
tick_size = (max_tick - min_tick) / num_ticks

ax4 = Axis(
    fig[2, 2],
    xlabel = "(# of columns per process, h_elem)",
    ylabel = "Efficiency (T1/TN)",
    title = "Weak scaling efficiency (T_int = $t_int)",
    xticks = (ncols_per_process, string.(ncols_per_process)),
    xgridvisible = true,
    ygridvisible = true,
)
scatter!(ax4, ncols_per_process, weak_scaling_efficiency_clima_atmos)

save(joinpath(output_dir, job_id, "GPU_weak_scaling.png"), fig)
save(joinpath(output_dir, job_id, "GPU_weak_scaling.pdf"), fig)
