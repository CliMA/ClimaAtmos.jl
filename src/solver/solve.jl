import ClimaTimeSteppers as CTS
import Base.Sys: maxrss

# Empty integrator.tstops to stop time-marching.
function terminate!(integrator::CTS.DistributedODEIntegrator)
    @info "Gracefully exiting simulation."
    empty!(integrator.tstops.valtree)
end

struct EfficiencyStats{TS <: Tuple, WT}
    tspan::TS
    walltime::WT
end

simulated_years_per_day(es::EfficiencyStats) =
    simulated_years(es) / walltime_in_days(es)

simulated_years(es::EfficiencyStats) =
    Float64(es.tspan[2] - es.tspan[1]) * (1 / (365 * 24 * 3600)) #=seconds * years per second=#
walltime_in_days(es::EfficiencyStats) = es.walltime * (1 / (24 * 3600)) #=seconds * days per second=#

function timed_solve!(integrator)
    device = ClimaComms.device(integrator.u.c)
    comms_ctx = ClimaComms.context(device)
    local sol
    walltime = ClimaComms.elapsed(device) do
        sol = SciMLBase.solve!(integrator)
    end
    @info "solve! walltime = $(round(walltime, digits = 3))"
    (; tspan) = integrator.sol.prob
    es = EfficiencyStats(tspan, walltime)
    _sypd = simulated_years_per_day(es)
    _sypd_str = string(round(_sypd; digits = 3))
    sypd = _sypd_str * if _sypd < 0.01
        sdpd = round(_sypd * 365, digits = 3)
        " (sdpd = $sdpd)"
    else
        ""
    end
    @info "sypd: $sypd"
    n_steps = (tspan[2] - tspan[1]) / integrator.dt
    wall_time_per_timestep = time_and_units_str(walltime / n_steps)
    @info "wall_time_per_timestep: $wall_time_per_timestep"
    return (sol, walltime)
end

struct AtmosSolveResults{S, RT, WT}
    sol::S
    ret_code::RT
    walltime::WT
end

"""
    solve_atmos!(integrator)

Call the ClimaTimeSteppers solve on the integrator.
Returns a `AtmosSolveResults`, containing the
solution, walltime, and a return code `Symbol`
indicating one of:

 - `:success`
 - `:success`
 - `:simulation_crashed`

`try-catch` is used to allow plotting
results for simulations that have crashed.
"""
function solve_atmos!(simulation)
    (; integrator, output_writers) = simulation
    (; tspan) = integrator.sol.prob
    @info "Running" job_id = simulation.job_id output_dir =
        simulation.output_dir tspan
    comms_ctx = ClimaComms.context(axes(integrator.u.c))
    SciMLBase.step!(integrator)
    precompile_callbacks(integrator)
    GC.gc()
    try
        if CA.is_distributed(comms_ctx)
            # GC.enable(false) # disabling GC causes a memory leak
            ClimaComms.barrier(comms_ctx)
            (sol, walltime) = timed_solve!(integrator)
            ClimaComms.barrier(comms_ctx)
            GC.enable(true)
            return AtmosSolveResults(sol, :success, walltime)
        else
            (sol, walltime) = timed_solve!(integrator)
            return AtmosSolveResults(sol, :success, walltime)
        end
    catch ret_code
        exc_msg = ret_code isa Exception ? "$(typeof(ret_code)): $(sprint(showerror, ret_code))" : "$ret_code"
        # Single line to stderr so the crash reason is easy to find in long logs
        println(stderr, "ClimaAtmos CRASH: ", exc_msg)
        bt = catch_backtrace()
        # Log only the first 20 stack frames so the trace is readable
        max_frames = 20
        short_bt = length(bt) > max_frames ? bt[1:max_frames] : bt
        @error "ClimaAtmos simulation crashed (stacktrace truncated to $max_frames frames)" exception =
            (ret_code, short_bt)
        if !CA.is_distributed(comms_ctx)
            # We can only save when not distributed because we don't have a way to sync the
            # MPI processes (maybe just one MPI rank crashes, leading to a hanging
            # simulation)
            CA.save_state_to_disk_func(integrator, simulation.output_dir)
        end
        return AtmosSolveResults(nothing, :simulation_crashed, nothing)
    finally
        # Close all the files opened by the writers

        maxrss_str = prettymemory(maxrss())
        @info "Memory currently used (after solve!) by the process (RSS): $maxrss_str"

        isnothing(output_writers) || foreach(close, output_writers)
    end
end

"""
    benchmark_step!(integrator)

# Example

To instantiate the environment:
```
julia --project=.buildkite
```

Then, to run interactively:
```julia
import ClimaAtmos as CA
import Random
Random.seed!(1234)
(; config_file, job_id) = CA.commandline_kwargs();
config = CA.AtmosConfig(config_file; job_id);
simulation = CA.get_simulation(config);
(; integrator) = simulation
Y₀ = deepcopy(integrator.u);
CA.benchmark_step!(integrator, Y₀);
```

Alternatively, this can be run non-interactively,
with adjusted flags as:
```
julia --project=.buildkite perf/benchmark_step.jl --h_elem 6
```

See [`argparse_settings`](@ref) for the
method defining the flags, and

    https://clima.github.io/ClimaAtmos.jl/dev/cl_args/

for the flags outlined in a table.

"""
function benchmark_step!(integrator, Y₀, n_steps = 10)
    for i in 1:n_steps
        SciMLBase.step!(integrator)
        integrator.u .= Y₀ # temporary hack to simplify performance benchmark.
    end
    return nothing
end

"""
    cycle!(integrator; n_cycles = 1)

Run `step!` the least common multiple times
for all callbacks. i.e., all callbacks will
have been called at least `n_cycles` times.

`cycle!` is the true atomic unit for performance,
because, unlike `step!`, it takes all callbacks
into account.
"""
function cycle!(integrator; n_cycles = 1)
    n_steps = n_steps_per_cycle(integrator) * n_cycles
    for i in 1:n_steps
        SciMLBase.step!(integrator)
    end
    return nothing
end

function call_all_callbacks!(integrator)
    for cb! in atmos_callbacks(integrator.callback)
        cb!(integrator)
    end
    return nothing
end

"""
    precompile_atmos(integrator)

Precompiles `step!` and all callbacks
in the `integrator`.
"""
function precompile_atmos(integrator)
    B = Base.precompile(SciMLBase.step!, (typeof(integrator),))
    @assert B
    precompile_callbacks(integrator)
    return nothing
end

function precompile_callbacks(integrator)
    B = Base.precompile(call_all_callbacks!, (typeof(integrator),))
    @assert B
    return nothing
end


check_conservation(simulation::AtmosSimulation) =
    check_conservation(simulation.integrator.sol)
check_conservation(integrator::CTS.DistributedODEIntegrator) =
    check_conservation(integrator.sol)
check_conservation(atmos_sol::AtmosSolveResults) =
    check_conservation(atmos_sol.sol)

"""
    check_conservation(sol)
    check_conservation(simulation)
    check_conservation(integrator)

Return:
- `energy_conservation = energy_net / energy_total`
- `mass_conservation = (mass(t_end) - mass(t_0)) / mass(t_0)`
- `water_conservation = (water_atmos + water_surface) / water_total`
"""
function check_conservation(sol)
    # energy
    energy_total = sum(sol.u[1].c.ρe_tot)
    energy_atmos_change = sum(sol.u[end].c.ρe_tot) - sum(sol.u[1].c.ρe_tot)
    p = sol.prob.p
    sfc = p.atmos.surface_model
    if sfc isa SlabOceanSST
        sfc_cρh = sfc.ρ_ocean * sfc.cp_ocean * sfc.depth_ocean
        energy_total +=
            horizontal_integral_at_boundary(sol.u[1].sfc.T .* sfc_cρh)
        energy_surface_change =
            horizontal_integral_at_boundary(
                sol.u[end].sfc.T .- sol.u[1].sfc.T,
            ) * sfc_cρh
    else
        energy_surface_change = -p.net_energy_flux_sfc[][]
    end
    energy_radiation_input = -p.net_energy_flux_toa[][]

    energy_conservation =
        abs(
            energy_atmos_change + energy_surface_change -
            energy_radiation_input,
        ) / energy_total

    water_surface_change = zero(Spaces.undertype(axes(sol.u[end].c.ρ)))
    if :ρq_tot in propertynames(sol.u[1].c)
        water_total = sum(sol.u[end].c.ρq_tot)
        water_atmos_change = sum(sol.u[end].c.ρq_tot) - sum(sol.u[1].c.ρq_tot)
        if sfc isa SlabOceanSST
            water_surface_change = horizontal_integral_at_boundary(
                sol.u[end].sfc.water .- sol.u[1].sfc.water,
            )
        end
    end

    mass_conservation =
        (sum(sol.u[end].c.ρ) - sum(sol.u[1].c.ρ) + water_surface_change) /
        sum(sol.u[1].c.ρ)

    # We set water_conservation to zero for the dry model as there is no water
    water_conservation = zero(eltype(sol))
    if :ρq_tot in propertynames(sol.u[1].c)
        water_conservation =
            abs(water_atmos_change + water_surface_change) / water_total
    end

    return (; energy_conservation, mass_conservation, water_conservation)
end

function write_diagnostics_as_txt(simulation::AtmosSimulation)
    foreach(
        w -> write_diagnostics_as_txt(w, simulation.output_dir),
        filter(w -> w isa CAD.DictWriter, simulation.output_writers),
    )
    return nothing
end


"""
    write_diagnostics_as_txt(writer, output_dir)

Write diagnostics in DictWriter to text files. This function is
added because currently we do not support writing scalars to netcdf.
It only supports diagnostics that are 1-element vectors.
Related issue: https://github.com/CliMA/ClimaDiagnostics.jl/issues/100
"""
function write_diagnostics_as_txt(
    writer::ClimaDiagnostics.Writers.DictWriter,
    output_dir,
)
    @info "Writing diagnostics to text files"
    for diagnostic in keys(writer.dict)
        first(values(writer[diagnostic])) isa Vector ||
            "write_diagnostics_as_txt is not supported for diagnostics that are not vectors"
        filename = joinpath(output_dir, diagnostic * ".txt")
        times = collect(keys(writer[diagnostic]))
        values_all = getindex.(collect(values(writer[diagnostic])), 1)
        open(filename, "w") do io
            for (ti, vi) in zip(times, values_all)
                println(io, "$ti $vi")
            end
        end
    end
end
