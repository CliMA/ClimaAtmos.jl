import ClimaTimeSteppers as CTS
import Base.Sys: maxrss

"""
    terminate!(integrator::CTS.TimeStepperIntegrator)

Stop time-marching gracefully by emptying `integrator.tstops`.

Returns `nothing`. Called by the graceful-exit callback.
"""
function terminate!(integrator::CTS.TimeStepperIntegrator)
    @info "Gracefully exiting simulation."
    empty!(integrator.tstops)
end

"""
    EfficiencyStats

Timing record for a completed solve, used to report throughput.

# Fields

  - `tspan`: Simulated time span `(t_start, t_end)` [s].
  - `walltime`: Wall-clock duration of the solve [s].
"""
struct EfficiencyStats{TS <: Tuple, WT}
    tspan::TS
    walltime::WT
end

"""
    simulated_years_per_day(es::EfficiencyStats)
    simulated_years(es::EfficiencyStats)
    walltime_in_days(es::EfficiencyStats)

Report model throughput from an `EfficiencyStats` record: simulated years per wall-clock
day (SYPD), simulated years [yr], and wall-clock time [d]. A year is taken to be 365
days.
"""
simulated_years_per_day(es::EfficiencyStats) =
    simulated_years(es) / walltime_in_days(es)

simulated_years(es::EfficiencyStats) =
    Float64(es.tspan[2] - es.tspan[1]) * (1 / (365 * 24 * 3600)) #=seconds * years per second=#
walltime_in_days(es::EfficiencyStats) = es.walltime * (1 / (24 * 3600)) #=seconds * days per second=#

"""
    timed_solve!(integrator)

Run the integrator to completion, timing it on the compute device, and log the walltime,
the throughput in simulated years per day, and the walltime per timestep.

# Returns

`(sol, walltime)`: the solution object and the elapsed wall-clock time [s].
"""
function timed_solve!(integrator)
    device = ClimaComms.device(integrator.u.c)
    comms_ctx = ClimaComms.context(device)
    local sol
    walltime = ClimaComms.elapsed(device) do
        sol = CTS.solve!(integrator)
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

"""
    AtmosSolveResults

Outcome of `solve_atmos!`.

# Fields

  - `sol`: Solution object, or `nothing` if the simulation crashed.
  - `ret_code`: `:success` or `:simulation_crashed`.
  - `walltime`: Wall-clock duration of the solve [s], or `nothing` if it crashed.
"""
struct AtmosSolveResults{S, RT, WT}
    sol::S
    ret_code::RT
    walltime::WT
end

"""
    solve_atmos!(simulation)

Run `simulation` to its end time and return an `AtmosSolveResults` with the solution,
the return code (`:success` or `:simulation_crashed`), and the walltime [s].

The first step is taken outside the timed solve so that compilation is not counted, and
the callbacks are precompiled. Failures are caught rather than rethrown, so that partial
results can still be inspected: in a serial run the crashed state is written to the
output directory first. The diagnostic writers are closed on every path.

# Examples

```julia
import ClimaAtmos as CA
simulation = CA.AtmosSimulation(CA.AtmosModel(CA.SphereGrid(Float64)); t_end = 86400)
results = CA.solve_atmos!(simulation)
results.ret_code == :success
```
"""
function solve_atmos!(simulation)
    (; integrator, output_writers) = simulation
    (; tspan) = integrator.sol.prob
    @info "Running" job_id = simulation.job_id output_dir =
        simulation.output_dir tspan
    comms_ctx = ClimaComms.context(axes(integrator.u.c))
    CTS.step!(integrator)
    precompile_callbacks(integrator)
    GC.gc()
    try
        if is_distributed(comms_ctx)
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
        if !is_distributed(comms_ctx)
            # We can only save when not distributed because we don't have a way to sync the
            # MPI processes (maybe just one MPI rank crashes, leading to a hanging
            # simulation)
            save_state_to_disk_func(integrator, simulation.output_dir)
        end
        @error "ClimaAtmos simulation crashed. Stacktrace for failed simulation" exception =
            (ret_code, catch_backtrace())
        return AtmosSolveResults(nothing, :simulation_crashed, nothing)
    finally
        # Close all the files opened by the writers

        maxrss_str = prettymemory(maxrss())
        @info "Memory currently used (after solve!) by the process (RSS): $maxrss_str"

        isnothing(output_writers) || foreach(close, output_writers)
    end
end

"""
    call_all_callbacks!(integrator)

Invoke every ClimaAtmos callback attached to `integrator`, in order. Returns `nothing`.
Used to precompile the callbacks before the timed solve.
"""
function call_all_callbacks!(integrator)
    for cb! in atmos_callbacks(integrator.callback)
        cb!(integrator)
    end
    return nothing
end

"""
    precompile_callbacks(integrator)

Precompile `call_all_callbacks!` for this integrator type, so that callback compilation
does not pollute the timing of the first steps. Returns `nothing`.
"""
function precompile_callbacks(integrator)
    B = Base.precompile(call_all_callbacks!, (typeof(integrator),))
    @assert B
    return nothing
end

check_conservation(simulation::AtmosSimulation) =
    check_conservation(simulation.integrator.sol)
check_conservation(integrator::CTS.TimeStepperIntegrator) =
    check_conservation(integrator.sol)
check_conservation(atmos_sol::AtmosSolveResults) =
    check_conservation(atmos_sol.sol)

"""
    check_conservation(sol)
    check_conservation(simulation)
    check_conservation(integrator)

Measure how well total energy, mass, and water are conserved between the first and last
saved states.

Only meaningful when the run saved both endpoints, and only exact for setups whose
boundary fluxes are all accounted for below.

# Returns

A `NamedTuple` of dimensionless relative errors:

  - `energy_conservation`: |Δ(atmosphere energy) + Δ(surface energy) − net radiative
    input at the top| divided by the initial total energy. Surface energy change is the
    slab ocean heat content change when a slab ocean is used, and the accumulated net
    surface energy flux otherwise.
  - `mass_conservation`: change in total dry-plus-moist mass, including water taken from
    or given to a slab ocean surface, divided by the initial mass.
  - `water_conservation`: |Δ(atmospheric water) + Δ(surface water)| divided by the final
    total atmospheric water. Zero for dry runs.
"""
function check_conservation(sol)
    # energy
    energy_total = sum(sol.u[1].c.ρe_tot)
    energy_atmos_change = sum(sol.u[end].c.ρe_tot) - sum(sol.u[1].c.ρe_tot)
    p = sol.prob.p
    sfc = p.atmos.surface.temperature
    if sfc isa SurfaceConditions.SlabOceanTemperature
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
        if sfc isa SurfaceConditions.SlabOceanTemperature
            water_surface_change = horizontal_integral_at_boundary(
                sol.u[end].sfc.water .- sol.u[1].sfc.water,
            )
        end
    end

    mass_conservation =
        (sum(sol.u[end].c.ρ) - sum(sol.u[1].c.ρ) + water_surface_change) /
        sum(sol.u[1].c.ρ)

    # We set water_conservation to zero for the dry model as there is no water
    FT = Spaces.undertype(axes(sol.u[end].c.ρ))
    water_conservation = zero(FT)
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
    write_diagnostics_as_txt(simulation::AtmosSimulation)
    write_diagnostics_as_txt(writer::ClimaDiagnostics.Writers.DictWriter, output_dir)

Write the diagnostics held in memory by a `DictWriter` to one text file per diagnostic
in `output_dir`, each line a time and a value.

The simulation method applies this to every `DictWriter` among the simulation's writers.
Only diagnostics that are 1-element vectors are supported; this exists because scalars
cannot yet be written to NetCDF (see
https://github.com/CliMA/ClimaDiagnostics.jl/issues/100).
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
