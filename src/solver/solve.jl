import ClimaTimeSteppers as CTS
import OrdinaryDiffEq as ODE

struct EfficiencyStats{TS <: Tuple, WT}
    tspan::TS
    walltime::WT
end

simulated_years_per_day(es::EfficiencyStats) =
    simulated_years(es) / walltime_in_days(es)

simulated_years(es::EfficiencyStats) =
    (es.tspan[2] - es.tspan[1]) * (1 / (365 * 24 * 3600)) #=seconds * years per second=#
walltime_in_days(es::EfficiencyStats) = es.walltime * (1 / (24 * 3600)) #=seconds * days per second=#

function timed_solve!(integrator)
    # `step!(integrator)` may have been called
    # prior to this function in order to remove
    # compile time effects. So, the `solve!`
    # above may have solved from (Δt, tspan[2]).
    # Compute efficiency taking this into account:
    (; tspan) = integrator.sol.prob
    _tspan = (integrator.t, tspan[2])
    walltime = @elapsed begin
        s = @timed_str begin
            sol = ODE.solve!(integrator)
        end
    end
    @info "solve!: $s"
    @assert 0 ≤ _tspan[1] ≤ _tspan[2]
    es = EfficiencyStats(_tspan, walltime)
    @info "sypd: $(simulated_years_per_day(es))"
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
function solve_atmos!(integrator)
    (; p) = integrator
    (; tspan) = integrator.sol.prob
    @info "Running" job_id = p.simulation.job_id output_dir =
        p.simulation.output_dir tspan
    comms_ctx = ClimaComms.context(axes(integrator.u.c))
    try
        if CA.is_distributed(comms_ctx)
            # TODO: should we also trigger callbacks?
            ODE.step!(integrator)
            # GC.enable(false) # disabling GC causes a memory leak
            GC.gc()
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
        @error "ClimaAtmos simulation crashed. Stacktrace for failed simulation" exception =
            (ret_code, catch_backtrace())
        return AtmosSolveResults(nothing, :simulation_crashed, nothing)
    end
end

"""
    benchmark_step!(integrator)

# Example

To instantiate the environment:
```
julia --project=examples
```

Then, to run interactively:
```julia
import ClimaAtmos as CA
import Random
Random.seed!(1234)
config = CA.AtmosPerfConfig();
integrator = CA.get_integrator(config);
Y₀ = deepcopy(integrator.u);
CA.benchmark_step!(integrator, Y₀);
```

Alternatively, this can be run non-interactively,
with adjusted flags as:
```
julia --project=examples perf/benchmark_step.jl --h_elem 6
```

See [`argparse_settings`](@ref) for the
method defining the flags, and

    https://clima.github.io/ClimaAtmos.jl/dev/cl_args/

for the flags outlined in a table.

"""
function benchmark_step!(integrator, Y₀)
    ODE.step!(integrator)
    integrator.u .= Y₀ # temporary hack to simplify performance benchmark.
    return nothing
end
