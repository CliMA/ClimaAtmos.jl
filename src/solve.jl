import ClimaTimeSteppers as CTS
import OrdinaryDiffEq as ODE

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
            ODE.step!(integrator)
            # GC.enable(false) # disabling GC causes a memory leak
            GC.gc()
            ClimaComms.barrier(comms_ctx)
            if ClimaComms.iamroot(comms_ctx)
                @timev "solve!" begin
                    walltime = @elapsed sol = ODE.solve!(integrator)
                end
            else
                walltime = @elapsed sol = ODE.solve!(integrator)
            end
            ClimaComms.barrier(comms_ctx)
            GC.enable(true)
            return AtmosSolveResults(sol, :success, walltime)
        else
            sol = @timev "solve!" ODE.solve!(integrator)
            return AtmosSolveResults(sol, :success, nothing)
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
