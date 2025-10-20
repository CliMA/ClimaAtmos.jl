#=
Run this script with, for example:
```
julia --project=.buildkite perf/benchmark_step.jl --h_elem 6
```
Or, interactively,
```
julia --project=.buildkite
push!(ARGS, "--h_elem", "6")
# push!(ARGS, "--device", "CPUSingleThreaded") # uncomment to run on CPU
include(joinpath("perf", "benchmark_step.jl"));
```
=#
redirect_stderr(IOContext(stderr, :stacktrace_types_limited => Ref(false)))
import ClimaComms
ClimaComms.@import_required_backends
import Random
Random.seed!(1234)
import ClimaAtmos as CA
import ClimaComms
import CUDA

include("common.jl")
(; config_file, job_id) = CA.commandline_kwargs()
config = CA.AtmosConfig(config_file; job_id)

# Robustly parse boolean-like environment variables
function getenv_bool(var::AbstractString; default::Bool = false)
    raw = get(ENV, var, nothing)
    raw === nothing && return default
    s = lowercase(strip(String(raw)))
    if s in ("1", "true", "t", "yes", "y", "on")
        return true
    elseif s in ("0", "false", "f", "no", "n", "off")
        return false
    else
        # fall back to parse as integer (non-zero -> true)
        try
            return parse(Int, s) != 0
        catch
            @warn "Unrecognized boolean env var value; using default" var = var val = raw default =
                default
            return default
        end
    end
end

# Handle stacktrace-based kernel naming before compiling
if getenv_bool("CLIMA_NAME_CUDA_KERNELS_FROM_STACK_TRACE", default = false)
    import ClimaCore
    ClimaCore.DebugOnly.name_kernels_from_stack_trace() = true
end

simulation = CA.get_simulation(config)
(; integrator) = simulation;
Y₀ = deepcopy(integrator.u);
@info "Compiling benchmark_step!..."
CA.benchmark_step!(integrator, Y₀); # compile first

@info "Running benchmark_step!..."
comms_ctx = ClimaComms.context(integrator.u.c)
device = ClimaComms.device(comms_ctx)

# If we're running on CUDA, use CUDA's profiler
if device isa ClimaComms.CUDADevice
    e = 0.0
    n_steps = 5
    use_external_profiler = CUDA.Profile.detect_cupti()
    if use_external_profiler
        @info "Using external CUDA profiler"
        CUDA.@profile external = true begin
            e = CUDA.@elapsed begin
                CA.benchmark_step!(integrator, Y₀, n_steps)
            end
        end
    else
        @info "Using internal CUDA profiler"
        res = CUDA.@profile external = false begin
            e = CUDA.@elapsed begin
                CA.benchmark_step!(integrator, Y₀, n_steps)
            end
        end
        println(res)
    end
    @info "Ran step! with CUDA $n_steps times in $e s, ($(CA.prettytime(e/n_steps*1e9)) per step)"
else
    # Profile with Julia's built-in profiler
    n_steps = 10
    local e
    s = CA.@timed_str begin
        e = ClimaComms.elapsed(device) do
            CA.benchmark_step!(integrator, Y₀, n_steps) # run
        end
    end
    @info "Ran step! $n_steps times in $s, ($(CA.prettytime(e/n_steps*1e9)) per step)"
end
