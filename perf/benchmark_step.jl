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
import CUDA
import SciMLBase

include("common.jl")
(; config_file, job_id) = CA.commandline_kwargs()
config = CA.AtmosConfig(config_file; job_id)

simulation = CA.get_simulation(config)
(; integrator) = simulation;
# Run one step to compile
@info "Compiling step!..."
SciMLBase.step!(integrator);

@info "Running step!..."
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
                for i in 1:n_steps
                    SciMLBase.step!(integrator)
                end
            end
        end
    else
        @info "Using internal CUDA profiler"
        res = CUDA.@profile external = false begin
            e = CUDA.@elapsed begin
                for i in 1:n_steps
                    SciMLBase.step!(integrator)
                end
            end
        end
        show(IOContext(stdout, :limit => false), res)
    end
    @info "Ran step! with CUDA $n_steps times in $e s, ($(CA.prettytime(e/n_steps*1e9)) per step)"
else
    # Profile with Julia's built-in profiler
    n_steps = 10
    local e
    s = CA.@timed_str begin
        e = ClimaComms.elapsed(device) do
            for i in 1:n_steps
                SciMLBase.step!(integrator)
            end
        end
    end
    @info "Ran step! $n_steps times in $s, ($(CA.prettytime(e/n_steps*1e9)) per step)"
end
