#=
```
julia --project=examples perf/benchmark_dump.jl
```
Or, interactively,
```
julia --project=examples
include(joinpath("perf", "benchmark_dump.jl"));
```
=#
import Random
Random.seed!(1234)
import ClimaAtmos as CA
using CUDA
import ClimaComms
using PrettyTables

parsed_args = CA.AtmosTargetParsedArgs(; target_job = "gpu_implicit_barowave");
steptimes = []

for h_elem in 6:2:12
    parsed_args["h_elem"] = h_elem
    config = CA.AtmosConfig(; parsed_args)
    integrator = CA.get_integrator(config)
    Y₀ = deepcopy(integrator.u)

    @info "Compiling benchmark_step for h_elem=$h_elem"
    CA.benchmark_step!(integrator, Y₀) # compile first

    @info "Running benchmark_step for h_elem=$h_elem"
    n_steps = 10
    device = ClimaComms.device(integrator.p.comms_ctx)
    if device isa ClimaComms.CUDADevice
        e = CUDA.@elapsed begin
            s = CA.@timed_str begin
                CA.benchmark_step!(integrator, Y₀, n_steps) # run
            end
        end
    else
        e = @elapsed begin
            s = CA.@timed_str begin
                CA.benchmark_step!(integrator, Y₀, n_steps) # run
            end
        end
    end
    @info "Ran step! $n_steps times in $s, ($(CA.prettytime(e/n_steps*1e9)) per step)"
    steptime = CA.prettytime(e / n_steps * 1e9)
    push!(steptimes, (h_elem, steptime))
end

data = hcat(first.(steptimes), last.(steptimes))

pretty_table(
    data;
    title = "Step times",
    header = ["h_elem", "step time"],
    alignment = :l,
    crop = :none,
)
