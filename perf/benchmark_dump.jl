#=
```
julia --project=examples perf/benchmark_dump.jl --output=report
```
Or, interactively,
```
julia --project=examples
push!(ARGS, "--output", "report")
include(joinpath("perf", "benchmark_dump.jl"));
```
=#
import Random
Random.seed!(1234)
import ClimaAtmos as CA
using CUDA
import ClimaComms
using Plots
using PrettyTables

s = CA.argparse_settings()
parsed_args = CA.parse_commandline(s);
output_dir = "gpu_implicit_barowave_wrt_h_elem/report"

# Set non-varying arguments
parsed_args["z_elem"] = 50
parsed_args["dt"] = "50secs"

steptimes = []

# Iterate through varying number of horizontal elements
for h_elem in 8:8:64
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

# Output a table with step times
data = hcat(first.(steptimes), last.(steptimes))
pretty_table(
    data;
    title = "Step times v/s horizontal elements",
    header = ["h_elem", "step time"],
    alignment = :l,
    crop = :none,
)

# Output a plot of step time scaling
p = Plots.plot(first.(steptimes), last.(steptimes); title="Step Times v/s Horizontal Elements", xlabel="h_elem", ylabel="time (ms)", label="step time", linewidth=3)
Plots.png(p, joinpath(output_dir, "scaling.png"))
