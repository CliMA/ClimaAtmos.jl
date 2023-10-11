import Random
Random.seed!(1234)
import ClimaAtmos as CA
using CUDA
import ClimaComms
using Plots
using PrettyTables
import YAML

# Need to generate config_dict here to override `h_elem` in the loop below
parsed_args = CA.parse_commandline(CA.argparse_settings())
config_dict = YAML.load_file(parsed_args["config_file"])
output_dir = joinpath(config_dict["job_id"])

steptimes = []
# Iterate through varying number of horizontal elements
for h_elem in 8:8:40
    config_dict["h_elem"] = h_elem
    config = CA.AtmosConfig(config_dict)
    integrator = CA.get_integrator(config)
    Y₀ = deepcopy(integrator.u)

    @info "Compiling benchmark_step for h_elem=$h_elem"
    CA.benchmark_step!(integrator, Y₀) # compile first

    @info "Running benchmark_step for h_elem=$h_elem"
    n_steps = 10
    comms_ctx = ClimaComms.context(integrator.u.c)
    device = ClimaComms.device(comms_ctx)
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
p = Plots.plot(
    first.(steptimes),
    last.(steptimes);
    title = "Step Times v/s Horizontal Elements",
    xlabel = "h_elem",
    ylabel = "time (ms)",
    label = "step time",
    linewidth = 3,
    left_margin = 20Plots.mm,
    bottom_margin = 10Plots.mm,
)
Plots.png(p, joinpath(output_dir, "scaling.png"))
