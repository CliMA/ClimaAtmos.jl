#=
julia --project=perf
using Revise; if !("--config_file" in ARGS)
    push!(ARGS, "--config_file")
    push!(ARGS, "config/default_configs/default_perf.yml")
end; include("perf/jet.jl")
=#
redirect_stderr(IOContext(stderr, :stacktrace_types_limited => Ref(false)))
if !("--config_file" in ARGS)
    push!(ARGS, "--config_file")
    push!(ARGS, "config/default_configs/default_perf.yml")
end
import Random
Random.seed!(1234)
import ClimaAtmos as CA

include("common.jl")

# config_dict = Dict("z_elem" => 63, "dt" => "10secs", "t_end" => "3600secs")
# config = AtmosCoveragePerfConfig(config_dict)
# config_file = ARGS[1]
# config_dict = YAML.load_file(config_file)
config = CA.AtmosConfig()

simulation = CA.get_simulation(config)
(; integrator) = simulation

import JET

import SciMLBase
SciMLBase.step!(integrator) # Make sure no errors
JET.@test_opt SciMLBase.step!(integrator)
