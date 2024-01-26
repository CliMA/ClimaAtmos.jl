redirect_stderr(IOContext(stderr, :stacktrace_types_limited => Ref(false)))
import Random
Random.seed!(1234)
import ClimaAtmos as CA

include("common.jl")

config_dict = Dict("z_elem" => 63, "dt" => "10secs", "t_end" => "3600secs")
config = AtmosCoveragePerfConfig(config_dict)

simulation = CA.get_simulation(config)
(; integrator) = simulation

import JET

import SciMLBase
SciMLBase.step!(integrator) # Make sure no errors
JET.@test_opt SciMLBase.step!(integrator)
