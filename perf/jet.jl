import Random
Random.seed!(1234)
import ClimaAtmos as CA

include("common.jl")

config_dict = Dict("z_elem" => 63, "dt" => "10secs", "t_end" => "3600secs")
config = AtmosCoveragePerfConfig(config_dict)

integrator = CA.get_integrator(config)

import JET

import SciMLBase
SciMLBase.step!(integrator) # Make sure no errors
JET.@test_opt SciMLBase.step!(integrator)
