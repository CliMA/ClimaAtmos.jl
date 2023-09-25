import Random
Random.seed!(1234)
import ClimaAtmos as CA
include("common.jl")
config = AtmosTargetConfig(; target_job = "edmfx_adv_test_box")
integrator = CA.get_integrator(config)

import JET

import SciMLBase
SciMLBase.step!(integrator) # Make sure no errors
JET.@test_opt SciMLBase.step!(integrator)
