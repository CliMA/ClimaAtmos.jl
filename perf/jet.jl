import Random
Random.seed!(1234)
import ClimaAtmos as CA
config = CA.AtmosConfig(;
    parsed_args = CA.AtmosTargetParsedArgs(; target_job = "edmfx_adv_test_box"),
)
integrator = CA.get_integrator(config)

import JET

import OrdinaryDiffEq
OrdinaryDiffEq.step!(integrator) # Make sure no errors
JET.@test_opt OrdinaryDiffEq.step!(integrator)
