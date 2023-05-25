import Random
Random.seed!(1234)
import ClimaAtmos as CA
config = CA.AtmosPerfConfig()
integrator = CA.get_integrator(config)

import JET

import OrdinaryDiffEq
OrdinaryDiffEq.step!(integrator) # Make sure no errors
JET.@test_opt OrdinaryDiffEq.step!(integrator)
