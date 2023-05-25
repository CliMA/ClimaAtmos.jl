# Launch with `julia --project --track-allocation=user`
import Random
Random.seed!(1234)
import ClimaAtmos as CA
config = CA.AtmosPerfConfig()
integrator = CA.get_integrator(config)

import OrdinaryDiffEq
OrdinaryDiffEq.step!(integrator) # compile first
Profile.clear()
Profile.clear_malloc_data()
OrdinaryDiffEq.step!(integrator)
