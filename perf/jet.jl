import Random
Random.seed!(1234)
import ClimaAtmos as CA
config = CA.AtmosCoveragePerfConfig(;
    config_dict = CA.config_from_target_job("edmfx_adv_test_box"),
)
integrator = CA.get_integrator(config)

import JET

import SciMLBase
SciMLBase.step!(integrator) # Make sure no errors
JET.@test_opt SciMLBase.step!(integrator)
