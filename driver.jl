using Revise, Infiltrator
# import SciMLBase: step!

import CUDA
# CUDA.device!(0)  # Set specific GPU device
ENV["CLIMACOMMS_DEVICE"]="CUDA"

import ClimaComms
import ClimaComms.@import_required_backends

import ClimaAtmos as CA
import ClimaCore: Fields, Geometry, Operators, Spaces, Grids, Utilities

# config_file = "./config/model_configs/baroclinic_wave_equil.yml"
# simulation = CA.AtmosSimulation(config_file)
# CA.solve_atmos!(simulation)

config_file = "./config/model_configs/plane_schar_mountain_float64_test.yml"
config = CA.AtmosConfig(config_file)
simulation = CA.get_simulation(config)
(; integrator) = simulation
sol_res = CA.solve_atmos!(simulation)