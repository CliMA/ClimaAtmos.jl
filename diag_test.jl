import ClimaAtmos as CA
import Random
Random.seed!(1234)

config = CA.AtmosConfig("config/diag_config.yml")
simulation = CA.get_simulation(config)
sol_res = CA.solve_atmos!(simulation)
