"""
    A script to automate each ClimaAtmos run in the interactive sessions.
"""

import ClimaAtmos as CA

LWP_N_config_dir = "LWP_N_config"
LWP_N_config_files = readdir(LWP_N_config_dir)


for config_yaml in LWP_N_config_files
    config = CA.AtmosConfig(joinpath(LWP_N_config_dir, config_yaml))
    include("./examples/hybrid/driver.jl")
end