"""
    A script to automate each ClimaAtmos run in the interactive sessions.
"""

import ClimaAtmos as CA

# Compile once, so that next runs will be faster.
config = CA.AtmosConfig("prognostic_edmfx_dycoms_rf02_column.yml")
include("./examples/hybrid/driver.jl")

# Directory with yaml files.
LWP_N_config_dir = "LWP_N_config"
LWP_N_config_files = readdir(LWP_N_config_dir)

# Run simulation for each file in the directory.
for config_yaml in LWP_N_config_files
    config = CA.AtmosConfig(joinpath(LWP_N_config_dir, config_yaml))
    include("./examples/hybrid/driver.jl")
end