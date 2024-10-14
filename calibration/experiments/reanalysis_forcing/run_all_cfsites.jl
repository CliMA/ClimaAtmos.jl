using YAML
using Revise
import ClimaAtmos as CA

config_dict = YAML.load_file("prognostic_edmfx_gcmdriven_column.yml")

# Set up the forward model
for site in 2:23
    
    config_dict["output_dir"] = "output/era5_driven/site$site"
    # change cfsite number for each site for correct forcing
    config_dict["cfsite_number"] = "site$site" 
    config = CA.AtmosConfig(config_dict)
    simulation = CA.get_simulation(config)
    sol_res = CA.solve_atmos!(simulation)
end