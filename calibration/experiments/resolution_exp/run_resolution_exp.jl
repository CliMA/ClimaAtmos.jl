using YAML
using Revise
import ClimaAtmos as CA

config_dict = YAML.load_file("prognostic_edmf.yml")

for resolution in [20, 30, 60, 80, 100, 120, 150, 200, 300]
    for site in [30]
    
        config_dict["output_dir"] = "output/site_$site/resolution_$resolution"
        # change cfsite number for each site for correct forcing
        config_dict["cfsite_number"] = "site$site" 
        config_dict["z_elem"] = resolution
        config_dict["external_forcing_type"] = "deep"

        config = CA.AtmosConfig(config_dict)
        simulation = CA.get_simulation(config)
        sol_res = CA.solve_atmos!(simulation)
    end
end