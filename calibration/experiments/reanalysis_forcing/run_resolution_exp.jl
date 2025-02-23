using YAML
using Revise
import ClimaAtmos as CA

config_dict = YAML.load_file("prognostic_edmfx_gcmdriven_column.yml")

for resolution in [200, 300, 500]#[20, 30, 60, 80, 100, 120, 150, 200, 300, 500]
    
    config_dict["output_dir"] = "output/resolution_exp/resolution_$resolution"
    # change cfsite number for each site for correct forcing
    config_dict["cfsite_number"] = "site13" 
    config_dict["z_elem"] = resolution
    config_dict["dt_rad"] = "10mins"
    if resolution == 200
        config_dict["dt"] = "5secs"
    else
        config_dict["dt"] = "2secs"
    end
    config = CA.AtmosConfig(config_dict)
    simulation = CA.get_simulation(config)
    sol_res = CA.solve_atmos!(simulation)
end