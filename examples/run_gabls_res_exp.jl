using YAML
using Revise
import ClimaAtmos as CA
import ClimaAnalysis: window

includet("../post_processing/ci_plots.jl")

config_dict = YAML.load_file("prognostic_edmfx_gabls_column.yml")

for resolution in [10, 20, 30, 40, 60, 80, 100, 120, 150, 200, 300, 500]
    
    config_dict["output_dir"] = "output/gabls_test3/resolution_$resolution"
    # change cfsite number for each site for correct forcing
    config_dict["z_elem"] = resolution
    config_dict["netcdf_interpolation_num_points"] = [2, 2, resolution]
    #config_dict["external_forcing_type"] = "deep"

    config = CA.AtmosConfig(config_dict; job_id = "gabls_test")
    simulation = CA.get_simulation(config)
    sol_res = CA.solve_atmos!(simulation)

    make_plots(Val{:gabls_test}(), "output/gabls_test3/resolution_$resolution/output_active")
end