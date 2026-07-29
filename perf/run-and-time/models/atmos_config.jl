#=
This file is intended to be included by all small components models.

It is supposed to be a 'single source of truth' with regards to the base
ClimaAtmos configuration we  are using.
=#

import ClimaAtmos as CA


function get_base_atmos_config()
    config_path = "config"
    config_file = [
        joinpath(pkgdir(CA), config_path, "common_configs", "numerics_sphere_he16ze63.yml"),
        joinpath(
            pkgdir(CA),
            config_path,
            "longrun_configs",
            "amip_target.yml",
        ),
    ]
    config = CA.AtmosConfig(config_file; job_id = nothing)
    return config
end
