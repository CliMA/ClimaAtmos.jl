import ClimaAtmos as CA
import YAML
import ClimaCalibrate:
    set_up_forward_model, run_forward_model, path_to_ensemble_member

"""
    set_up_forward_model(member, iteration, experiment_dir::AbstractString)

Returns an AtmosConfig object for the given member and iteration.
If given an experiment id string, it will load the config from the corresponding YAML file.
Turns off default diagnostics and sets the TOML parameter file to the member's path.
This assumes that the config dictionary has `output_dir` and `restart_file` keys.
"""
function set_up_forward_model(
    member,
    iteration,
    experiment_dir::AbstractString,
)
    config_dict = YAML.load_file(joinpath(experiment_dir, "model_config.yml"))
    if haskey(config_dict, "restart_file")
        config_dict["restart_file"] =
            joinpath(experiment_dir, config_dict["restart_file"])
    end
    output_dir = config_dict["output_dir"]
    member_path = path_to_ensemble_member(output_dir, iteration, member)
    config_dict["output_dir"] = member_path
    parameter_path = joinpath(member_path, "parameters.toml")
    if haskey(config_dict, "toml")
        push!(config_dict["toml"], parameter_path)
    else
        config_dict["toml"] = [parameter_path]
    end

    # Turn off default diagnostics
    config_dict["output_default_diagnostics"] = false

    return CA.AtmosConfig(config_dict)
end

"""
    run_forward_model(atmos_config::CA.AtmosConfig)

Runs the atmosphere model with the given an AtmosConfig object.
Currently only has basic error handling.
"""
function run_forward_model(atmos_config::CA.AtmosConfig)
    simulation = CA.get_simulation(atmos_config)
    sol_res = CA.solve_atmos!(simulation)
    if sol_res.ret_code == :simulation_crashed
        !isnothing(sol_res.sol) && sol_res.sol .= eltype(sol_res.sol)(NaN)
        error(
            "The ClimaAtmos simulation has crashed. See the stack trace for details.",
        )
    end
end