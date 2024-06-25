# When Julia 1.10+ is used interactively, stacktraces contain reduced type information to make them shorter.
# On the other hand, the full type information is printed when julia is not run interactively.
# Given that ClimaCore objects are heavily parametrized, non-abbreviated stacktraces are hard to read,
# so we force abbreviated stacktraces even in non-interactive runs.
# (See also Base.type_limited_string_from_context())
redirect_stderr(IOContext(stderr, :stacktrace_types_limited => Ref(false)))

import ClimaAtmos as CA
import YAML
<<<<<<< HEAD
import ClimaCalibrate:
    set_up_forward_model, run_forward_model, path_to_ensemble_member

"""
    set_up_forward_model(member, iteration, experiment_dir::AbstractString)

Returns an AtmosConfig object for the given member and iteration.
If given an experiment id string, it will load the config from the corresponding YAML file.
Turns off default diagnostics and sets the TOML parameter file to the member's path.
This assumes that the config dictionary has `output_dir` and `restart_file` keys.
=======
import ClimaComms
@static pkgversion(ClimaComms) >= v"0.6" && ClimaComms.@import_required_backends
using ClimaUtilities.ClimaArtifacts
import ClimaCalibrate:
    set_up_forward_model,
    run_forward_model,
    path_to_ensemble_member,
    ExperimentConfig

"""
    set_up_forward_model(member, iteration, experiment_dir::AbstractString)
    set_up_forward_model(member, iteration, ::ExperimentConfig; experiment_dir)
    set_up_forward_model(member, iteration, config_dict::AbstractDict)

Return an AtmosConfig object for the given member and iteration.

Turns off default diagnostics and sets the TOML parameter file to the member's path.
This assumes that the  config dictionary has an `output_dir` key.
>>>>>>> 3199df12 (Add calibration framework, perfectmodel experiment)
"""
function set_up_forward_model(
    member,
    iteration,
<<<<<<< HEAD
    experiment_dir::AbstractString,
)
    config_dict = YAML.load_file(joinpath(experiment_dir, "model_config.yml"))
    if haskey(config_dict, "restart_file")
        config_dict["restart_file"] =
            joinpath(experiment_dir, config_dict["restart_file"])
    end
=======
    ::ExperimentConfig;
    experiment_dir = dirname(Base.active_project()),
)
    # Assume experiment_dir is project dir
    config_dict = YAML.load_file(joinpath(experiment_dir, "model_config.yml"))
    set_up_forward_model(member, iteration, config_dict::AbstractDict)
end

function set_up_forward_model(member, iteration, experiment_dir::AbstractString)
    config_dict = YAML.load_file(joinpath(experiment_dir, "model_config.yml"))
    set_up_forward_model(member, iteration, config_dict::AbstractDict)
end

function set_up_forward_model(member, iteration, config_dict::AbstractDict)
>>>>>>> 3199df12 (Add calibration framework, perfectmodel experiment)
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
<<<<<<< HEAD

=======
>>>>>>> 3199df12 (Add calibration framework, perfectmodel experiment)
    return CA.AtmosConfig(config_dict)
end

"""
    run_forward_model(atmos_config::CA.AtmosConfig)

<<<<<<< HEAD
Runs the atmosphere model with the given an AtmosConfig object.
=======
Run the atmosphere model with the given an AtmosConfig object.
>>>>>>> 3199df12 (Add calibration framework, perfectmodel experiment)
Currently only has basic error handling.
"""
function run_forward_model(atmos_config::CA.AtmosConfig)
    simulation = CA.get_simulation(atmos_config)
    sol_res = CA.solve_atmos!(simulation)
    if sol_res.ret_code == :simulation_crashed
<<<<<<< HEAD
<<<<<<< HEAD
        !isnothing(sol_res.sol) && sol_res.sol .= eltype(sol_res.sol)(NaN)
=======
        !isnothing(sol_res.sol) && sol_res.sol .= NaN
>>>>>>> 3199df12 (Add calibration framework, perfectmodel experiment)
=======
>>>>>>> 39f292de (Remove conditional in calibration model interface)
        error(
            "The ClimaAtmos simulation has crashed. See the stack trace for details.",
        )
    end
<<<<<<< HEAD
end
=======
end
>>>>>>> 3199df12 (Add calibration framework, perfectmodel experiment)
