# When Julia 1.10+ is used interactively, stacktraces contain reduced type information to make them shorter.
# On the other hand, the full type information is printed when julia is not run interactively.
# Given that ClimaCore objects are heavily parametrized, non-abbreviated stacktraces are hard to read,
# so we force abbreviated stacktraces even in non-interactive runs.
# (See also Base.type_limited_string_from_context())
redirect_stderr(IOContext(stderr, :stacktrace_types_limited => Ref(false)))

import ClimaAtmos as CA
import YAML
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
"""
function set_up_forward_model(
    member,
    iteration,
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

Run the atmosphere model with the given an AtmosConfig object.
Currently only has basic error handling.
"""
function run_forward_model(atmos_config::CA.AtmosConfig)
    simulation = CA.get_simulation(atmos_config)
    sol_res = CA.solve_atmos!(simulation)
    if sol_res.ret_code == :simulation_crashed
        error(
            "The ClimaAtmos simulation has crashed. See the stack trace for details.",
        )
    end
end
