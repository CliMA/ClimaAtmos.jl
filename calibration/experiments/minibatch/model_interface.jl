import ClimaAtmos as CA
import YAML
import ClimaComms
@static pkgversion(ClimaComms) >= v"0.6" && ClimaComms.@import_required_backends
using ClimaUtilities.ClimaArtifacts
import ClimaCalibrate:
    set_up_forward_model, run_forward_model, path_to_ensemble_member

get_forcing_file(i)
# Varying: cfsite, month, forcing model type
# "/groups/esm/zhaoyi/GCMForcedLES/cfsite/07/HadGEM2-A/amip/Output.cfsite23_HadGEM2-A_amip_2004-2008.07.4x/stats/Stats.cfsite23_HadGEM2-A_amip_2004-2008.07.nc"

const config_dict = YAML.load_file(joinpath(@__DIR__, "experiment_config.yml"))
const output_dir = config_dict["output_dir"]

"""
    set_up_forward_model(member, iteration, experiment_dir::AbstractString)
    set_up_forward_model(member, iteration, config_dict::AbstractDict)

Return an AtmosConfig object for the given member and iteration.

Turns off default diagnostics and sets the TOML parameter file to the member's path.
This assumes that the  config dictionary has an `output_dir` key.
"""
function set_up_forward_model(member, iteration, experiment_dir::AbstractString)
    config_dict = YAML.load_file(joinpath(experiment_dir, "model_config.yml"))
    iter_path = CAL.path_to_iteration(output_dir, iteration)
    eki = JLD2.load_object(joinpath(iter_path, "eki_file.jld2"))
    member_path = path_to_ensemble_member(output_dir, iteration, member)
    config_dict["output_dir"] = member_path
    parameter_path = joinpath(member_path, "parameters.toml")
    if haskey(config_dict, "toml")
        push!(config_dict["toml"], parameter_path)
    else
        config_dict["toml"] = [parameter_path]
    end
    config_dict["output_default_diagnostics"] = false
    atmos_configs = map(EKP.get_current_minibatch(eki)) do i
        config = deepcopy(config_dict)
        config["external_forcing_file"] = get_forcing_file(i)
        config["output_dir"] = config["output_dir"] * "_$i"
        CA.AtmosConfig.(config)
    end
    return atmos_configs
end

"""
    run_forward_model(atmos_config::CA.AtmosConfig)

Run the atmosphere model with the given an AtmosConfig object.
Currently only has basic error handling.
"""
function run_forward_model(atmos_configs::Vector{CA.AtmosConfig})
    for atmos_config in atmos_configs
        simulation = CA.get_simulation(atmos_config)
        sol_res = CA.solve_atmos!(simulation)
        if sol_res.ret_code == :simulation_crashed
            !isnothing(sol_res.sol) && sol_res.sol .= eltype(sol_res.sol)(NaN)
            error(
                "The ClimaAtmos simulation has crashed. See the stack trace for details.",
            )
        end
    end
end