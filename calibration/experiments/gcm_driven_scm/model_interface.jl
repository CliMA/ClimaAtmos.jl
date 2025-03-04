import ClimaAtmos as CA
import YAML
import ClimaComms
@static pkgversion(ClimaComms) >= v"0.6" && ClimaComms.@import_required_backends
using ClimaUtilities.ClimaArtifacts
import ClimaCalibrate:
    set_up_forward_model, run_forward_model, path_to_ensemble_member
import EnsembleKalmanProcesses as EKP
using JLD2

include("get_les_metadata.jl")
include("helper_funcs.jl")

using Distributed
const experiment_config_dict =
    YAML.load_file(joinpath(@__DIR__, "experiment_config.yml"))
const output_dir = experiment_config_dict["output_dir"]
const model_config = experiment_config_dict["model_config"]
const batch_size = experiment_config_dict["batch_size"]


"""
    set_up_forward_model(member, iteration, experiment_dir::AbstractString)
    set_up_forward_model(member, iteration, config_dict::AbstractDict)

Return an AtmosConfig object for the given member and iteration.

Turns off default diagnostics and sets the TOML parameter file to the member's path.
This assumes that the  config dictionary has an `output_dir` key.
"""
function set_up_forward_model(member, iteration, experiment_dir::AbstractString)
    experiment_config_dict =
        YAML.load_file(joinpath(experiment_dir, model_config))
    config_dict = YAML.load_file(joinpath(experiment_dir, model_config))
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

    ref_paths, cfsite_numbers = get_les_calibration_library()
    atmos_configs = map(EKP.get_current_minibatch(eki)) do i
        config = deepcopy(config_dict)
        config["external_forcing_file"] = get_forcing_file(i, ref_paths)
        config["cfsite_number"] = get_cfsite_id(i, cfsite_numbers)
        config["output_dir"] = joinpath(member_path, "config_$i")
        config["external_forcing_type"] = get_cfsite_type(i, cfsite_numbers)
        comms_ctx = ClimaComms.SingletonCommsContext()
        CA.AtmosConfig(config; comms_ctx)
    end

    return atmos_configs
end


addprocs(min(batch_size, 5))
@everywhere begin
    import ClimaAtmos as CA
    using JLD2
end

@everywhere function run_atmos_simulation(atmos_config)
    simulation = CA.get_simulation(atmos_config)
    sol_res = CA.solve_atmos!(simulation)
    if sol_res.ret_code == :simulation_crashed
        !isnothing(sol_res.sol) && sol_res.sol .= eltype(sol_res.sol)(NaN)
        error(
            "The ClimaAtmos simulation has crashed. See the stack trace for details.",
        )
    end
end

function run_forward_model(atmos_configs)
    @info "Preparing to run $(length(atmos_configs)) model simulations in parallel."
    println("Number of workers: ", nprocs())

    start_time = time()

    pmap(run_atmos_simulation, atmos_configs)

    end_time = time()
    elapsed_time = (end_time - start_time) / 60.0

    @info "Finished all model simulations. Total time taken: $(elapsed_time) minutes."
end
