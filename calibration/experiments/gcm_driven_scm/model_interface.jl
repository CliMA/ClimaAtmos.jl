import ClimaAtmos as CA
import YAML
import ClimaComms
@static pkgversion(ClimaComms) >= v"0.6" && ClimaComms.@import_required_backends
using ClimaUtilities.ClimaArtifacts
import ClimaCalibrate: path_to_ensemble_member

import ClimaCalibrate as CAL
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

@everywhere function run_atmos_simulation(atmos_config)
    simulation = CA.get_simulation(atmos_config)
    sol_res = CA.solve_atmos!(simulation)
    if sol_res.ret_code == :simulation_crashed
        if !isnothing(sol_res.sol)
            T = eltype(sol_res.sol)
            if T !== Any && isconcretetype(T)
                sol_res.sol .= T(NaN)
            else
                fill!(sol_res.sol, NaN)
            end
        end
        error(
            "The ClimaAtmos simulation has crashed. See the stack trace for details.",
        )
    end
end

function CAL.forward_model(iteration, member)
    base_config_dict = YAML.load_file(joinpath(@__DIR__, model_config))

    iter_path = CAL.path_to_iteration(output_dir, iteration)
    eki = JLD2.load_object(joinpath(iter_path, "eki_file.jld2"))
    member_path = path_to_ensemble_member(output_dir, iteration, member)

    config_dict = deepcopy(base_config_dict)
    config_dict["output_dir"] = member_path
    parameter_path = joinpath(member_path, "parameters.toml")
    if haskey(config_dict, "toml")
        config_dict["toml"] = abspath.(config_dict["toml"])
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
        forcing_type = get_cfsite_type(i, cfsite_numbers)

        if haskey(experiment_config_dict, "forcing_toml_files") && haskey(
            experiment_config_dict["forcing_toml_files"],
            forcing_type,
        )
            forcing_config_file =
                experiment_config_dict["forcing_toml_files"][forcing_type]
            forcing_config_path =
                joinpath(output_dir, "configs", forcing_config_file)

            if isfile(forcing_config_path)
                forcing_config_path = abspath(forcing_config_path)
                if haskey(config, "toml")
                    config["toml"] = abspath.(config["toml"])
                    push!(config["toml"], forcing_config_path)
                    @info "Added forcing config: $forcing_config_path for case $i"
                else
                    config["toml"] = [forcing_config_path]
                end
            else
                @warn "Forcing config file not found at $forcing_config_path for case $i."
            end
        else
            @warn "No forcing config file specified for type '$forcing_type' for case $i."
        end

        comms_ctx = ClimaComms.SingletonCommsContext()
        CA.AtmosConfig(config; comms_ctx)
    end

    @info "Preparing to run $(length(atmos_configs)) model simulations in parallel."
    println("Number of workers: ", nprocs())

    start_time = time()
    map(run_atmos_simulation, atmos_configs)
    end_time = time()

    elapsed_time = (end_time - start_time) / 60.0

    @info "Finished all model simulations. Total time taken: $(elapsed_time) minutes."

end
