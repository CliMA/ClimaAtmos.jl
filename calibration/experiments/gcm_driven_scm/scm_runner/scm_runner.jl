import ClimaCalibrate as CAL
import ClimaAtmos as CA
import EnsembleKalmanProcesses as EKP
import YAML
import TOML
using Distributions
import EnsembleKalmanProcesses as EKP
using JLD2

import ClimaComms
@static pkgversion(ClimaComms) >= v"0.6" && ClimaComms.@import_required_backends

import JLD2
using LinearAlgebra

include("../helper_funcs.jl")
include("runner_helper.jl")

using Distributed


NUM_WORKERS = 14
rel_path = "/groups/esm/cchristo/climaatmos_scm_calibrations/scm_runs"
run_output_dir = joinpath(rel_path, "exp1")
base_config_path = "model_config_prognostic_runner.yml"
parameter_path = "parameters_nearest_neig_particle_i9_m3_precal_exp2.toml"


"""
base_config_path - path to the base ClimaAtmos config file
run_output_dir - dir to output scm sims 

"""
function generate_atmos_configs(base_config_path::String, parameter_path::String, run_output_dir::String)

    config_dict = YAML.load_file(base_config_path)
    # set output
    config_dict["output_dir"] = run_output_dir
    config_dict["toml"] = [parameter_path]
    config_dict["output_default_diagnostics"] = false

    ref_paths, cfsite_numbers = get_les_calibration_library_runner()
    num_cases = length(ref_paths)
    atmos_configs = map(collect(1:num_cases)) do i
        config = deepcopy(config_dict)

        cfsite_info = get_cfsite_info_from_path(ref_paths[i])
        forcing_model = cfsite_info["forcing_model"]
        experiment = cfsite_info["experiment"]
        month = cfsite_info["month"]
        cfsite_number = cfsite_info["cfsite_number"]

        config["external_forcing_file"] = get_forcing_file(i, ref_paths)
        # config["cfsite_number"] = get_cfsite_id(i, cfsite_numbers)
        config["cfsite_number"] = string("site", cfsite_number)

        config["output_dir"] = joinpath(run_output_dir, "cfsite_$(cfsite_number)_$(forcing_model)_$(experiment)_$(month)")
        config["external_forcing_type"] = get_cfsite_type(i, cfsite_numbers)
        comms_ctx = ClimaComms.SingletonCommsContext()
        CA.AtmosConfig(config; comms_ctx)
    end

    return atmos_configs
end



addprocs(NUM_WORKERS)
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

function run_all_simluations(atmos_configs)
    @info "Preparing to run $(length(atmos_configs)) model simulations in parallel."
    println("Number of workers: ", nprocs())

    start_time = time()

    pmap(run_atmos_simulation, atmos_configs)

    end_time = time()
    elapsed_time = (end_time - start_time) / 60.0

    @info "Finished all model simulations. Total time taken: $(elapsed_time) minutes."
end





all_configs = generate_atmos_configs(base_config_path, parameter_path, run_output_dir)
run_all_simluations(all_configs)