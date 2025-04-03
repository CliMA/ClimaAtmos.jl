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

# Load experiment configuration to find forcing TOML files
const experiment_config_dict =
    YAML.load_file(joinpath(@__DIR__, "..", "experiment_config.yml"))
const calib_output_dir = experiment_config_dict["output_dir"] # Base directory for calibration outputs/configs

using Distributed

using ArgParse

NUM_WORKERS = 10


# example usage
# sbatch scm_runner.sbatch --run_output_dir=/groups/esm/cchristo/climaatmos_scm_calibrations/scm_runs/nearest_neig_particle_i6_m31_exp51 --parameter_path=./optimal_tomls/parameters_nearest_neig_particle_i6_m31_exp51.toml

function main()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--run_output_dir"
        help = "Directory to output SCM simulations"
        required = true

        "--parameter_path"
        help = "Path to parameter TOML file"
        required = true
    end

    args = parse_args(s)

    base_config_path = "./model_config_prognostic_runner.yml"
    run_output_dir = args["run_output_dir"]
    parameter_path = args["parameter_path"]

    all_configs =
        generate_atmos_configs(base_config_path, parameter_path, run_output_dir)
    run_all_simluations(all_configs)
end

function generate_atmos_configs(
    base_config_path::String,
    parameter_path::String,
    run_output_dir::String,
)
    config_dict = YAML.load_file(base_config_path)
    config_dict["output_dir"] = run_output_dir
    config_dict["toml"] = [abspath(parameter_path)]
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
        config["cfsite_number"] = string("site", cfsite_number)
        config["output_dir"] = joinpath(
            run_output_dir,
            "cfsite_$(cfsite_number)_$(forcing_model)_$(experiment)_$(month)",
        )
        # config["external_forcing_type"] = get_cfsite_type(i, cfsite_numbers) # Removed: No longer used

        # Add forcing-specific TOML file if specified
        forcing_type = get_cfsite_type(i, cfsite_numbers)
        if haskey(experiment_config_dict, "forcing_toml_files") && haskey(
            experiment_config_dict["forcing_toml_files"],
            forcing_type,
        )
            forcing_config_file =
                experiment_config_dict["forcing_toml_files"][forcing_type]
            # Assume forcing configs are in a 'configs' subdir relative to the main calibration output dir
            forcing_config_path =
                joinpath(calib_output_dir, "configs", forcing_config_file)

            if isfile(forcing_config_path)
                forcing_config_path = abspath(forcing_config_path)
                # config["toml"] should already exist and contain parameter_path
                push!(config["toml"], forcing_config_path)
                @info "Added forcing config: $forcing_config_path for case $i (type: $forcing_type)"
            else
                @warn "Forcing config file not found at $forcing_config_path for case $i (type: $forcing_type)."
            end
        else
            @warn "No forcing config file specified in experiment_config.yml for type '$forcing_type' for case $i."
        end

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

main()
