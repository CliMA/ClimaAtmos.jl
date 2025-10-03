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
using ArgParse

NUM_WORKERS = 10


# example usage
# sbatch scm_runner.sbatch --run_output_dir=/groups/esm/cchristo/climaatmos_scm_calibrations/scm_runs/nearest_neig_particle_i6_m31_exp51 --parameter_path=./optimal_tomls/parameters_nearest_neig_particle_i6_m31_exp51.toml

function main()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--cal_output_dir"
        help = "Base calibration output directory that contains a 'configs' subfolder"
        required = true

        "--parameter_path"
        help = "Path to parameter TOML file"
        required = true

        "--run_output_dir"
        help = "Directory to output SCM simulations. If omitted, defaults to <cal_output_dir>/scm_runs/<basename(parameter_path)>"
        required = false
    end

    args = parse_args(s)

    cal_output_dir = abspath(args["cal_output_dir"]) # renamed from calib_output_dir
    parameter_path = abspath(args["parameter_path"])

    # Determine default run_output_dir if not provided
    if haskey(args, "run_output_dir") && !isnothing(args["run_output_dir"]) 
        run_output_dir = abspath(args["run_output_dir"]) 
    else
        param_base = splitext(basename(parameter_path))[1] # strip .toml
        run_output_dir = joinpath(cal_output_dir, "scm_runs", param_base)
    end

    base_config_path = joinpath(cal_output_dir, "configs", "model_config.yml")

    mkpath(run_output_dir)

    all_configs =
        generate_atmos_configs(base_config_path, parameter_path, run_output_dir, cal_output_dir)
    run_all_simluations(all_configs)
end

function generate_atmos_configs(
    base_config_path::String,
    parameter_path::String,
    run_output_dir::String,
    cal_output_dir::String,
)
    config_dict = YAML.load_file(base_config_path)
    config_dict["output_dir"] = run_output_dir
    config_dict["toml"] = [abspath(parameter_path)]
    config_dict["output_default_diagnostics"] = false

    # Load experiment config from cal_output_dir/configs
    exp_config_path = joinpath(cal_output_dir, "configs", "experiment_config.yml")
    experiment_config_dict = isfile(exp_config_path) ? YAML.load_file(exp_config_path) : Dict{String,Any}()

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

        # Add forcing-specific TOML file if specified
        forcing_type = get_cfsite_type(i, cfsite_numbers)
        if haskey(experiment_config_dict, "forcing_toml_files") && haskey(
            experiment_config_dict["forcing_toml_files"],
            forcing_type,
        )
            forcing_config_file =
                experiment_config_dict["forcing_toml_files"][forcing_type]
            # Prefer configs/scm_tomls/<file>, fallback to configs/<file>
            forcing_path_candidate_1 = joinpath(cal_output_dir, "configs", "scm_tomls", forcing_config_file)
            forcing_path_candidate_2 = joinpath(cal_output_dir, "configs", forcing_config_file)
            forcing_config_path = if isfile(forcing_path_candidate_1)
                forcing_path_candidate_1
            else
                forcing_path_candidate_2
            end

            if isfile(forcing_config_path)
                forcing_config_path = abspath(forcing_config_path)
                push!(config["toml"], forcing_config_path)
                @info "Added forcing config: $forcing_config_path for case $i (type: $forcing_type)"
            else
                @warn "Forcing config file not found for case $i (type: $forcing_type). Looked at $forcing_path_candidate_1 and $forcing_path_candidate_2."
            end
        else
            @warn "No forcing config file specified in experiment_config.yml for type '$forcing_type' for case $i."
        end

        @info "Prepared SCM config for case $i" output_dir=config["output_dir"] toml=config["toml"] forcing_type=forcing_type cfsite=config["cfsite_number"]

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
