"""
Helper functions for parameter sensitivity analysis.
"""

import ClimaAtmos as CA
import YAML
import TOML
using Printf
using Dates

import ClimaComms
@static pkgversion(ClimaComms) >= v"0.6" && ClimaComms.@import_required_backends

"""
Run a single sensitivity configuration (all cases for one parameter setting).
This function is designed to be called via pmap on workers.
"""
function run_single_sensitivity_config(sim_config::Dict)
    config_name = sim_config["name"]
    output_dir = sim_config["output_subdir"]
    toml_path = sim_config["toml_path"]
    model_config_path = sim_config["model_config_path"]
    cases = sim_config["cases"]
    experiment_config = get(sim_config, "experiment_config", Dict())

    @info "Running sensitivity config: $config_name"

    mkpath(output_dir)

    # Track results for each case
    case_results = Dict{String,Any}[]
    start_time = time()

    for (i, case) in enumerate(cases)
        case_result = run_single_case(
            case,
            model_config_path,
            toml_path,
            output_dir,
            i;
            experiment_config = experiment_config,
        )
        push!(case_results, case_result)
    end

    elapsed = time() - start_time

    # Handle Nothing values for TOML serialization
    param_value = sim_config["param_value"]
    param_value_toml = isnothing(param_value) ? "baseline" : param_value

    result = Dict(
        "name" => config_name,
        "param_name" => sim_config["param_name"],
        "param_value" => param_value_toml,
        "status" => all(r["status"] == "success" for r in case_results) ? "success" : "failed",
        "elapsed_sec" => elapsed,
        "num_cases" => length(cases),
        "cases" => case_results,
    )

    # Save individual config results
    results_file = joinpath(output_dir, "run_results.toml")
    open(results_file, "w") do io
        TOML.print(io, result)
    end

    return result
end

"""
Run simulation for a single case (cfsite + month combination).
"""
function run_single_case(
    case::Dict,
    model_config_path::String,
    toml_path::String,
    output_dir::String,
    case_index::Int;
    experiment_config::Dict = Dict(),
)
    cfsite_number = case["cfsite_number"]
    month = case["month"]
    forcing_model = case["forcing_model"]
    experiment = case["experiment"]
    forcing_file = case["forcing_file"]

    # Format month as 2-digit string
    month_str = @sprintf("%02d", month)
    case_name = "cfsite$(cfsite_number)_$(forcing_model)_$(experiment)_$(month_str)"
    case_output_dir = joinpath(output_dir, case_name)

    # Determine forcing type (shallow vs deep) for this cfsite
    forcing_type = get_cfsite_type_from_number(cfsite_number)

    @info "Running case $case_index: $case_name (type: $forcing_type)"

    try
        # Load base model config
        config_dict = YAML.load_file(model_config_path)

        # Set output directory
        config_dict["output_dir"] = case_output_dir

        # Build TOML file list
        # The sensitivity toml_path already contains the complete parameter set
        # (base parameters + any overrides), so we REPLACE the model config's
        # toml list to avoid duplicate key errors.
        toml_list = String[]

        # Add forcing-type-specific TOML (shallow vs deep relaxation settings)
        # These contain forcing parameters that don't overlap with sensitivity params
        if haskey(experiment_config, "forcing_toml_files") &&
           haskey(experiment_config["forcing_toml_files"], forcing_type)
            forcing_toml_rel = experiment_config["forcing_toml_files"][forcing_type]
            model_config_dir = dirname(model_config_path)
            forcing_toml_path = joinpath(model_config_dir, forcing_toml_rel)
            if isfile(forcing_toml_path)
                push!(toml_list, abspath(forcing_toml_path))
                @info "Added forcing config: $forcing_toml_path for $forcing_type case"
            else
                @warn "Forcing TOML not found: $forcing_toml_path"
            end
        end

        # Add the sensitivity parameter file (contains base params + overrides)
        # This replaces the model config's original toml list
        push!(toml_list, abspath(toml_path))
        config_dict["toml"] = toml_list

        # Set forcing file and cfsite
        config_dict["external_forcing_file"] = forcing_file
        config_dict["cfsite_number"] = "site$cfsite_number"

        # Disable default diagnostics to reduce output
        config_dict["output_default_diagnostics"] = false

        # Create AtmosConfig
        comms_ctx = ClimaComms.SingletonCommsContext()
        atmos_config = CA.AtmosConfig(config_dict; comms_ctx)

        # Run simulation
        simulation = CA.get_simulation(atmos_config)
        sol_res = CA.solve_atmos!(simulation)

        if sol_res.ret_code == :simulation_crashed
            @warn "Simulation crashed for $case_name"
            return Dict(
                "case" => case_name,
                "status" => "crashed",
                "error" => "Simulation crashed",
            )
        end

        return Dict(
            "case" => case_name,
            "status" => "success",
            "output_dir" => case_output_dir,
        )

    catch e
        @error "Error running case $case_name" exception = (e, catch_backtrace())
        return Dict(
            "case" => case_name,
            "status" => "failed",
            "error" => string(e),
        )
    end
end

"""
Generate list of cases from config (cfsite numbers × months).
"""
function generate_cases_from_config(config::Dict)
    cases = Dict{String,Any}[]

    # Get cfsite numbers
    cfsite_numbers = get(config["run"], "cfsite_cases", nothing)
    if isnothing(cfsite_numbers)
        # Default shallow cfsites if not specified
        cfsite_numbers = [2, 3, 4, 5, 6, 7]
    end
    cfsite_numbers = collect(cfsite_numbers)

    # Get forcing config
    forcing_config = get(config, "forcing", Dict())
    months = get(forcing_config, "months", [7])
    forcing_model = get(forcing_config, "model", "HadGEM2-A")
    experiment = get(forcing_config, "experiment", "amip")
    base_path = get(forcing_config, "base_path", "/resnick/groups/esm/zhaoyi/GCMForcedLES/forcing/corrected")

    # Handle single month vs list
    if isa(months, Number)
        months = [months]
    end
    months = collect(months)

    # Generate all combinations
    for cfsite_number in cfsite_numbers
        for month in months
            month_str = @sprintf("%02d", month)
            forcing_file = joinpath(base_path, "$(forcing_model)_$(experiment).2004-2008.$(month_str).nc")

            push!(cases, Dict{String,Any}(
                "cfsite_number" => cfsite_number,
                "month" => month,
                "forcing_model" => forcing_model,
                "experiment" => experiment,
                "forcing_file" => forcing_file,
            ))
        end
    end

    # Apply max_cases limit
    max_cases = get(config["run"], "max_cases", nothing)
    if !isnothing(max_cases) && length(cases) > max_cases
        cases = cases[1:max_cases]
    end

    return cases
end

"""
Get cfsite type (shallow or deep) from cfsite number.
"""
function get_cfsite_type_from_number(cfsite_number::Int)
    shallow_cfsites = Set([2, 3, 4, 5, 6, 7, 17, 18, 22, 23, 30, 94])
    return cfsite_number in shallow_cfsites ? "shallow" : "deep"
end
