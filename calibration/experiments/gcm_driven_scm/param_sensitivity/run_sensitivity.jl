"""
Parameter Sensitivity Analysis for SCM Calibration
===================================================

This script performs a parameter sweep for sensitivity analysis, running SCM simulations
with one parameter varied at a time while keeping others at baseline values.

Usage:
    julia --project=.. run_sensitivity.jl [--config sensitivity_config.toml] [--dry-run]

The script:
1. Reads parameter sweep values from the config file
2. Generates TOML parameter files for each sweep point
3. Runs simulations in parallel using distributed workers
4. Organizes output by parameter name and sweep value
"""

import ClimaAtmos as CA
import YAML
import TOML
using Distributed
using ArgParse
using Dates
using Printf

import ClimaComms
@static pkgversion(ClimaComms) >= v"0.6" && ClimaComms.@import_required_backends

include("sensitivity_helpers.jl")

function parse_commandline()
    s = ArgParseSettings(
        description = "Run parameter sensitivity analysis for SCM simulations",
    )
    @add_arg_table! s begin
        "--config"
        help = "Path to sensitivity configuration TOML file"
        default = "sensitivity_config.toml"

        "--dry-run"
        help = "Only generate configs without running simulations"
        action = :store_true

        "--baseline-only"
        help = "Only run baseline simulation (no parameter sweeps)"
        action = :store_true
    end
    return parse_args(s)
end

function main()
    args = parse_commandline()
    config_path = args["config"]
    dry_run = args["dry-run"]
    baseline_only = args["baseline-only"]

    # Load configuration
    @info "Loading sensitivity configuration from $config_path"
    config = TOML.parsefile(config_path)

    # Setup paths (relative to gcm_driven_scm directory)
    experiment_dir = dirname(@__DIR__)
    model_config_path = joinpath(experiment_dir, config["paths"]["model_config"])
    base_toml_path = joinpath(experiment_dir, config["paths"]["base_toml"])
    output_dir = config["paths"]["output_dir"]

    # Override dry_run from config if specified
    dry_run = dry_run || get(config["run"], "dry_run", false)

    @info "Configuration loaded" model_config_path base_toml_path output_dir

    # Create output directory structure
    mkpath(output_dir)
    mkpath(joinpath(output_dir, "configs"))
    mkpath(joinpath(output_dir, "logs"))

    # Copy configuration files
    cp(config_path, joinpath(output_dir, "configs", "sensitivity_config.toml"), force = true)
    cp(model_config_path, joinpath(output_dir, "configs", basename(model_config_path)), force = true)
    cp(base_toml_path, joinpath(output_dir, "configs", "base_parameters.toml"), force = true)

    # Load base TOML to get baseline parameter values
    base_params = TOML.parsefile(base_toml_path)

    # Parse parameter sweep definitions
    param_sweeps = parse_parameter_sweeps(config)
    @info "Parsed $(length(param_sweeps)) parameter sweeps"

    # Parse parameter groups (coupled parameters)
    param_groups = parse_parameter_groups(config)
    if !isempty(param_groups)
        @info "Parsed $(length(param_groups)) parameter groups" param_groups
    end

    # Generate all simulation configurations
    sim_configs = generate_simulation_configs(
        param_sweeps,
        param_groups,
        base_params,
        model_config_path,
        base_toml_path,
        output_dir,
        config;
        baseline_only = baseline_only,
    )

    @info "Generated $(length(sim_configs)) simulation configurations"

    # Save simulation manifest
    save_simulation_manifest(sim_configs, output_dir)

    if dry_run
        @info "Dry run mode - not running simulations"
        print_simulation_summary(sim_configs)
        return nothing, nothing, true
    end

    # Return config and sim_configs for top-level execution
    return config, sim_configs, false
end

"""
Parse parameter sweep definitions from config.
"""
function parse_parameter_sweeps(config::Dict)
    sweeps = Dict{String,Any}()

    if haskey(config, "parameters")
        for (param_name, param_def) in config["parameters"]
            sweeps[param_name] = Dict(
                "values" => param_def["values"],
                "type" => get(param_def, "type", "float"),
            )
        end
    end

    return sweeps
end

"""
Parse parameter groups (coupled parameters) from config.
"""
function parse_parameter_groups(config::Dict)
    groups = Dict{String,Vector{String}}()

    if haskey(config, "groups")
        for (group_name, group_def) in config["groups"]
            groups[group_name] = group_def["parameters"]
        end
    end

    return groups
end

"""
Generate simulation configurations for all parameter sweep points.
"""
function generate_simulation_configs(
    param_sweeps::Dict,
    param_groups::Dict,
    base_params::Dict,
    model_config_path::String,
    base_toml_path::String,
    output_dir::String,
    config::Dict;
    baseline_only::Bool = false,
)
    sim_configs = Vector{Dict{String,Any}}()

    # Generate cases from config (cfsite numbers × months)
    cases = generate_cases_from_config(config)
    @info "Generated $(length(cases)) cases from config"

    # Extract experiment config settings needed for simulations (e.g., forcing TOML files)
    experiment_config = Dict{String,Any}()
    if haskey(config, "forcing_toml_files")
        experiment_config["forcing_toml_files"] = config["forcing_toml_files"]
    end

    # 1. Baseline simulation (all parameters at default values from base_toml)
    baseline_toml_path = joinpath(output_dir, "configs", "baseline_parameters.toml")
    baseline_config = Dict{String,Any}(
        "name" => "baseline",
        "param_name" => "baseline",
        "param_value" => nothing,
        "param_index" => 0,
        "output_subdir" => joinpath(output_dir, "baseline"),
        "toml_path" => baseline_toml_path,
        "model_config_path" => model_config_path,
        "cases" => cases,
        "experiment_config" => experiment_config,
        "replaces_model_toml" => true,  # Signal to replace model config's toml list
    )

    # For baseline, copy the base_toml as-is (this IS the baseline we want)
    mkpath(dirname(baseline_toml_path))
    cp(base_toml_path, baseline_toml_path, force = true)

    push!(sim_configs, baseline_config)

    if baseline_only
        return sim_configs
    end

    # 2. Generate configs for each parameter sweep
    # Track which parameters are part of groups (to avoid duplicate sweeps)
    grouped_params = Set{String}()
    for (_, params) in param_groups
        for p in params[2:end]  # Skip first (driver) param
            push!(grouped_params, p)
        end
    end

    for (param_name, sweep_def) in param_sweeps
        # Skip if this parameter is a follower in a group
        if param_name in grouped_params
            continue
        end

        values = sweep_def["values"]
        param_type = sweep_def["type"]

        for (idx, value) in enumerate(values)
            # Create parameter override dict
            param_overrides = Dict{String,Any}()

            # Set this parameter
            param_overrides[param_name] = format_param_value(value, param_type)

            # Check if this parameter drives a group
            for (group_name, group_params) in param_groups
                if group_params[1] == param_name
                    # This parameter is the driver - apply same value to followers
                    for follower in group_params[2:end]
                        param_overrides[follower] = format_param_value(value, param_type)
                    end
                end
            end

            # Generate descriptive names
            value_str = format_value_for_filename(value)
            config_name = "$(param_name)_$(value_str)"

            # Create output directory and TOML path
            # Structure: output_dir/<param_name>/val_<value>/
            param_output_dir = joinpath(output_dir, param_name, "val_$(value_str)")
            toml_path = joinpath(output_dir, "configs", "params", "$(config_name).toml")

            # Create the parameter TOML file
            mkpath(dirname(toml_path))
            create_override_toml(base_toml_path, param_overrides, toml_path)

            sim_config = Dict{String,Any}(
                "name" => config_name,
                "param_name" => param_name,
                "param_value" => value,
                "param_index" => idx,
                "output_subdir" => param_output_dir,
                "toml_path" => toml_path,
                "model_config_path" => model_config_path,
                "cases" => cases,
                "experiment_config" => experiment_config,
                "overrides" => param_overrides,
                "replaces_model_toml" => true,  # Signal to replace model config's toml list
            )

            push!(sim_configs, sim_config)
        end
    end

    return sim_configs
end

"""
Format parameter value for TOML output.
"""
function format_param_value(value, param_type::String)
    return Dict("value" => value, "type" => param_type)
end

"""
Format value for use in filename (filesystem-safe, descriptive).

Examples:
- 0.22 -> "0.22"
- 1.5e-4 -> "1.5e-4"  
- [1.0, 2.0, 0.3] -> "1_2_0.3"
"""
function format_value_for_filename(value)
    if isa(value, AbstractVector)
        # For vectors, join with underscores
        return join([format_single_value(v) for v in value], "_")
    else
        return format_single_value(value)
    end
end

function format_single_value(v)
    if isa(v, AbstractFloat)
        if v == 0.0
            return "0"
        elseif abs(v) >= 0.001 && abs(v) < 10000
            # Normal notation for reasonable values
            s = @sprintf("%.4g", v)
            # Remove trailing zeros after decimal
            s = replace(s, r"\.?0+$" => "")
            return s
        else
            # Scientific notation for very small/large
            return @sprintf("%.2e", v)
        end
    else
        return string(v)
    end
end

"""
Create a TOML file with parameter overrides applied to the base.

This creates a complete parameter file by starting with base_toml and applying
the overrides. The resulting file REPLACES the model config's toml list to
avoid duplicate key errors when ClimaParams merges TOML files.
"""
function create_override_toml(base_toml_path::String, overrides::Dict, output_path::String)
    # Start with base parameters
    merged = TOML.parsefile(base_toml_path)

    # Apply overrides (replaces values in base)
    for (param_name, param_def) in overrides
        merged[param_name] = param_def
    end

    # Write merged result
    open(output_path, "w") do io
        TOML.print(io, merged)
    end
end

"""
Save simulation manifest for tracking/resuming.
"""
function save_simulation_manifest(sim_configs::Vector, output_dir::String)
    manifest = Dict(
        "created_at" => string(now()),
        "num_configs" => length(sim_configs),
        "configs" => [
            Dict(
                "name" => c["name"],
                "param_name" => c["param_name"],
                "param_value" => isnothing(c["param_value"]) ? "baseline" : c["param_value"],
                "output_subdir" => c["output_subdir"],
            ) for c in sim_configs
        ],
    )

    manifest_path = joinpath(output_dir, "manifest.toml")
    open(manifest_path, "w") do io
        TOML.print(io, manifest)
    end
    @info "Saved manifest to $manifest_path"
end

"""
Print summary of simulations to run.
"""
function print_simulation_summary(sim_configs::Vector)
    println("\n" * "="^60)
    println("SIMULATION SUMMARY")
    println("="^60)
    println("Total configurations: $(length(sim_configs))")
    println()

    # Group by parameter
    by_param = Dict{String,Int}()
    for c in sim_configs
        param = c["param_name"]
        by_param[param] = get(by_param, param, 0) + 1
    end

    println("Configurations per parameter:")
    for (param, count) in sort(collect(by_param))
        println("  $param: $count")
    end

    println()
    cases = sim_configs[1]["cases"]
    println("Cases per config: $(length(cases))")
    if !isempty(cases)
        # Show case details
        cfsites = unique([c["cfsite_number"] for c in cases])
        months = unique([c["month"] for c in cases])
        println("  Cfsite numbers: $(join(cfsites, ", "))")
        println("  Months: $(join(months, ", "))")
    end
    println("="^60 * "\n")
end

"""
Run sensitivity simulations using distributed workers.
"""
function run_sensitivity_simulations(sim_configs::Vector, config::Dict)
    # Run simulations in parallel
    @info "Starting $(length(sim_configs)) simulation configurations"
    start_time = time()

    results = pmap(run_single_sensitivity_config, sim_configs)

    elapsed = (time() - start_time) / 60.0
    @info "All simulations complete in $(@sprintf("%.1f", elapsed)) minutes"

    # Summarize results
    n_success = count(r -> r["status"] == "success", results)
    n_failed = count(r -> r["status"] == "failed", results)
    @info "Results: $n_success succeeded, $n_failed failed"

    # Save results summary
    results_path = joinpath(config["paths"]["output_dir"], "results_summary.toml")
    open(results_path, "w") do io
        TOML.print(io, Dict("results" => results))
    end

    return results
end

# =============================================================================
# TOP-LEVEL SCRIPT EXECUTION
# =============================================================================
# IMPORTANT: Workers must be added early, before heavy setup work.
# Otherwise workers timeout waiting for master to accept connections.

# Quick parse to check for dry-run (don't add workers if dry run)
args = parse_commandline()
if !args["dry-run"]
    # Load config just to get num_workers
    config_path = args["config"]
    config_for_workers = TOML.parsefile(config_path)
    num_workers = config_for_workers["run"]["num_workers"]
    
    # Add workers EARLY - before any heavy setup
    @info "Adding $num_workers workers"
    addprocs(num_workers)
    @info "Workers added: $(nprocs()) processes total"
    
    # Load modules on all workers immediately after addprocs
    # Note: import ClimaComms must be in a separate @everywhere block before the @static check
    # because @static evaluates at parse time and needs ClimaComms to be defined
    @everywhere import ClimaComms
    @everywhere @static pkgversion(ClimaComms) >= v"0.6" && ClimaComms.@import_required_backends
    
    @everywhere begin
        import ClimaAtmos as CA
        import YAML
        import TOML
        using JLD2

        include(joinpath(@__DIR__, "sensitivity_helpers.jl"))
    end
end

# Now do the heavy setup work (main())
config, sim_configs, is_dry_run = main()

# Exit early if dry run
is_dry_run && exit(0)
isnothing(config) && exit(0)

# Run simulations
run_sensitivity_simulations(sim_configs, config)

@info "Sensitivity analysis complete. Results saved to $(config["paths"]["output_dir"])"
