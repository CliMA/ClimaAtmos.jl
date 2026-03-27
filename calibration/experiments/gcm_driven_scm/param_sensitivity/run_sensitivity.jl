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

import ClimaCalibrate as CAL
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
Compute the number of grid combinations and warn if unreasonable.
Returns (num_combos, is_ok, param_counts).
"""
function check_grid_size(param_sweeps::Dict, grid_params::Union{Nothing,Vector{String}}, max_configs::Int)
    # Filter to grid parameters if specified
    params_to_use = if isnothing(grid_params)
        collect(keys(param_sweeps))
    else
        filter(p -> haskey(param_sweeps, p), grid_params)
    end
    
    if isempty(params_to_use)
        return 0, false, Dict{String,Int}()
    end
    
    param_counts = Dict{String,Int}()
    for p in params_to_use
        param_counts[p] = length(param_sweeps[p]["values"])
    end
    
    num_combos = prod(values(param_counts))
    is_ok = num_combos <= max_configs
    
    return num_combos, is_ok, param_counts
end

"""
Print grid search warning/summary.
"""
function print_grid_warning(num_combos::Int, param_counts::Dict, max_configs::Int, num_cases::Int)
    total_sims = num_combos * num_cases
    
    println("\n" * "="^70)
    println("GRID SEARCH MODE")
    println("="^70)
    println("\nParameters in grid ($(length(param_counts)) total):")
    for (p, n) in sort(collect(param_counts), by=x->x[1])
        println("  $p: $n values")
    end
    println("\nTotal combinations: $num_combos")
    println("Cases per combination: $num_cases")
    println("Total simulations: $total_sims")
    
    if num_combos > max_configs
        println("\n" * "!"^70)
        println("WARNING: Grid size ($num_combos) exceeds max_grid_configs ($max_configs)!")
        println("!"^70)
        println("\nOptions:")
        println("  1. Reduce the number of parameters in grid_parameters")
        println("  2. Reduce the number of values per parameter")
        println("  3. Increase max_grid_configs (if you really want this many runs)")
        println("\nRecommendation: Keep grid search to ≤5 parameters with ≤5 values each")
        println("  (5^5 = 3,125 combinations, manageable but expensive)")
        println("!"^70 * "\n")
    else
        println("\nGrid size is within limits. Proceeding...")
    end
    println("="^70 * "\n")
end

"""
Generate all combinations from parameter value lists (Cartesian product).
"""
function generate_grid_combinations(param_sweeps::Dict, grid_params::Vector{String})
    # Get ordered parameter names and their values
    param_names = sort(grid_params)
    value_lists = [param_sweeps[p]["values"] for p in param_names]
    type_list = [param_sweeps[p]["type"] for p in param_names]
    
    # Generate Cartesian product using recursion
    combinations = Vector{Dict{String,Any}}()
    
    function recurse(idx::Int, current::Dict{String,Any})
        if idx > length(param_names)
            push!(combinations, copy(current))
            return
        end
        
        pname = param_names[idx]
        ptype = type_list[idx]
        for val in value_lists[idx]
            current[pname] = format_param_value(val, ptype)
            recurse(idx + 1, current)
        end
        delete!(current, pname)
    end
    
    recurse(1, Dict{String,Any}())
    return combinations, param_names
end

"""
Generate simulation configurations for all parameter sweep points.
Supports both "one_at_a_time" and "grid" modes.
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
        "replaces_model_toml" => true,
    )

    # For baseline, copy the base_toml as-is (this IS the baseline we want)
    mkpath(dirname(baseline_toml_path))
    cp(base_toml_path, baseline_toml_path, force = true)

    push!(sim_configs, baseline_config)

    if baseline_only
        return sim_configs
    end

    # Check sweep mode
    mode = get(config["run"], "mode", "one_at_a_time")
    
    if mode == "grid"
        return generate_grid_configs!(
            sim_configs, param_sweeps, param_groups, base_toml_path, model_config_path,
            output_dir, cases, experiment_config, config
        )
    else
        return generate_one_at_a_time_configs!(
            sim_configs, param_sweeps, param_groups, base_toml_path,
            model_config_path, output_dir, cases, experiment_config
        )
    end
end

"""
Generate configs for one-at-a-time sensitivity analysis.
"""
function generate_one_at_a_time_configs!(
    sim_configs::Vector,
    param_sweeps::Dict,
    param_groups::Dict,
    base_toml_path::String,
    model_config_path::String,
    output_dir::String,
    cases::Vector,
    experiment_config::Dict,
)
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
                "replaces_model_toml" => true,
            )

            push!(sim_configs, sim_config)
        end
    end

    return sim_configs
end

"""
Generate configs for full grid (combinatorial) search.
Supports parameter groups (coupled parameters) - followers take the driver's value.
"""
function generate_grid_configs!(
    sim_configs::Vector,
    param_sweeps::Dict,
    param_groups::Dict,
    base_toml_path::String,
    model_config_path::String,
    output_dir::String,
    cases::Vector,
    experiment_config::Dict,
    config::Dict,
)
    # Track which parameters are followers in groups (to exclude from grid)
    grouped_params = Set{String}()
    for (_, params) in param_groups
        for p in params[2:end]  # Skip first (driver) param
            push!(grouped_params, p)
        end
    end
    
    # Get grid parameters (subset or all), excluding followers
    grid_params_raw = get(config["run"], "grid_parameters", nothing)
    grid_params = if isnothing(grid_params_raw)
        collect(keys(param_sweeps))
    else
        filter(p -> haskey(param_sweeps, p), grid_params_raw)
    end
    
    # Filter out follower parameters from the grid
    grid_params_filtered = filter(p -> !(p in grouped_params), grid_params)
    
    # Warn if any followers were filtered out
    filtered_out = filter(p -> p in grouped_params, grid_params)
    if !isempty(filtered_out)
        @info "Coupled parameters excluded from grid (will follow their drivers): $(join(filtered_out, ", "))"
    end
    
    if isempty(grid_params_filtered)
        @warn "No valid parameters for grid search (all are followers?)"
        return sim_configs
    end
    
    # Check grid size and warn
    max_configs = get(config["run"], "max_grid_configs", 1000)
    num_combos, is_ok, param_counts = check_grid_size(param_sweeps, grid_params_filtered, max_configs)
    
    print_grid_warning(num_combos, param_counts, max_configs, length(cases))
    
    if !is_ok
        error("Grid size ($num_combos) exceeds max_grid_configs ($max_configs). " *
              "Reduce parameters/values or increase max_grid_configs in config.")
    end
    
    # Generate all combinations (only for non-follower params)
    combinations, param_names = generate_grid_combinations(param_sweeps, grid_params_filtered)
    @info "Generated $(length(combinations)) grid combinations for $(length(param_names)) parameters"
    
    # Create configs for each combination
    mkpath(joinpath(output_dir, "configs", "grid"))
    
    for (idx, combo) in enumerate(combinations)
        # Apply coupled parameter values (followers get driver's value)
        combo_with_followers = copy(combo)
        for (group_name, group_params) in param_groups
            driver = group_params[1]
            if haskey(combo, driver)
                # Driver is in the grid - apply its value to all followers
                driver_value = combo[driver]
                for follower in group_params[2:end]
                    combo_with_followers[follower] = driver_value
                end
            end
        end
        
        # Build config name from all parameter values
        name_parts = String[]
        for pname in param_names
            val = combo[pname]["value"]
            push!(name_parts, "$(pname)=$(format_value_for_filename(val))")
        end
        config_name = "grid_$(idx)"
        
        # Output directory uses index for simplicity (full names get too long)
        combo_output_dir = joinpath(output_dir, "grid", @sprintf("combo_%04d", idx))
        toml_path = joinpath(output_dir, "configs", "grid", "$(config_name).toml")
        
        # Create the parameter TOML file (includes followers)
        mkpath(dirname(toml_path))
        create_override_toml(base_toml_path, combo_with_followers, toml_path)
        
        # Build a summary dict of the parameter values (for manifest/analysis)
        # Include both drivers and followers for transparency
        param_values = Dict{String,Any}()
        for pname in param_names
            param_values[pname] = combo[pname]["value"]
        end
        # Also record follower values
        for (group_name, group_params) in param_groups
            driver = group_params[1]
            if haskey(combo, driver)
                for follower in group_params[2:end]
                    param_values[follower] = combo[driver]["value"]
                end
            end
        end
        
        sim_config = Dict{String,Any}(
            "name" => config_name,
            "param_name" => "grid",
            "param_value" => param_values,
            "param_index" => idx,
            "output_subdir" => combo_output_dir,
            "toml_path" => toml_path,
            "model_config_path" => model_config_path,
            "cases" => cases,
            "experiment_config" => experiment_config,
            "overrides" => combo_with_followers,
            "replaces_model_toml" => true,
            "grid_params" => param_names,
        )
        
        push!(sim_configs, sim_config)
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
    # Check if grid mode
    is_grid = any(c -> c["param_name"] == "grid", sim_configs)
    
    manifest = Dict(
        "created_at" => string(now()),
        "num_configs" => length(sim_configs),
        "mode" => is_grid ? "grid" : "one_at_a_time",
        "configs" => [
            begin
                pval = c["param_value"]
                # Handle different param_value types for TOML serialization
                pval_out = if isnothing(pval)
                    "baseline"
                elseif isa(pval, Dict)
                    # Grid mode: param_value is a Dict of param_name => value
                    pval
                else
                    pval
                end
                
                entry = Dict(
                    "name" => c["name"],
                    "param_name" => c["param_name"],
                    "param_value" => pval_out,
                    "output_subdir" => c["output_subdir"],
                )
                
                # Add grid_params if present
                if haskey(c, "grid_params")
                    entry["grid_params"] = c["grid_params"]
                end
                
                entry
            end for c in sim_configs
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

    # Check if this is grid mode
    is_grid = any(c -> c["param_name"] == "grid", sim_configs)
    
    if is_grid
        # Grid mode summary
        grid_configs = filter(c -> c["param_name"] == "grid", sim_configs)
        if !isempty(grid_configs) && haskey(grid_configs[1], "grid_params")
            println("Mode: GRID SEARCH")
            println("Grid parameters: $(join(grid_configs[1]["grid_params"], ", "))")
            println("Grid combinations: $(length(grid_configs))")
            
            # Show first few combinations as examples
            println("\nFirst 5 combinations:")
            for (i, c) in enumerate(grid_configs[1:min(5, length(grid_configs))])
                vals = c["param_value"]
                parts = ["  $i: "]
                for (k, v) in vals
                    push!(parts, "$k=$(format_value_for_filename(v))")
                end
                println(join(parts, " "))
            end
            if length(grid_configs) > 5
                println("  ... and $(length(grid_configs) - 5) more")
            end
        end
    else
        # One-at-a-time mode summary
        by_param = Dict{String,Int}()
        for c in sim_configs
            param = c["param_name"]
            by_param[param] = get(by_param, param, 0) + 1
        end

        println("Mode: ONE-AT-A-TIME")
        println("\nConfigurations per parameter:")
        for (param, count) in sort(collect(by_param))
            println("  $param: $count")
        end
    end

    println()
    cases = sim_configs[1]["cases"]
    println("Cases per config: $(length(cases))")
    total_sims = length(sim_configs) * length(cases)
    println("Total simulations: $total_sims")
    if !isempty(cases)
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
    # Load config to get worker/slurm settings
    config_path = args["config"]
    config_for_workers = TOML.parsefile(config_path)
    num_workers = config_for_workers["run"]["num_workers"]
    slurm_config = get(config_for_workers, "slurm", Dict())
    
    # Get Slurm settings for workers
    worker_time = get(slurm_config, "job_time", "06:00:00")
    worker_mem = get(slurm_config, "mem_per_cpu", "16G")
    worker_cpus = get(slurm_config, "cpus_per_task", 1)
    
    # Spawn workers via SlurmManager - each worker is a separate Slurm job
    # This allows scaling across multiple nodes
    @info "Starting $num_workers workers via SlurmManager"
    flush(stderr)
    addprocs(
        CAL.SlurmManager(num_workers);
        t = worker_time,
        mem_per_cpu = worker_mem,
        cpus_per_task = worker_cpus,
    )
    @info "Workers added: $(nprocs()) processes total"
    flush(stderr)
    
    # Load modules on all workers
    @info "Loading ClimaComms on workers..."
    flush(stderr)
    @everywhere import ClimaComms
    @everywhere @static pkgversion(ClimaComms) >= v"0.6" && ClimaComms.@import_required_backends
    
    @info "Loading ClimaAtmos and other packages on workers..."
    flush(stderr)
    @everywhere begin
        import ClimaAtmos as CA
        import YAML
        import TOML
        using JLD2

        include(joinpath(@__DIR__, "sensitivity_helpers.jl"))
    end
    @info "All workers ready"
    flush(stderr)
end

# Now do the heavy setup work (main())
config, sim_configs, is_dry_run = main()

# Exit early if dry run
is_dry_run && exit(0)
isnothing(config) && exit(0)

# Run simulations
run_sensitivity_simulations(sim_configs, config)

@info "Sensitivity analysis complete. Results saved to $(config["paths"]["output_dir"])"
