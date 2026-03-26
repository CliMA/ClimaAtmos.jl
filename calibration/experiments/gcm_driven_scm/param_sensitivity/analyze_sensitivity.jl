"""
Analyze and Plot Parameter Sensitivity Results
===============================================

This script loads sensitivity simulation outputs and creates visualizations
showing how each parameter affects the SCM outputs.

Usage:
    julia --project=.. analyze_sensitivity.jl --output-dir /path/to/sensitivity/output
"""

using ArgParse
using TOML
using YAML
using Statistics
using LinearAlgebra
using Printf
using Dates

import ClimaAtmos as CA
import ClimaComms
@static pkgversion(ClimaComms) >= v"0.6" && ClimaComms.@import_required_backends

using NCDatasets
using CairoMakie

include(joinpath(@__DIR__, "..", "helper_funcs.jl"))

function parse_commandline()
    s = ArgParseSettings(description = "Analyze parameter sensitivity results")
    @add_arg_table! s begin
        "--output-dir"
        help = "Path to sensitivity output directory"
        required = true

        "--vars"
        help = "Variables to analyze (comma-separated)"
        default = "thetaa,hus,clw"

        "--save-plots"
        help = "Save plots to output directory"
        action = :store_true
    end
    return parse_args(s)
end

"""
Load simulation results from a sensitivity run directory.
"""
function load_sensitivity_results(output_dir::String)
    # Load manifest
    manifest_path = joinpath(output_dir, "manifest.toml")
    if !isfile(manifest_path)
        error("Manifest not found at $manifest_path")
    end
    manifest = TOML.parsefile(manifest_path)

    # Load sensitivity config
    config_path = joinpath(output_dir, "configs", "sensitivity_config.toml")
    config = TOML.parsefile(config_path)

    results = Dict{String,Any}()
    results["manifest"] = manifest
    results["config"] = config
    results["parameters"] = Dict{String,Any}()

    # Load baseline results
    baseline_dir = joinpath(output_dir, "baseline")
    if isdir(baseline_dir)
        results["baseline"] = load_case_outputs(baseline_dir)
    end

    # Load parameter sweep results
    for cfg in manifest["configs"]
        param_name = cfg["param_name"]
        if param_name == "baseline"
            continue
        end

        if !haskey(results["parameters"], param_name)
            results["parameters"][param_name] = Dict{String,Any}(
                "values" => [],
                "outputs" => [],
            )
        end

        param_dir = cfg["output_subdir"]
        if isdir(param_dir)
            param_results = load_case_outputs(param_dir)
            push!(results["parameters"][param_name]["values"], cfg["param_value"])
            push!(results["parameters"][param_name]["outputs"], param_results)
        end
    end

    return results
end

"""
Load outputs from all LES cases in a directory.
"""
function load_case_outputs(sim_dir::String)
    outputs = Dict{String,Any}()

    # Find all case directories
    case_dirs = filter(d -> isdir(joinpath(sim_dir, d)) && startswith(d, "cfsite_"), readdir(sim_dir))

    for case_dir in case_dirs
        case_path = joinpath(sim_dir, case_dir, "output_active")
        if !isdir(case_path)
            continue
        end

        # Find diagnostic files
        diag_files = filter(f -> endswith(f, ".nc"), readdir(case_path))
        if isempty(diag_files)
            continue
        end

        outputs[case_dir] = Dict{String,Any}()

        for diag_file in diag_files
            nc_path = joinpath(case_path, diag_file)
            try
                NCDataset(nc_path) do ds
                    for var_name in keys(ds)
                        if var_name in ["time", "z", "lat", "lon"]
                            continue
                        end
                        if !haskey(outputs[case_dir], var_name)
                            outputs[case_dir][var_name] = Array(ds[var_name])
                        end
                    end
                    if haskey(ds, "z")
                        outputs[case_dir]["z"] = Array(ds["z"])
                    end
                    if haskey(ds, "time")
                        outputs[case_dir]["time"] = Array(ds["time"])
                    end
                end
            catch e
                @warn "Could not load $nc_path: $e"
            end
        end
    end

    return outputs
end

"""
Compute profile statistics across cases.
"""
function compute_profile_stats(outputs::Dict, var_name::String; time_avg::Bool = true)
    profiles = []
    z_common = nothing

    for (case_name, case_data) in outputs
        if !haskey(case_data, var_name)
            continue
        end

        data = case_data[var_name]
        z = get(case_data, "z", nothing)

        if time_avg && ndims(data) > 1
            # Average over time (last dimension)
            data = mean(data, dims = ndims(data))
            data = dropdims(data, dims = ndims(data))
        end

        push!(profiles, data)
        if isnothing(z_common) && !isnothing(z)
            z_common = z
        end
    end

    if isempty(profiles)
        return nothing, nothing, nothing
    end

    # Stack profiles and compute stats
    profile_matrix = hcat(profiles...)
    mean_profile = vec(mean(profile_matrix, dims = 2))
    std_profile = vec(std(profile_matrix, dims = 2))

    return mean_profile, std_profile, z_common
end

"""
Plot sensitivity analysis for a single parameter.
"""
function plot_parameter_sensitivity(
    results::Dict,
    param_name::String,
    var_name::String;
    ax = nothing,
)
    if !haskey(results["parameters"], param_name)
        @warn "Parameter $param_name not found in results"
        return nothing
    end

    param_data = results["parameters"][param_name]
    values = param_data["values"]
    outputs_list = param_data["outputs"]

    # Get baseline profile
    baseline_mean, baseline_std, z = compute_profile_stats(results["baseline"], var_name)
    if isnothing(baseline_mean)
        @warn "No baseline data for $var_name"
        return nothing
    end

    # Create figure if not provided
    if isnothing(ax)
        fig = Figure(size = (800, 600))
        ax = Axis(
            fig[1, 1],
            xlabel = var_name,
            ylabel = "Height [m]",
            title = "Sensitivity of $var_name to $param_name",
        )
    end

    # Plot baseline
    lines!(ax, baseline_mean, z, color = :black, linewidth = 2, label = "Baseline")

    # Plot each parameter value
    n_values = length(values)
    colors = cgrad(:viridis, n_values, categorical = true)

    for (i, (val, outputs)) in enumerate(zip(values, outputs_list))
        mean_prof, _, _ = compute_profile_stats(outputs, var_name)
        if !isnothing(mean_prof)
            val_str = isa(val, AbstractVector) ? join([@sprintf("%.2g", v) for v in val], ",") : @sprintf("%.3g", val)
            lines!(ax, mean_prof, z, color = colors[i], linewidth = 1.5, label = val_str)
        end
    end

    axislegend(ax, position = :rt, framevisible = false, labelsize = 10)

    return ax
end

"""
Create summary sensitivity plot for all parameters.
"""
function plot_sensitivity_summary(results::Dict, var_names::Vector{String}; save_path = nothing)
    param_names = collect(keys(results["parameters"]))
    n_params = length(param_names)
    n_vars = length(var_names)

    fig = Figure(size = (300 * n_vars, 250 * n_params))

    for (j, var_name) in enumerate(var_names)
        for (i, param_name) in enumerate(param_names)
            ax = Axis(
                fig[i, j],
                xlabel = i == n_params ? var_name : "",
                ylabel = j == 1 ? "z [m]" : "",
                title = i == 1 ? var_name : "",
            )

            if j == 1
                # Add parameter name on left side
                Label(fig[i, 0], param_name, rotation = π / 2, tellheight = false)
            end

            plot_parameter_sensitivity(results, param_name, var_name; ax = ax)
        end
    end

    if !isnothing(save_path)
        save(save_path, fig)
        @info "Saved summary plot to $save_path"
    end

    return fig
end

"""
Compute sensitivity metrics for each parameter.
"""
function compute_sensitivity_metrics(results::Dict, var_name::String)
    metrics = Dict{String,Any}()

    baseline_mean, _, z = compute_profile_stats(results["baseline"], var_name)
    if isnothing(baseline_mean)
        return metrics
    end

    for (param_name, param_data) in results["parameters"]
        values = param_data["values"]
        outputs_list = param_data["outputs"]

        # Compute RMSE from baseline for each value
        rmses = Float64[]
        for outputs in outputs_list
            mean_prof, _, _ = compute_profile_stats(outputs, var_name)
            if !isnothing(mean_prof) && length(mean_prof) == length(baseline_mean)
                rmse = sqrt(mean((mean_prof .- baseline_mean) .^ 2))
                push!(rmses, rmse)
            end
        end

        if !isempty(rmses)
            metrics[param_name] = Dict(
                "values" => values,
                "rmse_from_baseline" => rmses,
                "max_rmse" => maximum(rmses),
                "mean_rmse" => mean(rmses),
            )
        end
    end

    return metrics
end

"""
Print sensitivity summary table.
"""
function print_sensitivity_table(results::Dict, var_names::Vector{String})
    println("\n" * "="^80)
    println("PARAMETER SENSITIVITY SUMMARY")
    println("="^80)

    for var_name in var_names
        metrics = compute_sensitivity_metrics(results, var_name)

        println("\nVariable: $var_name")
        println("-"^60)
        println(@sprintf("%-35s %12s %12s", "Parameter", "Max RMSE", "Mean RMSE"))
        println("-"^60)

        # Sort by max RMSE (most sensitive first)
        sorted_params = sort(collect(keys(metrics)), by = k -> -metrics[k]["max_rmse"])

        for param_name in sorted_params
            m = metrics[param_name]
            println(@sprintf("%-35s %12.4g %12.4g", param_name, m["max_rmse"], m["mean_rmse"]))
        end
    end

    println("\n" * "="^80)
end

function main()
    args = parse_commandline()
    output_dir = args["output-dir"]
    var_names = split(args["vars"], ",")
    save_plots = args["save-plots"]

    @info "Loading sensitivity results from $output_dir"
    results = load_sensitivity_results(output_dir)

    @info "Loaded results for $(length(results["parameters"])) parameters"

    # Print summary table
    print_sensitivity_table(results, var_names)

    # Create plots
    if save_plots
        plots_dir = joinpath(output_dir, "plots")
        mkpath(plots_dir)

        # Summary plot
        summary_path = joinpath(plots_dir, "sensitivity_summary.png")
        plot_sensitivity_summary(results, var_names; save_path = summary_path)

        # Individual parameter plots
        for param_name in keys(results["parameters"])
            for var_name in var_names
                fig = Figure(size = (600, 500))
                ax = Axis(
                    fig[1, 1],
                    xlabel = var_name,
                    ylabel = "Height [m]",
                    title = "Sensitivity of $var_name to $param_name",
                )
                plot_parameter_sensitivity(results, param_name, var_name; ax = ax)
                plot_path = joinpath(plots_dir, "$(param_name)_$(var_name).png")
                save(plot_path, fig)
            end
        end

        @info "Plots saved to $plots_dir"
    end

    return results
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
