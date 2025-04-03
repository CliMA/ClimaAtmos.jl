#!/usr/bin/env julia

using ArgParse
using Distributed
addprocs(1)

@everywhere begin
    using EnsembleKalmanProcesses: TOMLInterface
    import EnsembleKalmanProcesses as EKP
    using EnsembleKalmanProcesses.ParameterDistributions
    using ClimaCalibrate: observation_map, ExperimentConfig
    using ClimaAnalysis
    using Plots
    using JLD2
    using Statistics
    using YAML
    using DataFrames
    using CSV

    include("helper_funcs.jl")
    include("observation_map.jl")
    include("get_les_metadata.jl")
end

function parse_with_settings(s)
    return ArgParse.parse_args(s)
end

function parse_args()
    s = ArgParseSettings(description = "Process ensemble Kalman statistics")
    @add_arg_table s begin
        "--output_dir"
        help = "Calibration output directory"
        required = true
        "--var_names"
        help = "Variable names to process (comma-separated)"
        default = "thetaa,hus,clw,arup,entr,detr,waup,tke"
        "--reduction"
        help = "Reduction method to use (default: inst)"
        default = "inst"
        "--iterations"
        help = "Iterations to plot (e.g., 0:11), default is all iterations"
        default = nothing
        "--save_as_csv"
        help = "Save results as CSV"
        default = true
        arg_type = Bool
        "--load_from_csv"
        help = "Load results from CSV"
        default = false
        arg_type = Bool
        "--plot_dir"
        help = "Directory to save plots (default: edmf_stats_plots)"
        default = "edmf_stats_plots"
    end
    return parse_with_settings(s)
end

@everywhere function validate_ensemble_member(iteration_dir, batch_size)
    config_dirs =
        filter(x -> isdir(joinpath(iteration_dir, x)), readdir(iteration_dir))
    num_configs = count(x -> startswith(x, "config_"), config_dirs)
    return num_configs == batch_size
end

function main()
    args = parse_args()

    output_dir = args["output_dir"]
    var_names = map(String, split(args["var_names"], ","))
    reduction = args["reduction"]
    save_as_csv = args["save_as_csv"]
    load_from_csv = args["load_from_csv"]
    plot_dir = args["plot_dir"]

    if isnothing(args["iterations"])
        iterations = nothing
    else
        iterations = eval(Meta.parse(args["iterations"]))
    end

    mkpath(joinpath(output_dir, "plots", plot_dir))

    # Load configuration data
    config_dict =
        YAML.load_file(joinpath(output_dir, "configs", "experiment_config.yml"))
    n_vert_levels = config_dict["dims_per_var"]
    z_max = config_dict["z_max"]
    ensemble_size = config_dict["ensemble_size"]
    cal_vars = config_dict["y_var_names"]
    const_noise_by_var = config_dict["const_noise_by_var"]
    n_iterations = config_dict["n_iterations"]
    batch_size = config_dict["batch_size"]
    model_config_dict =
        YAML.load_file(joinpath(output_dir, "configs", "model_config.yml"))

    if isnothing(iterations)
        iterations = 0:(n_iterations - 1)
    end

    ref_paths, _ = get_les_calibration_library()

    @everywhere function calculate_statistics(y_var)
        non_nan_values = y_var[.!isnan.(y_var)]
        if length(non_nan_values) == 0
            return NaN, NaN, NaN
        end
        col_mean = mean(non_nan_values)
        col_max = maximum(non_nan_values)
        col_min = minimum(non_nan_values)
        return col_mean, col_max, col_min
    end

    @everywhere function compute_ensemble_squared_error(ensemble_data, y_true)
        return vec(sum((ensemble_data .- y_true) .^ 2, dims = 1))
    end

    @everywhere function process_iteration(
        iteration,
        output_dir,
        var_names,
        n_vert_levels,
        config_dict,
        z_max,
        cal_vars,
        const_noise_by_var,
        ref_paths,
        reduction,
        ensemble_size,
        batch_size,
    )
        println("Processing Iteration: $iteration")
        stats_df = DataFrame(
            iteration = Int[],
            var_name = String[],
            mean = Float64[],
            max = Float64[],
            min = Float64[],
            rmse = Union{Missing, Float64}[],
            rmse_min = Union{Missing, Float64}[],
            rmse_max = Union{Missing, Float64}[],
            rmse_std = Union{Missing, Float64}[],
        )
        config_indices = get_batch_indicies_in_iteration(iteration, output_dir)
        iteration_dir =
            joinpath(output_dir, "iteration_$(lpad(iteration, 3, '0'))")

        valid_ensemble_members = filter(
            config_i -> validate_ensemble_member(
                joinpath(iteration_dir, "member_$(lpad(config_i, 3, '0'))"),
                batch_size,
            ),
            config_indices,
        )

        for var_name in var_names
            means = Float64[]
            maxs = Float64[]
            mins = Float64[]
            sum_squared_errors = zeros(Float64, ensemble_size)

            for config_i in valid_ensemble_members
                data, zc_model = ensemble_data(
                    process_profile_variable,
                    iteration,
                    config_i,
                    config_dict;
                    var_name = var_name,
                    reduction = reduction,
                    output_dir = output_dir,
                    z_max = z_max,
                    n_vert_levels = n_vert_levels,
                    return_z_interp = true,
                )
                for i in 1:size(data, 2)
                    y_var = data[:, i]
                    col_mean, col_max, col_min = calculate_statistics(y_var)
                    push!(means, col_mean)
                    push!(maxs, col_max)
                    push!(mins, col_min)
                end
                if in(var_name, cal_vars)
                    ref_path = ref_paths[config_i]
                    cfsite_number, _, _, _ = parse_les_path(ref_path)
                    forcing_type = get_cfsite_type(cfsite_number)

                    ti = config_dict["y_t_start_sec"]
                    ti = isa(ti, AbstractFloat) ? ti : ti[forcing_type]
                    tf = config_dict["y_t_end_sec"]
                    tf = isa(tf, AbstractFloat) ? tf : tf[forcing_type]

                    y_true, Σ_obs, norm_vec_obs = get_obs(
                        ref_path,
                        [var_name],
                        zc_model;
                        ti = ti,
                        tf = tf,
                        Σ_const = const_noise_by_var,
                        z_score_norm = false,
                    )
                    sum_squared_errors +=
                        compute_ensemble_squared_error(data, y_true)
                end
            end

            if in(var_name, cal_vars)
                rmse_per_member = sqrt.(sum_squared_errors / n_vert_levels)
                valid_rmse = rmse_per_member[.!isnan.(rmse_per_member)]
                mean_rmse = mean(valid_rmse)
                min_rmse = minimum(valid_rmse)
                max_rmse = maximum(valid_rmse)
                rmse_std = std(valid_rmse)
            else
                mean_rmse = missing
                min_rmse = missing
                max_rmse = missing
                rmse_std = missing
            end
            push!(
                stats_df,
                (
                    iteration,
                    var_name,
                    mean(means[.!isnan.(means)]),
                    maximum(maxs[.!isnan.(maxs)]),
                    minimum(mins[.!isnan.(mins)]),
                    mean_rmse,
                    min_rmse,
                    max_rmse,
                    rmse_std,
                ),
            )
        end
        return stats_df
    end

    if !load_from_csv
        iterations_list = collect(iterations)
        stats_dfs = pmap(
            iteration -> process_iteration(
                iteration,
                output_dir,
                var_names,
                n_vert_levels,
                config_dict,
                z_max,
                cal_vars,
                const_noise_by_var,
                ref_paths,
                reduction,
                ensemble_size,
                batch_size,
            ),
            iterations_list,
        )

        stats_df = vcat(stats_dfs...)
        if save_as_csv
            CSV.write(joinpath(output_dir, "stats_df.csv"), stats_df)
        end

    elseif load_from_csv
        @info "Loading existing from CSV"
        stats_df = CSV.read(joinpath(output_dir, "stats_df.csv"), DataFrame)
    end

    stats_df = CSV.read(joinpath(output_dir, "stats_df.csv"), DataFrame)
    rmse_df = dropmissing(stats_df, [:rmse, :rmse_min, :rmse_max, :rmse_std])
    unique_vars = unique(rmse_df.var_name)
    n_vars = length(unique_vars)

    p = plot(layout = (n_vars, 1), size = (600, 400 * n_vars))

    for (i, var_name) in enumerate(unique_vars)
        df_var = rmse_df[rmse_df.var_name .== var_name, :]
        Plots.plot!(
            p[i],
            df_var.iteration,
            df_var.rmse,
            label = "Mean RMSE",
            lw = 2,
            marker = :o,
            color = :blue,
            ribbon = 1 .* df_var.rmse_std,
            fillalpha = 0.3,
            fillcolor = :blue,
        )
        Plots.plot!(
            p[i],
            df_var.iteration,
            df_var.rmse_min,
            linestyle = :dash,
            color = :black,
            label = "",
        )
        Plots.plot!(
            p[i],
            df_var.iteration,
            df_var.rmse_max,
            linestyle = :dash,
            color = :black,
            label = "",
        )
        Plots.xlabel!("Iteration")
        Plots.ylabel!("RMSE")
        Plots.title!(p[i], var_name)
    end
    savefig(joinpath(output_dir, "plots", plot_dir, "rmse_vs_iteration.pdf"))

    n_vars = length(var_names)
    p = plot(layout = (n_vars, 1), size = (800, 400 * n_vars))

    for (i, var_name) in enumerate(var_names)
        df_var = stats_df[stats_df.var_name .== var_name, :]
        Plots.plot!(
            p[i],
            df_var.iteration,
            df_var.mean,
            label = "Mean RMSE",
            lw = 2,
            marker = :o,
            color = :blue,
        )
        Plots.plot!(
            p[i],
            df_var.iteration,
            df_var.min,
            linestyle = :dash,
            color = :black,
            label = "",
        )
        Plots.plot!(
            p[i],
            df_var.iteration,
            df_var.max,
            linestyle = :dash,
            color = :black,
            label = "",
        )
        Plots.xlabel!("Iteration")
        Plots.ylabel!("Ranges")
        Plots.title!(p[i], var_name)
    end
    savefig(joinpath(output_dir, "plots", plot_dir, "stats_vs_iteration.pdf"))
end

main()
