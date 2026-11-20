using TOML, YAML, JLD2, Distributions
import EnsembleKalmanProcesses as EKP
using EnsembleKalmanProcesses.ParameterDistributions
using EnsembleKalmanProcesses.TOMLInterface
import ClimaCalibrate as CAL
using Plots, LinearAlgebra, DataFrames, BSON

include("helper_funcs.jl")

output_dir1 = "/resnick/groups/esm/jschmitt/climaatmos_scm_calibrations/output_cf_ml_v1/exp3_quadrature_sgs_physonly"
output_dir2 = "/resnick/groups/esm/jschmitt/climaatmos_scm_calibrations/output_cf_ml_v1/exp2_calphys"
output_dir3 = "/resnick/groups/esm/jschmitt/climaatmos_scm_calibrations/output_cf_ml_v1/exp4_hess_cov_nn_prior_tenth"
output_dirs = Dict("exp3" => output_dir1, "exp2" => output_dir2, "exp4" => output_dir3)

################################################################################
# Helper functions
################################################################################

function inv_variance_weighted_loss(y_diff, gamma_i)
    var_vec = diag(gamma_i)
    inv_var_vec = 1.0 ./ var_vec
    losses = [mean(y_diff[:, j] .^ 2 .* inv_var_vec) for j in 1:size(y_diff, 2)]
    return mean(losses), std(losses), minimum(losses)
end

function mean_loss(y_diff)
    losses = [mean(y_diff[:, j] .^ 2) for j in 1:size(y_diff, 2)]
    return mean(losses), std(losses), minimum(losses)
end

"""
Compute all losses for a given output_dir
Returns: Dict of loss vectors
"""
function compute_losses(output_dir::String; iterations=nothing)

    config_dict = YAML.load_file(joinpath(output_dir, "configs", "experiment_config.yml"))
    n_iterations = config_dict["n_iterations"]

    if iterations === nothing
        iterations = collect(0:(n_iterations - 1))
    end

    loss_results = Dict(
        :mean_loss_avg => Float64[],
        :mean_loss_std => Float64[],
        :min_loss_avg => Float64[],
        :var_weighted_loss_avg => Float64[],
        :var_weighted_loss_std => Float64[],
        :var_weighted_min_loss_avg => Float64[],
    )

    for iteration in iterations
        @info "Processing $(output_dir) iteration $iteration"

        iter_path = CAL.path_to_iteration(output_dir, iteration)
        iter_path_next = CAL.path_to_iteration(output_dir, iteration+1)

        eki = JLD2.load_object(joinpath(iter_path, "eki_file.jld2"))
        eki_next = JLD2.load_object(joinpath(iter_path_next, "eki_file.jld2"))

        y_i = EKP.get_obs(eki)
        gamma_i = EKP.get_obs_noise_cov(eki)
        g = EKP.get_g_final(eki_next)

        # Filter invalid columns
        non_nan_cols = findall(x -> !x, vec(any(isnan, g, dims=1)))
        g_non_nan = g[:, non_nan_cols]

        y_diff = y_i .- g_non_nan

        # mean loss
        avg_loss, std_loss, min_loss = mean_loss(y_diff)
        push!(loss_results[:mean_loss_avg], avg_loss)
        push!(loss_results[:mean_loss_std], std_loss)
        push!(loss_results[:min_loss_avg], min_loss)

        # variance-weighted loss
        avg_var_loss, std_var_loss, min_var_loss = inv_variance_weighted_loss(y_diff, gamma_i)
        push!(loss_results[:var_weighted_loss_avg], avg_var_loss)
        push!(loss_results[:var_weighted_loss_std], std_var_loss)
        push!(loss_results[:var_weighted_min_loss_avg], min_var_loss)
    end

    return loss_results
end

################################################################################
# Compute losses for both experiments
################################################################################

loss1 = compute_losses(output_dir1)
loss2 = compute_losses(output_dir2)
loss3 = compute_losses(output_dir3)

iterations_range1 = 1:length(loss1[:mean_loss_avg])
iterations_range2 = 1:length(loss2[:mean_loss_avg])
iterations_range3 = 1:length(loss3[:mean_loss_avg])

################################################################################
# Overlay Plotting
################################################################################

function plot_loss_overlay(iter1, loss_dict1, label1,
                           iter2, loss_dict2, label2,
                           iter3, loss_dict3, label3,
                           key_avg, key_std, key_min,
                           title_str, file_name)

    plt = plot(
        iter1, loss_dict1[key_avg],
        label = "$(label1) Avg",
        lw = 2, marker = :o,
        xlabel = "Iteration", ylabel = "Loss",
        grid = false,
        legend = :topright,
        legendframe = :none,
        foreground_color_legend = nothing,
        # title = title_str,
    )
    ylims!(plt, 0, Inf)
    plot!(plt, iter2, loss_dict2[key_avg],
        label = "$(label2) Avg",
        lw = 2, marker = :o
    )

    plot!(plt, iter3, loss_dict3[key_avg],
        label = "$(label3) Avg",
        lw = 2, marker = :o
    )

    # Min curves
    plot!(plt, iter1, loss_dict1[key_min],
        linestyle = :dash, label = "$(label1) Min")

    plot!(plt, iter2, loss_dict2[key_min],
        linestyle = :dash, label = "$(label2) Min")

    plot!(plt, iter3, loss_dict3[key_min],
        linestyle = :dash, label = "$(label3) Min")

    savefig(plt, file_name)
    return plt
end


################################################################################
# Make both overlay plots
################################################################################

plot_loss_overlay(
    iterations_range1, loss1, "Quadrature",
    iterations_range2, loss2, "NN Uncalibrated",
    iterations_range3, loss3, "NN (Hessian Prior)",
    :mean_loss_avg, :mean_loss_std, :min_loss_avg,
    "Mean Loss", "mean_loss_overlay2.pdf"
)

plot_loss_overlay(
    iterations_range1, loss1, "Quadrature",
    iterations_range2, loss2, "NN Uncalibrated",
    iterations_range3, loss3, "NN (Hessian Prior)",
    :var_weighted_loss_avg, :var_weighted_loss_std, :var_weighted_min_loss_avg,
    "Variance-Weighted Loss", "var_weighted_loss_overlay2.pdf"
)

println("Overlay plots written.")


# print summary statistics 
(loss1[:mean_loss_avg] .- loss3[:mean_loss_avg]) ./loss1[:mean_loss_avg]