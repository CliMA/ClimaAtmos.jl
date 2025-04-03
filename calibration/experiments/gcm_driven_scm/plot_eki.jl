import TOML, YAML
import JLD2
using Distributions
import EnsembleKalmanProcesses as EKP
using EnsembleKalmanProcesses.ParameterDistributions
using EnsembleKalmanProcesses.TOMLInterface
import ClimaCalibrate as CAL
using Plots
using LinearAlgebra
using DataFrames

# output_dir = "output/exp_1"
# output_dir= "/groups/esm/cchristo/climaatmos_scm_calibrations/output_ml_mix/exp_40"
# output_dir= "/central/scratch/cchristo/output_ml_mix2/exp_6"
output_dir = "/central/scratch/cchristo/output_ml_mix2/exp_8"
iterations = 0:5
# iterations = nothing

include("helper_funcs.jl")

const config_dict =
    YAML.load_file(joinpath(output_dir, "configs", "experiment_config.yml"))
const n_iterations = config_dict["n_iterations"]

# plotting output dirs
plot_dir = joinpath(output_dir, "plots", "param_plots")
mkpath(plot_dir)
plot_dir_y_vec = joinpath(output_dir, "plots", "y_vec_plots")
mkpath(plot_dir_y_vec)


if isnothing(iterations)
    iterations = collect(0:(n_iterations - 1))
end

const prior = CAL.get_prior(joinpath(output_dir, "configs", "prior.toml"))


# const pretrained_nn_path = config_dict["pretrained_nn_path"]
# prior_path = joinpath(output_dir, "configs", "prior.toml")
# prior = create_prior_with_nn(prior_path, pretrained_nn_path)


function inv_variance_weighted_loss(
    y_diff::Matrix{Float64},
    gamma_i::Matrix{Float64},
)
    var_vec = diag(gamma_i)
    inv_var_vec = 1.0 ./ var_vec
    losses = [mean(y_diff[:, j] .^ 2 .* inv_var_vec) for j in 1:size(y_diff, 2)]
    return mean(losses), std(losses), minimum(losses)
end


function mean_loss(y_diff::Matrix{Float64})
    losses = [mean(y_diff[:, j] .^ 2) for j in 1:size(y_diff, 2)]
    return mean(losses), std(losses), minimum(losses)
end

loss_results = Dict(
    :mean_loss_avg => Float64[],
    :mean_loss_std => Float64[],
    :var_weighted_loss_avg => Float64[],
    :var_weighted_loss_std => Float64[],
    :min_loss_avg => Float64[],
    :var_weighted_min_loss_avg => Float64[],
)


for iteration in iterations

    @info iteration

    iter_path = CAL.path_to_iteration(output_dir, iteration)
    eki = JLD2.load_object(joinpath(iter_path, "eki_file.jld2"))

    iter_path_p1 = CAL.path_to_iteration(output_dir, iteration + 1)
    eki_p1 = JLD2.load_object(joinpath(iter_path_p1, "eki_file.jld2"))

    Δt = EKP.get_Δt(eki)
    y_i = EKP.get_obs(eki)
    gamma_i = EKP.get_obs_noise_cov(eki)

    g = EKP.get_g_final(eki_p1)

    nan_columns_count = sum(any(isnan, g, dims = 1))

    # Find successful simulations
    non_nan_columns_indices = findall(x -> !x, vec(any(isnan, g, dims = 1)))
    g_non_nan = g[:, non_nan_columns_indices]

    y_diff = y_i .- g_non_nan

    u = EKP.get_u_final(eki)
    N_obs = size(g, 1)
    cov_init = cov(u, dims = 2)

    avg_loss, std_loss, min_loss = mean_loss(y_diff)
    push!(loss_results[:mean_loss_avg], avg_loss)
    push!(loss_results[:mean_loss_std], std_loss)
    push!(loss_results[:min_loss_avg], min_loss)
    avg_var_loss, std_var_loss, var_min_loss =
        inv_variance_weighted_loss(y_diff, gamma_i)
    push!(loss_results[:var_weighted_loss_avg], avg_var_loss)
    push!(loss_results[:var_weighted_loss_std], std_var_loss)
    push!(loss_results[:var_weighted_min_loss_avg], var_min_loss)

    plt = plot()
    for i in 1:size(g_non_nan, 2)
        plot!(plt, 1:size(g_non_nan, 1), g_non_nan[:, i], label = false)
    end
    plot!(plt, 1:size(y_i, 1), y_i, label = "Observations", color = :black)

    noise_std = sqrt.(diag(gamma_i))
    x_range = 1:size(g_non_nan, 1)
    plot!(
        plt,
        x_range,
        y_i .+ 2 * noise_std,
        fillrange = y_i .- 2 * noise_std,
        fillalpha = 0.2,
        label = "Noise range",
        color = :gray,
        linewidth = 0.1,
    )

    ylims!(plt, -5.0, 8.0)
    savefig(plt, joinpath(plot_dir_y_vec, "eki_y_vs_g_iter_$(iteration).pdf"))

end


### print best particles and loss
loss_min_inds, loss_min_vals = get_loss_min(output_dir, iterations[end])
df = DataFrame(ParticleNumber = loss_min_inds, Loss = loss_min_vals)
display(df)

### Plot EKI loss evolution
function plot_loss(
    iter_range,
    avg_loss,
    std_loss,
    min_loss,
    title_str,
    file_name,
)
    plt = plot(
        iter_range,
        avg_loss,
        label = "$(title_str) Avg",
        lw = 2,
        xlabel = "Iteration",
        ylabel = "Loss",
        title = title_str,
        marker = :o,
        color = :black,
    )
    upper_2std = avg_loss .+ 2 .* std_loss
    lower_2std = avg_loss .- 2 .* std_loss
    plot!(
        plt,
        iter_range,
        upper_2std,
        fillrange = lower_2std,
        label = "±2 STD Shading",
        fillalpha = 0.3,
        lw = 0,
        color = :blue,
    )
    plot!(
        plt,
        iter_range,
        min_loss,
        linestyle = :dash,
        color = :black,
        label = "Min $(title_str)",
    )
    savefig(plt, joinpath(plot_dir_y_vec, file_name))
end

iterations_range = 1:length(iterations)

plot_loss(
    iterations_range,
    loss_results[:mean_loss_avg],
    loss_results[:mean_loss_std],
    loss_results[:min_loss_avg],
    "Mean Loss",
    "mean_loss_vs_iteration.png",
)
plot_loss(
    iterations_range,
    loss_results[:var_weighted_loss_avg],
    loss_results[:var_weighted_loss_std],
    loss_results[:var_weighted_min_loss_avg],
    "Variance-Weighted Loss",
    "var_weighted_loss_vs_iteration.png",
)


### Plot parameter evolution
iter_path = CAL.path_to_iteration(output_dir, iterations[end])
eki = JLD2.load_object(joinpath(iter_path, "eki_file.jld2"))
u_all = EKP.get_u(eki)

phi_all = EKP.transform_unconstrained_to_constrained(prior, u_all)
phi_all_stacked = cat(phi_all...; dims = 3)


names = []
for i in 1:length(prior.name)
    if isa(prior.distribution[i], EKP.Parameterized)
        push!(names, prior.name[i])
    elseif isa(prior.distribution[i], EKP.VectorOfParameterized)
        for j in 1:length(prior.distribution[i].distribution)
            push!(names, prior.name[i] * "_$j")
        end
    end
end



for param_i in 1:size(phi_all_stacked, 1)
    param_name = names[param_i]
    data = phi_all_stacked[param_i, :, :]

    mean_vals = mean(data, dims = 1)
    max_vals = maximum(data, dims = 1)
    min_vals = minimum(data, dims = 1)
    std_vals = std(data, dims = 1)

    mean_vals = vec(mean_vals)
    max_vals = vec(max_vals)
    min_vals = vec(min_vals)
    std_vals = vec(std_vals)

    upper_2std = mean_vals .+ 2 .* std_vals
    lower_2std = mean_vals .- 2 .* std_vals

    iterations = 1:size(phi_all_stacked, 3)
    plot(iterations, mean_vals, label = "Mean", lw = 2, color = :black)
    plot!(
        iterations,
        max_vals,
        label = "Max",
        lw = 1,
        linestyle = :dash,
        color = :black,
    )
    plot!(
        iterations,
        min_vals,
        label = "Min",
        lw = 1,
        linestyle = :dash,
        color = :black,
    )

    plot!(
        iterations,
        upper_2std,
        fillrange = lower_2std,
        label = "±2 STD Shading",
        fillalpha = 0.3,
        lw = 0,
        color = :blue,
    )

    xlabel!("Iteration")
    ylabel!("$param_name")
    title!("Parameter evol: $param_name")


    savefig(joinpath(plot_dir, "param_$(param_name)_stats.png"))

end
