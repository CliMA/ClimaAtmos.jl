"""
    ClimaCalibrate.analyze_iteration(
        interface::PerfectAtmosModelInterface,
        ekp,
        g_ensemble,
        prior,
        output_dir,
        iteration,
    )

Analyze the results of each iteration of the calibration.

This function only plots the constrained parameters and error of each iteration.
"""
function ClimaCalibrate.analyze_iteration(
    ::PerfectAtmosModelInterface,
    ekp,
    g_ensemble,
    prior,
    output_dir,
    iteration,
)
    plots_dir = joinpath(output_dir, "plots")
    isdir(plots_dir) || mkdir(plots_dir)
    plot_constrained_params_and_errors(plots_dir, ekp, prior)
    plot_ensemble(plots_dir, ekp)
    return nothing
end

"""
    plot_constrained_params_and_errors(output_dir, ekp, prior)

Plot the constrained parameters and errors from `ekp` and `prior` and save
them to `output_dir`.
"""
function plot_constrained_params_and_errors(output_dir, ekp, prior)
    dim_size = sum(length.(EKP.batch(prior)))
    fig = CairoMakie.Figure(size = ((dim_size + 1) * 500, 500))
    for i in 1:dim_size
        EKP.Visualize.plot_ϕ_over_iters(fig[1, i], ekp, prior, i)
    end
    EKP.Visualize.plot_error_over_iters(fig[1, dim_size + 1], ekp, error_metric = "loss")
    EKP.Visualize.plot_error_over_time(fig[1, dim_size + 2], ekp, error_metric = "loss")
    CairoMakie.save(joinpath(output_dir, "constrained_params_and_error.png"), fig)
    return nothing
end

"""
    plot_ensemble(output_dir, ekp)

Plot the columns of the G ensemble matrix ± standard deviation (from the
diagonal covariance matrix), and the true model data.

This function only supports `OutputVar`s that are one dimensional time series
data.
"""
function plot_ensemble(output_dir, ekp)
    curr_iter = EKP.get_N_iterations(ekp)
    # Size is number of variables by number of ensemble members
    g_ens_as_vars = reconstruct_g_ens(
        ekp,
        curr_iter,
    )

    obs = ClimaCalibrate.get_observations_for_nth_iteration(
        EKP.get_observation_series(ekp),
        curr_iter,
    )
    obs_as_vars = vcat(ObservationRecipe.reconstruct_vars.(obs)...)
    diag_covs_as_vars = vcat(ObservationRecipe.reconstruct_diag_cov.(obs)...)

    vars_upper = obs_as_vars .+ sqrt.(diag_covs_as_vars)
    vars_lower = obs_as_vars .- sqrt.(diag_covs_as_vars)

    # Number of columns of figure should scale with the number of variables
    ncols = size(g_ens_as_vars)[1]
    fig = CairoMakie.Figure(size = (600 * ncols, 250))
    for (var_idx, (var, var_lower, var_upper)) in
        enumerate(zip(obs_as_vars, vars_lower, vars_upper))
        # This is hard coded to only work for 1D time series data
        CairoMakie.lines(
            fig[1, var_idx],
            ClimaAnalysis.times(var),
            Vector(var.data);
            color = :black,
            linewidth = 2,
            axis = (;
                title = ClimaAnalysis.short_name(var),
                xlabel = "Time",
                ylabel = ClimaAnalysis.units(var),
            ),
        )

        CairoMakie.band!(fig[1, var_idx], ClimaAnalysis.times(var), Vector(var_lower.data),
            Vector(var_upper.data); color = :blue, alpha = 0.5)
    end

    for (var_idx, var) in enumerate(g_ens_as_vars)
        fig_idx = ((var_idx - 1) % length(obs_as_vars)) + 1
        CairoMakie.lines!(
            fig[1, fig_idx],
            ClimaAnalysis.times(var),
            Vector(var.data);
            color = :green,
            alpha = 0.35,
            linewidth = 2,
        )
    end
    CairoMakie.save(joinpath(output_dir, "ensemble_plot_of_iter_$(curr_iter).png"), fig)
end

"""
    reconstruct_g_ens(
        ekp::EKP.EnsembleKalmanProcess,
        it,
    )

Reconstruct the G ensemble matrix at the `it`th iteration as a matrix of
`OutputVar`s.
"""
function reconstruct_g_ens(
    ekp::EKP.EnsembleKalmanProcess,
    it,
)
    obs_series = EKP.get_observation_series(ekp)
    metadata = ClimaCalibrate.get_metadata_for_nth_iteration(
        obs_series,
        it,
    )
    all(m isa ClimaAnalysis.Var.Metadata for m in metadata) || error(
        "Getting the short names from an observation is only supported with metadata from ClimaAnalysis",
    )

    g_ens = EKP.get_g(ekp, it)
    num_cols = size(g_ens)[2]

    # Check if length of g ensemble is the same as the length of the data in the metadatas
    total_metadata_length =
        sum(ClimaAnalysis.flattened_length(m) for m in metadata)
    length(g_ens[:, 1]) != total_metadata_length && error(
        "Length of g_mean_final is not the same as the length of all the metadata",
    )

    # Reconstruct each OutputVar for every ensemble member (column of g_ens)
    ext = Base.get_extension(ClimaCalibrate, :ClimaCalibrateClimaAnalysisExt)
    minibatch_indices =
        ext.ObservationRecipe._get_minibatch_indices_for_nth_iteration(
            obs_series,
            EKP.get_N_iterations(ekp),
        )
    vars_per_ens = [
        map(metadata, minibatch_indices) do m, range
            ClimaAnalysis.unflatten(m, g_ens[range, col])
        end
        for col in 1:num_cols
    ]
    return hcat(vars_per_ens...)
end
