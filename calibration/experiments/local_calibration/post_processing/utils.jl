import ClimaAnalysis: SimDir, get, slice, average_xy
import LinearAlgebra: I
import EnsembleKalmanProcesses as EKP
import Statistics: var, mean
import ClimaCalibrate as CAL
import JLD2
using CairoMakie



function get_parameters(iteration, output_dir)
    # open files
    eki_filepath = joinpath(CAL.path_to_iteration(output_dir, iteration), "eki_file.jld2")
    eki = JLD2.load_object(eki_filepath)
    prior_path = joinpath("../prior.toml")
    prior = CAL.get_prior(prior_path)
    # process EKI object to get u
    params = EKP.transform_unconstrained_to_constrained(prior, EKP.get_u(eki))
    params = vcat(params...)
    names = EKP.get_name(prior)
    params = reshape(params, length(names), eki.N_ens, iteration+1)
    params = permutedims(params, (3, 1, 2));

    return params, EKP.get_g(eki), eki.Δt, names
end

function plot_parameters(iteration, output_dir)
    (params, g, dt, names) = get_parameters(iteration, output_dir)

    fig = Figure(size = (800, 600))

    num_per_row = 3
    for i in 1:length(names)
        row = div(i-1, num_per_row) + 1
        col = mod(i-1, num_per_row) + 1
        
        ax = Axis(fig[row, col], title = names[i], xlabel = "Iteration")
        
        for j in 1:size(params, 3)
            lines!(ax, 1:size(params, 1), params[:, i, j], color = :red, alpha = 0.3)
        end
    end
    fig
end


function process_member_data(
    simdir;
    y_names,
    reduction = "inst",
    t_start,
    t_end,
    # norm_vec_obs = [0.0, 1.0],
    # normalize = true,
)

    g = Float64[]

    for (i, y_name) in enumerate(y_names)
        var_i = get(simdir; short_name = y_name, reduction = reduction)
        sim_t_end = var_i.dims["time"][end]

        if sim_t_end < 0.95 * t_end
            throw(ErrorException("Simulation failed."))
        end
        # take time-mean
        var_i_ave = average_time(
            window(var_i, "time", left = t_start, right = sim_t_end),
        )

        y_var_i = slice(var_i_ave, x = 1, y = 1).data
        # if normalize
        #     y_μ, y_σ = norm_vec_obs[i, 1], norm_vec_obs[i, 2]
        #     y_var_i = (y_var_i .- y_μ) ./ y_σ
        # end

        append!(g, mean(y_var_i))
    end

    return g
end