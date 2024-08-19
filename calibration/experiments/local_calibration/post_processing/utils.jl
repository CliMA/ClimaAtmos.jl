import ClimaAnalysis: SimDir, get, slice, average_xy
import LinearAlgebra: I
import EnsembleKalmanProcesses as EKP
import Statistics: var, mean
import ClimaCalibrate as CAL
import JLD2
using CairoMakie

params_true = [.14, 1, .3, .22] # 4th position .0001, entr_inv_tau removed

function loss_plot(eki, observations; 
    n_iters = 9,
    n_metrics = 9,
    config_dict=config_dict)
    loss_vals = EKP.get_g(eki)
    all_loss = []
    for iteration in 1:n_iters
        vals = []
            for metric in 1:n_metrics
                vmean = mean(filter(!isnan, ((loss_vals[iteration] .- observations) .^2)[metric,:]))
                append!(vals, sqrt(vmean))
            end
        append!(all_loss, [vals])
        end
    all_loss
        
    fig = Figure(size = (800, 600))
    num_per_row = 3
    for metric in 1:n_metrics
        row = div(metric-1, num_per_row) + 1
        col = mod(metric-1, num_per_row) + 1
        vals = []
        for iteration in 1:n_iters
            append!(vals, all_loss[iteration][metric])
        end
        ax = Axis(fig[row, col], title = config_dict["y_var_names"][metric])
        lines!(ax, vals)
    end
    Label(fig[0,:], "Individual Observation Loss over Iteration")
    fig
end




function gen_obs(simdir)
    simdir = SimDir(simdir)
    process_member_data(simdir; y_names = config_dict["y_var_names"], t_start = config_dict["g_t_start_sec"], t_end = config_dict["g_t_end_sec"])
end

function plot_start_end_distributions(eki, config_dict, observations, variances, observations_true)
    fig = Figure(size = (1000, 600))
    loss_vals = EKP.get_g(eki)
    num_per_row = 3
    for i in 1:9
        row = div(i-1, num_per_row) + 1
        col = mod(i-1, num_per_row) + 1

        # get data 
        prior_dist = loss_vals[1][i, :]
        end_dist = filter(!isnan, loss_vals[end][i, :])
        gmin = minimum(vcat(prior_dist, end_dist))
        gmax = maximum(vcat(prior_dist, end_dist))
        bins = range(gmin, gmax, length = 30)

        # get perfect model run data
        obs = observations[i]
        obs_true = observations_true[i]
        vari = variances[i]
        
        
        ax = Axis(fig[row, col], title = config_dict["y_var_names"][i],
            limits = ((minimum([gmin, obs*.997, true_obs[i]*.997]), maximum([gmax, obs*1.003, true_obs[i]*1.003])), nothing))
        

        hist!(ax, prior_dist, bins = bins, color = (:blue, 0.75), label = "Initial Distribution")
        hist!(ax, end_dist, bins = bins, color = (:orange, 0.75), label = "Final Distribution")


        vlines!(ax, obs, color=:red, label = "Noisy Observation")
        vlines!(ax, true_obs[i], color="green", label = "True Observation")

        # shade polygon for 2x standard deviations 

        n_vari = 2
        poly!(ax, [obs_true-n_vari*vari, obs_true+n_vari*vari, obs_true + n_vari*vari, obs_true - n_vari*vari] ,
                    [0, 0, 60, 60],
                    color=(:red, .2))
        if i == 1 # add one legend for all 
            legend = Legend(fig, ax, orientation = :vertical, tellheight = false)
            fig[1, num_per_row + 1] = legend
        end
    end
    fig
end


function plot_parameters(eki, prior, params_true)
    phi = cat(EKP.get_ϕ(prior, eki)..., dims = 3)
    time = vcat(0, cumsum(eki.Δt))
    fig = Figure(size = (1000, 800))
    num_per_row = 3
    for (i, name) in enumerate(EKP.get_name(prior))
        row = div(i-1, num_per_row) + 1
        col = mod(i-1, num_per_row) + 1
        ax = Axis(fig[row, col], title = name)
        for ens in 1:100
            lines!(ax, time, phi[i, ens, :], color= :red)
        end    
        hlines!(ax, params_true[i])
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