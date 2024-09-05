import ClimaAnalysis: SimDir, get, slice, average_xy, window, average_time, units
import LinearAlgebra: I
import EnsembleKalmanProcesses as EKP
import Statistics: var, mean
import ClimaCalibrate as CAL
import JLD2
using CairoMakie

params_true = [.14, 1, .3, .22] # 4th position .0001, entr_inv_tau removed
config_dict = YAML.load_file(joinpath(simdir, "experiment_config.yml"))
output_dir = config_dict["output_dir"]

len_dict = Dict(
    "thetaa" => 60,
    "hus" => 60,
    "husv" => 60,
    "clw" => 60,
    "lwp" => 1,
    "prw" => 1,
    "clwvi" => 1,
    "clvi" => 1,
    "husvi" => 1,
    "hurvi" => 1,
    "rlut" => 1,
    "rlutcs" => 1,
    "rsut" => 1,
    "rsutcs" =>1,
)

function loss_plot(eki, config_dict)
    time = eki.Δt
    names = config_dict["y_var_names"]
    n_iters = length(time)
    n_metrics = length(names)

    # get values
    loss_vals = EKP.get_g(eki)
    observations = JLD2.load_object(config_dict["observations"])
    noise = diag(JLD2.load_object(config_dict["noise"]))

    f_diagnostics = JLD2.jldopen(
        joinpath(config_dict["output_dir"], "norm_factors.jld2"),
        "r+",
    )["norm_factors_dict"]
    #f_diagnostics = norm_factors_old


    #all_loss = []
    fig = Figure(size = (800, 600))
    num_per_row = 3
    sidx = 1
    for i in 1:n_metrics
        name = config_dict["y_var_names"][i]
        len = len_dict[name]
        simdir = SimDir(joinpath(output_dir, "iteration_001", "member_001", "output_active"))
        # get sample variable for units and z dimension 
        vari = get(simdir; short_name = name)
        unts = units(vari)
        #zdims = vari.dims["z"]

        μ, σ² = f_diagnostics[name][1], f_diagnostics[name][2]

        vals = []
        for iter in 1:n_iters
            # get observations and rescale to physical space
            ar_vals = loss_vals[iter][sidx:(sidx + len-1), :] .* σ² .+ μ
            obs = observations[sidx:(sidx + len-1), :]  .* σ² .+ μ
            vmean = mean(filter(!isnan, ((ar_vals .- obs) .^2)))
            append!(vals, sqrt(vmean))
        end
        row = div(i-1, num_per_row) + 1
        col = mod(i-1, num_per_row) + 1
        ax = nothing 
        if (n_metrics - i) < 3
            ax = Axis(fig[row, col], title = name, xlabel = "Iteration", ylabel = unts)
        else
            ax = Axis(fig[row, col], title = name, ylabel = unts)
        end

        lines!(ax, vals)
        sidx +=len
    end
    Label(fig[0,:], "Observation RMSE over Iteration")
    fig
end

function loss_plot_old(eki, observations; 
    n_iters = 9,
    n_metrics = 8,
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

function plot_start_end_distributions(eki, config_dict)
    fig = Figure(size = (1000, 900))

    n_iters = length(eki.Δt)
    n_metrics = length(config_dict["y_var_names"])

    observations = JLD2.load_object(config_dict["observations"])
    noise = diag(JLD2.load_object(config_dict["noise"]))

    sidx = 1
    loss_vals = EKP.get_g(eki)
    num_per_row = 3
    for i in 1:n_metrics
        row = div(i-1, num_per_row) + 1
        col = mod(i-1, num_per_row) + 1


        # get data for this metric or profile
        name = config_dict["y_var_names"][i]
        f_diagnostics = JLD2.jldopen(
            joinpath(config_dict["output_dir"], "norm_factors.jld2"),
            "r+",
        )["norm_factors_dict"]
        #f_diagnostics = norm_factors_old

        μ, σ² = f_diagnostics[name][1], f_diagnostics[name][2]

        len = len_dict[name]
        prior_dist = loss_vals[1][sidx:(sidx + len-1), :] .* σ² .+ μ
        end_dist = loss_vals[n_iters][sidx:(sidx + len-1), :] .* σ² .+ μ

        simdir = SimDir(joinpath(output_dir, "iteration_001", "member_001", "output_active"))
        # get sample variable for units and z dimension 
        vari = get(simdir; short_name = name)
        unts = units(vari)
        
        if size(prior_dist)[1] > 1
            zdims = vari.dims["z"]

            end_dist = end_dist[:,vec(.!any(isnan, end_dist; dims = 1))]

            ax = Axis(fig[row, col], title = name, xlabel = unts, ylabel = "height(m)")

            # then its a line plot
            for k in 1:size(prior_dist)[2]
                lines!(ax, prior_dist[1:30, k], zdims[1:30], color = (:blue, 0.75))
            end 
            for k in 1:size(end_dist)[2]
                lines!(ax, end_dist[1:30, k], zdims[1:30], color = (:orange, 0.75))
            end
            obs = observations[sidx:(sidx + 30-1), :] .* σ² .+ μ
            println(size(obs[:, 1]))
            lines!(ax, obs[:, 1], zdims[1:30], color = (:red, 0.75))
        else
            #return vec(hcat(prior_dist, end_dist))
            catall = vec(hcat(prior_dist, end_dist))
            gmin = minimum(catall[.!isnan.(catall)]) 
            gmax = maximum(catall[.!isnan.(catall)])

            bins = range(gmin, gmax, length = 30)
            ax = Axis(fig[row, col], title = config_dict["y_var_names"][i], ylabel = "Density", xlabel = unts)
            obs = observations[sidx] .* σ² .+ μ
            limits = ((minimum([gmin, obs*.997]), maximum([gmax, (obs*1.003)]), nothing))
            #println(size(prior_dist[1, :]))

            hist!(ax, prior_dist[1,:], bins = bins, color = (:blue, 0.75), label = "Initial Distribution")
            hist!(ax, end_dist[1,:], bins = bins, color = (:orange, 0.75), label = "Final Distribution")
            vlines!(ax, obs, color=:red, label = "Observation")


            nsd = 2
            poly!(ax, [obs-nsd*sqrt(σ²), obs+nsd*sqrt(σ²), obs + nsd*sqrt(σ²), obs - nsd*sqrt(σ²)] ,
                        [0, 0, 40, 40],
                        color=(:red, .2))
            if i == 8 # add one legend for all 
                legend = Legend(fig, ax, orientation = :vertical, tellheight = false)
                fig[1, num_per_row + 1] = legend
            end

        end

        sidx +=len
    end
    fig
end

function plot_start_end_distributions(eki, 
    config_dict, 
    observations, 
    variances, 
    observations_true)
    fig = Figure(size = (1000, 600))
    loss_vals = EKP.get_g(eki)
    num_per_row = 3
    for i in 1:8
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
        sd = sqrt(variances[i])
        
        
        ax = Axis(fig[row, col], title = config_dict["y_var_names"][i],
            limits = ((minimum([gmin, obs*.997, true_obs[i]*.997]), maximum([gmax, obs*1.003, true_obs[i]*1.003])), nothing))
        

        hist!(ax, prior_dist, bins = bins, color = (:blue, 0.75), label = "Initial Distribution")
        hist!(ax, end_dist, bins = bins, color = (:orange, 0.75), label = "Final Distribution")


        vlines!(ax, obs, color=:red, label = "Noisy Observation")
        vlines!(ax, true_obs[i], color="green", label = "True Observation")

        # shade polygon for 2x standard deviations 

        nsd = 2
        poly!(ax, [obs_true-nsd*sd, obs_true+nsd*sd, obs_true + nsd*sd, obs_true - nsd*sd] ,
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