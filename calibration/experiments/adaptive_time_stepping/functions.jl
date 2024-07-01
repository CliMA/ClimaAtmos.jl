# includet("functions.jl")
using Pkg
Pkg.activate(".")
ENV["CLIMACOMMS_DEVICE"] = "CPUSingleThreaded"

import YAML
import Random
import ClimaParams as CP
import EnsembleKalmanProcesses as EKP
import NetCDF
import LinearAlgebra
import Distributions
import ClimaAtmos as CA
import ClimaCalibrate

using CairoMakie
using ClimaAnalysis
using Statistics
using JLD2
using LinearAlgebra
using BlockDiagonals
using Base.Threads


# Generate perfect model calibrations

function gen_perfect(config, job_id)
    config = YAML.load_file(config)
    config = CA.AtmosConfig(config; job_id=job_id)
    sim = CA.get_simulation(config)
    CA.solve_atmos!(sim)
    sim
end


function H(dir::String, short_name, iteration, member, noise)
    simdir = ClimaCalibrate.path_to_ensemble_member(dir, iteration, member)

    H_perf(simdir, short_name, noise = noise, cutoff=40, output_cov=false)
end

# run simulations 
function G(iteration, member; job_id = "ensemble", config = "configs/ens_prog_bomex.yml")
    # Run simulation
    sim_path = ClimaCalibrate.path_to_ensemble_member("output/$job_id", iteration, member)

    config_dict = deepcopy(YAML.load_file(config))
    config_dict["output_dir"] = sim_path
    # update parameters for the simulation
    push!(config_dict["toml"], joinpath(sim_path, "parameters.toml"))

    atmos_config = CA.AtmosConfig(config_dict; job_id=job_id)
    # atmos_config.toml_dict["entr_coeff"]["value"] = entr # change entrainment rate
    # println(atmos_config.toml_dict["entr_coeff"]["value"])
    edmf_sim = CA.get_simulation(atmos_config)

    CA.solve_atmos!(edmf_sim)    
end


function calibrate(n_iters, ens_size, obs_mean, obs_cov, prior, scheduler, job_id, short_name, noise)
    # initialize the calibration by setting up the members of the first iteration
    ClimaCalibrate.initialize(ens_size, obs_mean, obs_cov, prior, "output/$job_id"; scheduler=scheduler, verbose=true)
    println("Calibration Initialized...")
    # run the calibration
    for iteration in 0:n_iters
        # run ensemble of forward models 
        for member in 1:ens_size
            println("Running iteration $iteration, member $member on thread $(threadid()).")
            G(iteration, member, job_id=job_id)
        end
        G_ensemble = Array{Float64}(undef, size(obs_mean)..., ens_size)
        for m in 1:ens_size # parallel loop of ensemble members 
            # get member path
            member_path = ClimaCalibrate.path_to_ensemble_member("output/$job_id", iteration, m)
            simdir = SimDir(joinpath(member_path, "output_active"))
            try
                # calculate observational data
                # data = H("output/$job_id", iteration, m)
                G_ensemble[:, m] .= H("output/$job_id", short_name, iteration, m, noise)
            catch err
                @info "Error during observation map for ensemble member $m" err
                G_ensemble[:, m] .= NaN # fill with nans 
            end
        end
        # save observations where climacalibrate can find them 
        save_object(joinpath("output/$job_id/iteration_"* lpad("$iteration", 3, "0"), "G_ensemble.jld2") , G_ensemble)
        
    
        terminated = ClimaCalibrate.update_ensemble("output/$job_id", iteration, prior)

        # if !isnothing(terminated)
        #     break
        # end
        #println(terminated)
    end
end

function extract_array(params, idx, n_iters = n_iters)
    arr = Array{Float64}(undef, ens_size, n_iters+1)
    for i in 1:(n_iters+1)
        arr[:, i] = params[i][idx, :]
    end
    arr
end

function get_results(job_id; nvars = 2, n_iters = n_iters)
    #eki_objs = map(i -> load_object(joinpath(ClimaCalibrate.path_to_iteration("output/$job_id", i), "eki_file.jld2")), 0:(n_iters+1))
    eki_obj = load_object(joinpath(ClimaCalibrate.path_to_iteration("output/$job_id", n_iters+1), "eki_file.jld2"))
    # load parameters
    params = EKP.transform_unconstrained_to_constrained(prior, EKP.get_u(eki_obj))
    #print(params)

    # reshape params 
    ar = []
    for i in 1:nvars
        push!(ar, extract_array(params, i, n_iters))
    end
    ar 
end

function get_dt(job_id; nvars=2, n_iters = n_iters)
    eki_obj = load_object(joinpath(ClimaCalibrate.path_to_iteration("output/$job_id", n_iters+1), "eki_file.jld2"))
    # load parameters
    return cumsum(eki_obj.Δt)
end

function plot_2param(job_id, n_iters=n_iters, ens_size=ens_size)
    println(job_id, n_iters, ens_size)
    ar = get_results(job_id; nvars=2, n_iters = n_iters)
    times = get_dt(job_id; nvars=2, n_iters = n_iters)
    println("Plotting...")
    f = Figure(size = (1000, 500))

    ax1 = Axis(f[1,1],
        title = "Entrainment Coefficient Calibration",
        xlabel = "Sudo-time",
        ylabel = "Entrainment Coefficient")

    hlines!(ax1, .3, color = :blue, label="True Entrainment Rate")

    ax2 = Axis(f[1,2],
        title = "Eddy Viscosity Calibration",
        xlabel = "Sudo-time",
        ylabel = "Eddy Viscosity")

    hlines!(ax2, .14, color = :blue, label="True Eddy Viscosity")

    for i in 1:ens_size
        lines!(ax1, times, ar[1][i, :], color = (:red, 0.3))
        lines!(ax2, times, ar[2][i, :], color = (:red, 0.3))
    end
    println("Saving plot...")
    # save in two locations to ensure no overwriting
    save("plots/2param_calibration_$job_id.png", f)
    save("output/$job_id/2param_calibration.png", f)
    jldsave("output/$job_id/2param_calibration.jld2"; ar)
    f
end