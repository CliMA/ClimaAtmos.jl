# srun --time=01:00:00  --ntasks=16 --cpus-per-task=1 --output=slurm_workers.txt julia --project=calibration/test/slurm_workers.jl
using Distributed, ClusterManagers
import ClimaCalibrate as CAL
import ClimaAnalysis: SimDir, get, slice, average_xy

function create_worker_pool()    
    addprocs(
        SlurmManager(10),
        t = "00:20:00",
        cpus_per_task = 1,
        exeflags = "--project=$(Base.active_project())"
    )
    return WorkerPool(workers())
end

function CAL.observation_map(iteration)
    single_member_dims = (1,)
    G_ensemble = Array{Float64}(undef, single_member_dims..., ensemble_size)

    for m in 1:ensemble_size
        member_path = CAL.path_to_ensemble_member(output_dir, iteration, m)
        simdir_path = joinpath(member_path, "output_active")
        if isdir(simdir_path)
            simdir = SimDir(simdir_path)
            G_ensemble[:, m] .= process_member_data(simdir)
        else
            G_ensemble[:, m] .= NaN
        end
    end
    return G_ensemble
end

function process_member_data(simdir::SimDir)
    isempty(simdir.vars) && return NaN
    rsut =
        get(simdir; short_name = "rsut", reduction = "average", period = "30d")
    return slice(average_xy(rsut); time = 30).data
end

function run_iteration(config, iter, worker_pool)
    # Create a channel to collect results
    results = Channel{Any}(config.ensemble_size)
    
    @sync begin
        for m in 1:config.ensemble_size
            @async begin
                # Get a worker from the pool
                worker = take!(worker_pool)
                try
                    @info "Running member $m on worker $worker"
                    # Run the model and put result in channel
                    model_config = CAL.set_up_forward_model(m, iter, config)
                    result = remotecall_fetch(CAL.run_forward_model, worker, model_config)
                        
                    # TODO: This doesn't work
                    # result = remotecall_fetch(worker) do 
                    #     model_config = CAL.set_up_forward_model(m, iter, config)
                    #     CAL.run_forward_model(model_config)
                    # end
                    put!(results, (m, result))
                catch e
                    @error "Error running member $m" exception=e
                    put!(results, (m, e))
                finally
                    # Always return worker to pool
                    put!(worker_pool, worker)
                end
            end
        end
    end

    # Collect all results
    ensemble_results = Dict{Int, Any}()
    for _ in 1:config.ensemble_size
        m, result = take!(results)
        if result isa Exception
            @error "Member $m failed" error=result
        else
            ensemble_results[m] = result
        end
    end
    results = values(ensemble_results)
    all(isa.(results, Exception)) && error("Full ensemble for iter $iter failed")

    # Process results
    G_ensemble = CAL.observation_map(iter)
    CAL.save_G_ensemble(config, iter, G_ensemble)
    CAL.update_ensemble(config, iter)
    iter_path = CAL.path_to_iteration(config.output_dir, iter)
    return JLD2.load_object(joinpath(iter_path, "eki_file.jld2"))
end

function calibrate(config, worker_pool)
    CAL.initialize(config)
    
    for iter in 0:config.n_iterations
        (; time, value) = @timed run_iteration(config, iter, worker_pool)
        @info "Iteration $iter time: $time"
    end
end

worker_pool = create_worker_pool()

@everywhere println(string(myid()))

@everywhere begin
    import ClimaCalibrate as CAL
    import ClimaAtmos as CA

    experiment_dir = joinpath(pkgdir(CA), "calibration", "test")
    model_interface = joinpath(pkgdir(CA), "calibration", "model_interface.jl")
    output_dir = "calibration_end_to_end_test"
    include(model_interface)
    ensemble_size = 50
    obs_path = joinpath(experiment_dir, "observations.jld2")
end

if !isfile(obs_path)
    import JLD2
    @info "Generating observations"
    atmos_config = CA.AtmosConfig(joinpath(experiment_dir, "model_config.yml"))
    simulation = CA.get_simulation(atmos_config)
    CA.solve_atmos!(simulation)
    observations = Vector{Float64}(undef, 1)
    observations .= process_member_data(SimDir(simulation.output_dir))
    JLD2.save_object(obs_path, observations)
end

# Initialize experiment data
@everywhere begin
    import JLD2
    import LinearAlgebra: I
    astronomical_unit = 149_597_870_000
    observations = JLD2.load_object(obs_path)
    noise = 0.1 * I
    n_iterations = 10
    prior = CAL.get_prior(joinpath(experiment_dir, "prior.toml"))

    config = CAL.ExperimentConfig(;
        n_iterations,
        ensemble_size,
        observations,
        noise,
        output_dir,
        prior,
    )
end

@everywhere run_model(m, iter, experiment_dir) = CAL.run_forward_model(CAL.set_up_forward_model(m, iter, experiment_dir))

calibrate(config, worker_pool)

import EnsembleKalmanProcesses as EKP
import Statistics: var, mean
using Test

function minimal_eki_test(eki)
    params = EKP.get_ϕ(prior, eki)
    spread = map(var, params)

    # Spread should be heavily decreased as particles have converged
    @test last(spread) / first(spread) < 0.1
    # Parameter should be close to true value
    @test mean(last(params)) ≈ astronomical_unit rtol = 0.02
end

import CairoMakie

function scatter_plot(eki::EKP.EnsembleKalmanProcess)
    f = CairoMakie.Figure(resolution = (800, 600))
    ax = CairoMakie.Axis(
        f[1, 1],
        ylabel = "Parameter Value",
        xlabel = "Top of atmosphere radiative SW flux",
    )

    g = vec.(EKP.get_g(eki; return_array = true))
    params = vec.((EKP.get_ϕ(prior, eki)))

    for (gg, uu) in zip(g, params)
        CairoMakie.scatter!(ax, gg, uu)
    end

    CairoMakie.hlines!(ax, [astronomical_unit], linestyle = :dash)
    CairoMakie.vlines!(ax, observations, linestyle = :dash)

    output = joinpath(output_dir, "scatter.png")
    CairoMakie.save(output, f)
    return output
end

function param_versus_iter_plot(eki::EKP.EnsembleKalmanProcess)
    f = CairoMakie.Figure(resolution = (800, 600))
    ax = CairoMakie.Axis(
        f[1, 1],
        ylabel = "Parameter Value",
        xlabel = "Iteration",
    )
    params = EKP.get_ϕ(prior, eki)
    for (i, param) in enumerate(params)
        CairoMakie.scatter!(ax, fill(i, length(param)), vec(param))
    end

    CairoMakie.hlines!(ax, [astronomical_unit]; color = :red, linestyle = :dash)

    output = joinpath(output_dir, "param_vs_iter.png")
    CairoMakie.save(output, f)
    return output
end

eki = JLD2.load_object(joinpath(output_dir, "iteration_011", "eki_file.jld2"))
scatter_plot(eki)
param_versus_iter_plot(eki)
minimal_eki_test(eki)
