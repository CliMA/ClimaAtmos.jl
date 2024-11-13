using Distributed, ClusterManagers

addprocs(SlurmManager(50), t="01:30:00", cpus_per_task=3,exeflags="--project=$(Base.active_project())")

@everywhere begin
    experiment_dir = dirname(Base.active_project())
    cd(experiment_dir)
    import ClimaCalibrate as CAL
    import ClimaAtmos as CA
    import EnsembleKalmanProcesses as EKP
    import YAML
    import JLD2
    using LinearAlgebra

    import ClimaComms
    ENV["CLIMACOMMS_CONTEXT"] = "SINGLETON"

    include("helper_funcs.jl")
    include("observation_map.jl")
    include("get_les_metadata.jl")
    const model_interface = joinpath(experiment_dir, "model_interface.jl")
    include(model_interface)
    const experiment_config =
        YAML.load_file(joinpath(experiment_dir, "experiment_config.yml"))

    ec_nt = NamedTuple(
        Symbol.(keys(experiment_config)) .=> values(experiment_config),
    )
    (; output_dir, prior_path, model_config, ensemble_size, z_max) = ec_nt
    (; norm_factors_by_var, n_iterations, const_noise_by_var, log_vars) = ec_nt
    const prior = CAL.get_prior(joinpath(experiment_dir, prior_path))

    # load configs and directories 
    model_config_dict = YAML.load_file(joinpath(experiment_dir, model_config))
    atmos_config = CA.AtmosConfig(model_config_dict)
    zc_model = get_z_grid(atmos_config; z_max)
end

### create output directories & copy configs
mkpath(output_dir)
mkpath(joinpath(output_dir, "configs"))
cp(
    model_config,
    joinpath(output_dir, "configs", "model_config.yml"),
    force = true,
)
cp(
    joinpath(experiment_dir, "experiment_config.yml"),
    joinpath(output_dir, "configs", "experiment_config.yml"),
    force = true,
)
cp(
    joinpath(experiment_dir, prior_path),
    joinpath(output_dir, "configs", "prior.toml"),
    force = true,
)
# save norm factors to output dir
JLD2.jldsave(
    joinpath(output_dir, "norm_factors.jld2");
    norm_factors_dict = norm_factors_by_var,
)

### get LES obs (Y) and norm factors
@everywhere begin
    ref_paths, _ = get_les_calibration_library()
    obs_vec = []

    for ref_path in ref_paths

        y_obs, Σ_obs, norm_vec_obs = get_obs(
            ref_path,
            experiment_config["y_var_names"],
            zc_model;
            ti = experiment_config["y_t_start_sec"],
            tf = experiment_config["y_t_end_sec"],
            norm_factors_dict = norm_factors_by_var,
            z_score_norm = true,
            log_vars = log_vars,
            Σ_const = const_noise_by_var,
            Σ_scaling = "const",
        )

        push!(
            obs_vec,
            EKP.Observation(
                Dict(
                    "samples" => y_obs,
                    "covariances" => Σ_obs,
                    "names" => split(ref_path, "/")[end],
                ),
            ),
        )
    end

    series_names = [ref_paths[i] for i in 1:length(ref_paths)]

    ### define minibatcher
    rfs_minibatcher =
        EKP.FixedMinibatcher(collect(1:experiment_config["batch_size"]))
    observations = EKP.ObservationSeries(obs_vec, rfs_minibatcher, series_names)
end

member_worker(m::Int) = workers()[mod(m - 1, length(workers())) + 1]

function assign_to_free_worker(futures; timeout=nothing, check_interval=5.0)
    start_time = time()
    
    while true
        # Then check if any worker has completed all its tasks
        @show futures
        for (worker_id, future) in futures
            @show (worker_id, future)
            if isnothing(future) || isready(future)
                return (worker_id, time() - start_time)
            end
        end
        
        elapsed = time() - start_time
        if !isnothing(timeout) && (elapsed > timeout)
            throw(ErrorException("Timeout waiting for free worker"))
        end
        
        # Log status periodically (every minute)
        if mod(round(elapsed), 60) == 0
            n_busy = sum(1 for (_, future) in futures if !isnothing(future) && !isready(future))
            @info "Waiting for free worker: $(n_busy)/$(length(workers())) workers busy"
        end
        
        sleep(check_interval)  # Sleep for longer between checks
    end
end

@everywhere run_model(m, iter, experiment_dir) = CAL.run_forward_model(CAL.set_up_forward_model(m, iter, experiment_dir))

function run_iteration(iter)
    worker_futures = Dict{Int, Any}(w => nothing for w in workers())
    all_futures = []
    wait_times = Float64[]  # Track wait times for monitoring
    
    # Distribute ensemble members across workers
    for m in 1:ensemble_size
        try
            # Get an available worker
            @show worker_futures
            worker, wait_time = assign_to_free_worker(
                worker_futures;
            )
            push!(wait_times, wait_time)
            
            @info "Running member $m on worker $worker (waited $(round(wait_time, digits=2))s)"
            future = remotecall(run_model, worker, m, iter, experiment_dir)
            
            worker_futures[worker] = future
            push!(all_futures, future)
            
        catch e
            if e isa ErrorException && occursin("Timeout", e.msg)
                @warn "Timeout waiting for free worker while processing member $m"
            end
            rethrow(e)
        end

    end
    
    s = @elapsed fetch.(all_futures)
    avg_wait = round(mean(wait_times), digits=2)
    max_wait = round(maximum(wait_times), digits=2)
    @info "Completed iteration $iter in $(round(s))s. Average wait for worker: $(avg_wait)s, Max wait: $(max_wait)s"
    G_ensemble = CAL.observation_map(iter; config_dict = experiment_config)
    CAL.save_G_ensemble(output_dir, iter, G_ensemble)
    eki = CAL.update_ensemble(output_dir, iter, prior)
    return eki
end

function calibrate()
    CAL.initialize(
        ensemble_size,
        observations,
        prior,
        output_dir;
        scheduler = EKP.DataMisfitController(on_terminate = "continue"),
        localization_method = EKP.Localizers.NoLocalization(),
        failure_handler_method = EKP.SampleSuccGauss(),
        accelerator = EKP.DefaultAccelerator(),
    )
    for iter in 0:n_iterations
        s = @elapsed run_iteration(iter)
        @info "Iteration $iter time: $s"
    end
end

run_iteration(0)
