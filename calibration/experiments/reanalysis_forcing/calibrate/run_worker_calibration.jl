# julia --project=calibration/experiments/gcm_driven_scm calibration/experiments/gcm_driven_scm/run_worker_calibration.jl
using Distributed, ClusterManagers
project=dirname(Base.active_project())
cd(project)
# addprocs(SlurmManager(1), t="01:30:00", cpus_per_task=3,exeflags=["--project=$project", "-p", "3"])
# addprocs(SlurmManager(5),  t="01:30:00", cpus_per_task=3,exeflags=["--project=$project", "-p", "3"])
# addprocs(SlurmManager(100),  
#         t="14:50:00", 
#         cpus_per_task=1,
#         exeflags=["--project=$project"], 
#         mem_per_cpu=8000)
function create_worker_pool()    
    addprocs(
        SlurmManager(100),
        t = "10:00:00",
        cpus_per_task = 1,
        exeflags = "--project=$(Base.active_project())"
    )
    return WorkerPool(workers())
end

worker_pool = create_worker_pool()

@info "Running Script..."
@everywhere begin
    import ClimaCalibrate as CAL
    import ClimaAtmos as CA
    
    import EnsembleKalmanProcesses as EKP
    import YAML
    import JLD2
    using LinearAlgebra

    import ClimaComms
    ENV["CLIMACOMMS_CONTEXT"] = "SINGLETON"
    using Revise
    includet("helper_funcs.jl")
    includet("observation_map.jl")
    # include("get_les_metadata.jl")
    experiment_dir = dirname(Base.active_project())
    model_interface = joinpath(experiment_dir, "model_interface.jl")
    includet(model_interface)
    experiment_config =
        YAML.load_file(joinpath(experiment_dir, "experiment_config.yml"))

    experiment_config_nt = NamedTuple(Symbol.(keys(experiment_config)) .=> values(experiment_config))
    (; output_dir, n_iterations, log_vars, prior_path, model_config, const_noise_by_var, z_max, norm_factors_by_var, ensemble_size) = experiment_config_nt

    prior = CAL.get_prior(joinpath(experiment_dir, prior_path))

    # load configs and directories 
    model_config_dict = YAML.load_file(model_config)
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
@info "Loaded packages and copied files..."

### get ERA5 obs (Y) and norm factors
@everywhere begin
    #include("get_les_metadata.jl")
    ref_paths, months, sites = get_era5_calibration_library()
    obs_vec = []

    for i in 1:length(ref_paths)
        
        y_obs = get_obs(
            ref_paths[i],
            months[i],
            sites[i],
            experiment_config["y_var_names"];
            normalize = true,
            norm_factors_dict = norm_factors_by_var,
            z_scm = zc_model,
            log_vars = log_vars,
        )
        Σ_obs = I(length(y_obs))

        push!(
            obs_vec,
            EKP.Observation(
                Dict(
                    "samples" => y_obs,
                    "covariances" => Σ_obs,
                    "names" => join([months[i], sites[i]], "_"),
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
@info "Obtained Observations..."

###  EKI hyperparameters/settings
@info "Initializing calibration" n_iterations ensemble_size output_dir
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

eki = nothing

member_worker(m::Int) = workers()[mod(m - 1, length(workers())) + 1]


function update_ensemble(output_dir::AbstractString, iteration, prior)
    iter_path = CAL.path_to_iteration(output_dir, iteration)
    eki = JLD2.load_object(joinpath(iter_path, "eki_file.jld2"))
    @info eki
    # Load data from the ensemble
    G_ens = JLD2.load_object(joinpath(iter_path, "G_ensemble.jld2"))
    @info G_ens
    terminate = EKP.update_ensemble!(eki, G_ens)
    CAL.save_eki_state(eki, output_dir, iteration + 1, prior)
    return terminate
end

# function run_iteration(ensemble_size, output_dir, iter)
#     futures = map(1:ensemble_size) do m
#         worker = member_worker(m)
#         atmos_configs = remotecall_fetch(CAL.set_up_forward_model, worker, m, iter, experiment_dir)
#         remotecall(CAL.run_forward_model, worker, atmos_configs)
#     end
#     s = @elapsed fetch.(futures)
#     @info "Completed iteration $iter in $(round(s)) seconds, updating ensemble"
#     G_ensemble = observation_map(iter; config_dict = experiment_config)
#     @info G_ensemble
#     CAL.save_G_ensemble(output_dir, iter, G_ensemble)
#     eki = update_ensemble(output_dir, iter, prior)
#     return eki
# end

function run_iteration(ensemble_size, output_dir, iter, worker_pool)
    @sync begin
        for m in 1:ensemble_size
            @async begin
                # Get worker from worker_pool
                worker = take!(worker_pool)
                try
                    @info "Running member $m on worker $worker"
                    # Run the model 
                    atmos_configs = CAL.set_up_forward_model(m, iter, experiment_dir)
                    remotecall_wait(CAL.run_forward_model, worker, atmos_configs)
                catch e 
                    @error "Error running member $m" exception = e
                finally 
                    # Return worker to the worker_pool
                    put!(worker_pool, worker)
                end
            end
        end
    end
end

for iter in 0:n_iterations
    @info "Running Iteration" iter
    (; time, value) = @timed run_iteration(ensemble_size, output_dir, iter, worker_pool)
    @info "Iteration $iter time: $time"
    G_ensemble = observation_map(iter; config_dict = experiment_config)
    @info size(G_ensemble)
    CAL.save_G_ensemble(output_dir, iter, G_ensemble)
    eki = update_ensemble(output_dir, iter, prior)
end
println("Finished!")
