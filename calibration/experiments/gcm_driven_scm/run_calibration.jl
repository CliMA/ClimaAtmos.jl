import ClimaCalibrate as CAL
import ClimaAtmos as CA
import EnsembleKalmanProcesses as EKP
import YAML

import YAML
import JLD2
using LinearAlgebra

include("helper_funcs.jl")
include("observation_map.jl")
include("get_les_metadata.jl")

experiment_dir = dirname(Base.active_project())
const model_interface = joinpath(experiment_dir, "model_interface.jl")
const experiment_config =
    YAML.load_file(joinpath(experiment_dir, "experiment_config.yml"))

# unpack experiment_config vars into scope 
for (key, value) in experiment_config
    @eval const $(Symbol(key)) = $value
end

const prior = CAL.get_prior(joinpath(experiment_dir, prior_path))

# load configs and directories 
model_config_dict = YAML.load_file(model_config)
atmos_config = CA.AtmosConfig(model_config_dict)
zc_model = get_z_grid(atmos_config; z_max)

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
hpc_kwargs = CAL.kwargs(
    time = 60,
    mem_per_cpu = "12G",
    cpus_per_task = batch_size + 1,
    ntasks = 1,
    nodes = 1,
    reservation = "clima",
)
module_load_str = CAL.module_load_string(CAL.CaltechHPCBackend)
for iter in 0:(n_iterations - 1)
    @info "Iteration $iter"
    jobids = map(1:ensemble_size) do member
        @info "Running ensemble member $member"
        CAL.slurm_model_run(
            iter,
            member,
            output_dir,
            experiment_dir,
            model_interface,
            module_load_str;
            hpc_kwargs,
        )
    end

    statuses = CAL.wait_for_jobs(
        jobids,
        output_dir,
        iter,
        experiment_dir,
        model_interface,
        module_load_str;
        hpc_kwargs,
        verbose = false,
        reruns = 0,
    )
    CAL.report_iteration_status(statuses, output_dir, iter)
    @info "Completed iteration $iter, updating ensemble"
    G_ensemble = CAL.observation_map(iter; config_dict = experiment_config)
    CAL.save_G_ensemble(output_dir, iter, G_ensemble)
    eki = CAL.update_ensemble(output_dir, iter, prior)
end
