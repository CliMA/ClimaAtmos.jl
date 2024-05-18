import ClimaCalibrate as CAL
import ClimaAtmos as CA
import EnsembleKalmanProcesses as EKP
import YAML

# import ClimaCalibrate:
#     calibrate, ExperimentConfig, CaltechHPC, get_prior, kwargs
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
const n_iterations = experiment_config["n_iterations"]
const ensemble_size = experiment_config["ensemble_size"]
const output_dir = experiment_config["output_dir"]
const model_config = experiment_config["model_config"]
const prior =
    CAL.get_prior(joinpath(experiment_dir, experiment_config["prior"]))

# load configs and directories 
model_config_dict = YAML.load_file(model_config)
atmos_config = CA.AtmosConfig(model_config_dict)


# get/store LES obs and norm factors 
zc_model = get_z_grid(atmos_config)

norm_factors_dict = Dict(
    "thetaa" => [306.172, 8.07383],
    "hus" => [0.0063752, 0.00471147],
    "clw" => [2.67537e-6, 4.44155e-6],
)



if !isdir(output_dir)
    mkpath(output_dir)
end

JLD2.jldsave(
    joinpath(output_dir, "norm_factors.jld2");
    norm_factors_dict = norm_factors_dict,
)




ref_paths = get_les_calibration_library()


obs_vec = []
for ref_path in ref_paths

    y_obs, Σ_obs, norm_vec_obs = get_obs(
        ref_path,
        experiment_config["y_var_names"],
        zc_model;
        ti = experiment_config["y_t_start_sec"],
        tf = experiment_config["y_t_end_sec"],
        norm_factors_dict = norm_factors_dict,
        Σ_const = 0.05,
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

# minibatcher
rfs_minibatcher =
    EKP.RandomFixedSizeMinibatcher(experiment_config["batch_size"])
observations = EKP.ObservationSeries(obs_vec, rfs_minibatcher, series_names)

# ExperimentConfig is created from a YAML file within the experiment_dir
@info "Initializing calibration" n_iterations ensemble_size output_dir
CAL.initialize(
    ensemble_size,
    observations,
    prior,
    output_dir;
    scheduler = EKP.DefaultScheduler(0.001),
)

eki = nothing
hpc_kwargs = CAL.kwargs(time = 60, mem = "16G")
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
        verbose = true,
        reruns = 0,
    )
    CAL.report_iteration_status(statuses, output_dir, iter)
    @info "Completed iteration $iter, updating ensemble"
    G_ensemble = CAL.observation_map(iter)
    CAL.save_G_ensemble(output_dir, iter, G_ensemble)
    eki = CAL.update_ensemble(output_dir, iter, prior)
end
