import ClimaCalibrate:
    calibrate, ExperimentConfig, CaltechHPC, get_prior, kwargs
import ClimaAtmos as CA
import EnsembleKalmanProcesses as EKP
import YAML
import JLD2
using LinearAlgebra

include("helper_funcs.jl")
include("observation_map.jl")

# load configs and directories 
experiment_dir = dirname(Base.active_project())
model_config_dict = YAML.load_file("model_config.yml")
atmos_config = CA.AtmosConfig(model_config_dict)
experiment_config_dict = YAML.load_file("experiment_config.yml")

# get/store LES obs and norm factors 
zc_model = get_z_grid(atmos_config)
y_obs, Σ_obs, norm_vec_obs = get_obs(
    model_config_dict["external_forcing_file"],
    experiment_config_dict["y_var_names"],
    zc_model;
    ti = experiment_config_dict["y_t_start_sec"],
    tf = experiment_config_dict["y_t_end_sec"],
    Σ_const = 0.05,
)


if !isdir(experiment_config_dict["output_dir"])
    mkpath(experiment_config_dict["output_dir"])
end

JLD2.jldsave(
    joinpath(experiment_config_dict["output_dir"], "norm_vec_obs.jld2");
    norm_vec_obs = norm_vec_obs,
)
JLD2.save_object("obs_mean.jld2", y_obs)
JLD2.save_object("obs_noise_cov.jld2", Σ_obs)


# define slurm kwargs and start calibration
slurm_kwargs = kwargs(time = 90, mem = "16G")
eki = calibrate(
    CaltechHPC,
    experiment_dir;
    slurm_kwargs,
    verbose = true,
    scheduler = EKP.DefaultScheduler(0.5),
)
