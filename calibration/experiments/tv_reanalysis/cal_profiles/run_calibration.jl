using ClimaCalibrate
import ClimaCalibrate as CAL
import ClimaAtmos as CA
import EnsembleKalmanProcesses as EKP
import YAML
import TOML
using Distributions
using Distributed
using Random
using Flux
using Logging

import JLD2
using LinearAlgebra


include("helper_funcs.jl")
include("observation_map.jl")
include("model_interface.jl")

# load configs
experiment_dir = dirname(Base.active_project())
const experiment_config =
    YAML.load_file(joinpath(experiment_dir, "experiment_config.yml"))

# unpack experiment_config vars into scope
for (key, value) in experiment_config
    @eval const $(Symbol(key)) = $value
end

# load configs and directories
model_config_dict = YAML.load_file(model_config)
atmos_config = CA.AtmosConfig(model_config_dict)

# start_datetime = Dates.DateTime(start_time, "yyyymmdd")
# obs_start, obs_end = start_datetime + Dates.Second(g_t_start_sec), start_datetime + Dates.Second(g_t_end_sec)

# add workers
@info "Starting $ensemble_size workers."
addprocs(
    CAL.SlurmManager(Int(ensemble_size)),
    t = experiment_config["slurm_time"],
    mem_per_cpu = experiment_config["slurm_mem_per_cpu"],
    cpus_per_task = experiment_config["slurm_cpus_per_task"],
)

@everywhere begin
    using ClimaCalibrate
    import ClimaCalibrate as CAL
    import ClimaAtmos as CA
    import JLD2
    import YAML

    include("observation_map.jl")

    experiment_dir = dirname(Base.active_project())
    const model_interface = joinpath(experiment_dir, "model_interface.jl")
    const experiment_config =
        YAML.load_file(joinpath(experiment_dir, "experiment_config.yml"))

    include(model_interface)

end


if get(model_config_dict, "mixing_length_model", "") == "nn"
    prior = create_prior_with_nn(
        prior_path,
        pretrained_nn_path;
        arc = [8, 20, 15, 10, 1],
    )
else
    const prior = CAL.get_prior(joinpath(experiment_dir, prior_path))
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

# Copy TOML files specified in model_config.yml
if haskey(model_config_dict, "toml") && !isnothing(model_config_dict["toml"])
    model_config_dir = dirname(model_config)
    for toml_rel_path in model_config_dict["toml"]
        source_toml_path = joinpath(model_config_dir, toml_rel_path)
        # Preserve relative path, don't use basename
        dest_toml_path = joinpath(output_dir, "configs", toml_rel_path)
        if isfile(source_toml_path)
            # Ensure the destination directory exists
            mkpath(dirname(dest_toml_path))
            cp(source_toml_path, dest_toml_path, force = true)
            @info "Copied $source_toml_path to $dest_toml_path"
        else
            @warn "TOML file specified in model_config not found: $source_toml_path"
        end
    end
end

# Copy forcing config files
if haskey(experiment_config, "forcing_toml_files")
    for (forcing_type, config_file) in experiment_config["forcing_toml_files"]
        source_path = joinpath(experiment_dir, config_file)
        # Use the relative path from config_file, not just the basename
        dest_path = joinpath(output_dir, "configs", config_file)

        if isfile(source_path)
            # Ensure the destination directory exists
            mkpath(dirname(dest_path))
            cp(source_path, dest_path, force = true)
            @info "Copied forcing config $source_path to $dest_path"
        else
            @warn "Forcing config file not found: $source_path"
        end
    end
end

# save norm factors to output dir
JLD2.jldsave(
    joinpath(output_dir, "norm_factors.jld2");
    norm_factors_dict = norm_factors_by_var,
)

### get LES obs (Y) and norm factors - also ensures all forcing files are created
start_dates, lats, lons, convection_type, num_sites = get_era5_calibration_library()
ref_paths = []
obs_vec = []

zc_model = get_z_grid(atmos_config; z_max)

for i in 1:num_sites
    # get forcing file path 
    forcing_file_path = CA.get_external_forcing_file_path(
        Dict(
            "start_date" => start_dates[i],
            "site_latitude" => lats[i],
            "site_longitude" => lons[i],
            "t_end" => "31hours",
        ),
    )
    @info forcing_file_path
    push!(ref_paths, forcing_file_path)
    # get obs_start, obs_end from start_date, more flexible for when we batch over months
    obs_start = Dates.DateTime(start_dates[i], "yyyymmdd") +
        Dates.Second(g_t_start_sec)
    obs_end = Dates.DateTime(start_dates[i], "yyyymmdd") +
        Dates.Second(g_t_end_sec)

    y_obs = get_obs(
        forcing_file_path,
        experiment_config["y_var_names"],
        obs_start,
        obs_end;
        normalize = true,
        norm_factors_dict = norm_factors_by_var,
        z_scm = zc_model,
        log_vars = log_vars,
    )
    # build noise covariance matrix - diagonal with rescaled noise
    Σ_obs = get_Σ_obs(
        forcing_file_path,
        experiment_config["y_var_names"],
        obs_start,
        obs_end;
        covariance_structure = covariance_structure,
        normalize = true,
        norm_factors_dict = norm_factors_by_var,
        z_scm = zc_model,
        log_vars = log_vars,
    )

    push!(
        obs_vec,
        EKP.Observation(
            Dict(
                "samples" => y_obs,
                "covariances" => Σ_obs,
                "names" => join([lats[i], lons[i], convection_type[i]], "_"),
            ),
        ),
    )
end

series_names = [ref_paths[i] for i in 1:length(ref_paths)]

### define minibatcher
rfs_minibatcher = EKP.RandomFixedSizeMinibatcher(experiment_config["batch_size"], "trim")
observation_series = EKP.ObservationSeries(obs_vec, rfs_minibatcher, series_names)

@info "Obtained Observations..."

###  EKI hyperparameters/settings
@info "Initializing calibration" n_iterations ensemble_size output_dir

# create the EKI object

ekp_obj = EKP.EnsembleKalmanProcess(
    EKP.construct_initial_ensemble(prior, ensemble_size),
    observation_series,
    EKP.Inversion();
    localization_method = EKP.Localizers.NoLocalization(),
    failure_handler_method = EKP.SampleSuccGauss(),
    accelerator = EKP.DefaultAccelerator(),
    scheduler = EKP.DataMisfitController(on_terminate = "continue"),
    verbose= true,
)

eki = CAL.calibrate(CAL.WorkerBackend, ekp_obj, n_iterations, prior, output_dir; failure_rate = 0.9)


# eki = CAL.calibrate(
#     CAL.WorkerBackend,
#     ensemble_size,
#     n_iterations,
#     observations,
#     nothing, # noise alread specified in observations
#     prior,
#     output_dir;
#     scheduler = EKP.DataMisfitController(on_terminate = "continue"),
#     localization_method = EKP.Localizers.NoLocalization(),
#     ## localization_method = EKP.Localizers.SECNice(nice_loc_ug, nice_loc_gg),
#     failure_handler_method = EKP.SampleSuccGauss(),
#     accelerator = EKP.DefaultAccelerator(),
#     failure_rate = 0.9,
#     #     # accelerator = EKP.NesterovAccelerator(),
# )

# make ekp struct first and then pass 




# eki_obj = JLD2.load_object(experiment_config["output_dir"] * "/iteration_000/" * "eki_file.jld2")
# G_ensemble = JLD2.load_object(experiment_config["output_dir"] * "/iteration_000/" * "G_ensemble.jld2")
# G_ensemble = CAL.observation_map(0)
# prior = CAL.get_prior(joinpath(dirname(Base.active_project()), prior_path))
# CAL.update_ensemble!(eki_obj, G_ensemble, experiment_config["output_dir"], 0, prior)
