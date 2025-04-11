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

# load configs
experiment_dir = dirname(Base.active_project())
const model_interface = joinpath(experiment_dir, "model_interface.jl")
const experiment_config =
    YAML.load_file(joinpath(experiment_dir, "experiment_config.yml"))


# unpack experiment_config vars into scope
for (key, value) in experiment_config
    @eval const $(Symbol(key)) = $value
end

# load configs and directories
model_config_dict = YAML.load_file(model_config)
atmos_config = CA.AtmosConfig(model_config_dict)

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

    experiment_config_nt = NamedTuple(Symbol.(keys(experiment_config)) .=> values(experiment_config))
    (; output_dir, n_iterations, log_vars, prior_path, model_config, const_noise_by_var, z_max, norm_factors_by_var, ensemble_size, start_time, g_t_start_sec, g_t_end_sec) = experiment_config_nt

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

### get LES obs (Y) and norm factors
ref_paths, latitudes, longitudes, convection_type = get_era5_calibration_library()
obs_vec = []

for i in 1:length(ref_paths)

    y_obs = get_obs(
        ref_paths[i],
        experiment_config["y_var_names"],
        obs_start,
        obs_end;
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
                "names" => join([latitudes[i], longitudes[i], convection_type[i]], "_"),
            ),
        ),
    )
end

series_names = [ref_paths[i] for i in 1:length(ref_paths)]

### define minibatcher
rfs_minibatcher =
    EKP.RandomFixedSizeMinibatcher(experiment_config["batch_size"])
observations = EKP.ObservationSeries(obs_vec, rfs_minibatcher, series_names)

@info "Obtained Observations..."

###  EKI hyperparameters/settings
@info "Initializing calibration" n_iterations ensemble_size output_dir

eki = CAL.calibrate(
    CAL.WorkerBackend,
    ensemble_size,
    n_iterations,
    observations,
    nothing, # noise alread specified in observations
    prior,
    output_dir;
    scheduler = EKP.DataMisfitController(on_terminate = "continue"),
    localization_method = EKP.Localizers.NoLocalization(),
    ## localization_method = EKP.Localizers.SECNice(nice_loc_ug, nice_loc_gg),
    failure_handler_method = EKP.SampleSuccGauss(),
    accelerator = EKP.DefaultAccelerator(),
    #     # accelerator = EKP.NesterovAccelerator(),
)