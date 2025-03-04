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
include("get_les_metadata.jl")
include("nn_helpers.jl")

# load configs
experiment_dir = dirname(Base.active_project())
const model_interface = joinpath(experiment_dir, "model_interface.jl")
const experiment_config =
    YAML.load_file(joinpath(experiment_dir, "experiment_config.yml"))


# unpack experiment_config vars into scope 
for (key, value) in experiment_config
    @eval const $(Symbol(key)) = $value
end


# add workers
@info "Starting $ensemble_size workers."
addprocs(CAL.SlurmManager(Int(ensemble_size)), t = "02:00:00", mem_per_cpu = "15G", cpus_per_task = 1)#cpus_per_task = 1)

# load configs and directories 
model_config_dict = YAML.load_file(model_config)
atmos_config = CA.AtmosConfig(model_config_dict)

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


if model_config_dict["mixing_length_model"] == "nn"
    prior = create_prior_with_nn(prior_path, pretrained_nn_path; arc = [8, 20, 15, 10, 1])
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
# save norm factors to output dir
JLD2.jldsave(
    joinpath(output_dir, "norm_factors.jld2");
    norm_factors_dict = norm_factors_by_var,
)

### get LES obs (Y) and norm factors
ref_paths, _ = get_les_calibration_library()
obs_vec = []

for ref_path in ref_paths

    cfsite_number, _, _, _ = parse_les_path(ref_path)
    forcing_type = get_cfsite_type(cfsite_number)
    zc_model = get_cal_z_grid(atmos_config, z_cal_grid, forcing_type)

    ti = experiment_config["y_t_start_sec"]
    ti = isa(ti, AbstractFloat) ? ti : ti[forcing_type]
    tf = experiment_config["y_t_end_sec"]
    tf = isa(tf, AbstractFloat) ? tf : tf[forcing_type]

    y_obs, Σ_obs, norm_vec_obs = get_obs(
        ref_path,
        experiment_config["y_var_names"],
        zc_model;
        ti = ti,
        tf = tf,
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

function create_minibatches_internal(n_indices::Int, batch_size::Int)
    shuffled_indices = shuffle(1:n_indices)
    num_full_batches = div(n_indices, batch_size)
    batches = [collect(shuffled_indices[(i-1)*batch_size + 1 : i*batch_size]) for i in 1:num_full_batches]
    return batches
end

minibatch_inds = create_minibatches_internal(length(series_names), experiment_config["batch_size"])

@show minibatch_inds

rfs_minibatcher =
    EKP.FixedMinibatcher(minibatch_inds)


# for (i, batch) in enumerate(rfs_minibatcher)
#     println("Batch $i: ", batch)
# end
    
# rfs_minibatcher =
#     EKP.FixedMinibatcher(collect(1:experiment_config["batch_size"]))

observations = EKP.ObservationSeries(obs_vec, rfs_minibatcher, series_names)

###  EKI hyperparameters/settings
@info "Initializing calibration" n_iterations ensemble_size output_dir

eki = CAL.calibrate(
    CAL.WorkerBackend,
    ensemble_size,
    n_iterations,
    observations,
    nothing, # nothing
    prior,
    output_dir;
    scheduler = EKP.DataMisfitController(on_terminate = "continue"),
    localization_method = EKP.Localizers.NoLocalization(),
    ## localization_method = EKP.Localizers.SECNice(nice_loc_ug, nice_loc_gg),
    # failure_handler_method = EKP.SampleSuccGauss(),
    accelerator = EKP.DefaultAccelerator(),
    #     # accelerator = EKP.NesterovAccelerator(),
)