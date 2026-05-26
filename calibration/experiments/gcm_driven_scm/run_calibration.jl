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

function tprintln(msg)
    println("[$(Dates.now())] $msg")
    flush(stdout)
end
using Dates

include("helper_funcs.jl")
include("get_les_metadata.jl")
include("nn_helpers.jl")

# load configs
experiment_dir = dirname(Base.active_project())
const model_interface = joinpath(experiment_dir, "model_interface.jl")
const experiment_config =
    YAML.load_file(joinpath(experiment_dir, "experiment_config.yml"))

struct GCMDrivenSCMInterface <: CAL.AbstractModelInterface end

include("observation_map.jl")


# unpack experiment_config vars into scope 
for (key, value) in experiment_config
    @eval const $(Symbol(key)) = $value
end

# load configs and directories 
model_config_dict = YAML.load_file(model_config)
atmos_config = CA.AtmosConfig(model_config_dict)

# --- Validate config consistency ---
let
    t_end_str = model_config_dict["t_end"]
    # Parse t_end to seconds (supports "Nhours", "Ndays", "Nsecs" formats)
    t_end_sec = if occursin("hours", t_end_str)
        parse(Float64, replace(t_end_str, "hours" => "")) * 3600
    elseif occursin("days", t_end_str)
        parse(Float64, replace(t_end_str, "days" => "")) * 86400
    elseif occursin("secs", t_end_str)
        parse(Float64, replace(t_end_str, "secs" => ""))
    else
        @warn "Could not parse model t_end='$t_end_str'; skipping config validation"
        nothing
    end

    if !isnothing(t_end_sec)
        g_t_end = experiment_config["g_t_end_sec"]
        g_t_end_max = isa(g_t_end, AbstractFloat) ? g_t_end :
                      maximum(values(g_t_end))
        if t_end_sec < g_t_end_max
            error(
                "Config mismatch: model t_end=$(t_end_str) ($(t_end_sec)s) " *
                "is shorter than g_t_end_sec=$(g_t_end_max)s. " *
                "The observation map requires simulation data up to g_t_end_sec. " *
                "Increase t_end in the model config or decrease g_t_end_sec.",
            )
        end
    end
end

# add workers
tprintln("Starting $ensemble_size workers...")
addprocs(
    CAL.SlurmManager(Int(ensemble_size)),
    t = experiment_config["slurm_time"],
    mem_per_cpu = experiment_config["slurm_mem_per_cpu"],
    cpus_per_task = experiment_config["slurm_cpus_per_task"],
)


tprintln("Workers added. Loading packages on workers...")

@everywhere begin
    using ClimaCalibrate
    import ClimaCalibrate as CAL
    import ClimaAtmos as CA
    import JLD2
    import YAML

    experiment_dir = dirname(Base.active_project())
    const experiment_config =
        YAML.load_file(joinpath(experiment_dir, "experiment_config.yml"))

    include(joinpath(experiment_dir, "model_interface.jl"))
    include(joinpath(experiment_dir, "observation_map.jl"))
end

tprintln("Worker setup complete.")


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
ref_paths, _ = get_les_calibration_library() #max_cases = 20, models = "HadGEM2-A")
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

# function create_minibatches_internal(n_indices::Int, batch_size::Int)
#     shuffled_indices = shuffle(1:n_indices)
#     num_full_batches = div(n_indices, batch_size)
#     batches = [collect(shuffled_indices[(i-1)*batch_size + 1 : i*batch_size]) for i in 1:num_full_batches]
#     return batches
# end

# minibatch_inds = create_minibatches_internal(length(series_names), experiment_config["batch_size"])

# @show minibatch_inds

### Random fixed 
rfs_minibatcher = EKP.RandomFixedSizeMinibatcher(experiment_config["batch_size"], "trim")
observations = EKP.ObservationSeries(obs_vec, rfs_minibatcher, series_names)

### no minibatching
# rfs_minibatcher =
#     EKP.FixedMinibatcher(collect(1:experiment_config["batch_size"]))
## rfs_minibatcher = EKP.no_minibatcher(experiment_config["batch_size"])
# observations = EKP.ObservationSeries(obs_vec, rfs_minibatcher, series_names)



###  EKI hyperparameters/settings
tprintln("Initializing calibration: n_iterations=$n_iterations, ensemble_size=$ensemble_size")

initial_ensemble =
    EKP.construct_initial_ensemble(prior, Int(ensemble_size))

ekp = EKP.EnsembleKalmanProcess(
    initial_ensemble,
    observations,
    EKP.Inversion();
    scheduler = EKP.DataMisfitController(on_terminate = "continue"),
    localization_method = EKP.Localizers.NoLocalization(),
    ## localization_method = EKP.Localizers.SECNice(nice_loc_ug, nice_loc_gg),
    failure_handler_method = EKP.SampleSuccGauss(),
    accelerator = EKP.DefaultAccelerator(),
    #     # accelerator = EKP.NesterovAccelerator(),
)

interface = GCMDrivenSCMInterface()
backend = CAL.WorkerBackend()

tprintln("Starting calibration loop...")

output_dir_abs = abspath(output_dir)
ensemble_size_int = EKP.get_N_ens(ekp)

# Only initialize if this is a fresh run; avoid overwriting the EKP
# (and its minibatch) when restarting from existing simulation data.
if !isfile(CAL.Calibration.ekp_path(output_dir_abs, 1))
    tprintln("Fresh run — initializing EKP and saving parameters...")
    ekp = CAL.Calibration.initialize(ekp, prior, output_dir_abs)
else
    tprintln("Resuming — loading existing EKP from iteration 1...")
    ekp = CAL.Calibration.load_ekp_struct(output_dir_abs, 1)
end

first_iter = CAL.Calibration.last_completed_iteration(output_dir_abs) + 1
tprintln("First iteration: $first_iter")

for iter in first_iter:Int(n_iterations)
    tprintln("=== Iteration $iter: starting forward model ===")
    t0 = time()
    CAL.Calibration.run_iteration(
        backend,
        interface,
        iter,
        ensemble_size_int,
        output_dir_abs,
    )
    tprintln("=== Iteration $iter: forward model done ($(round(time()-t0, digits=1))s) ===")

    tprintln("=== Iteration $iter: loading EKP struct ===")
    t1 = time()
    ekp = CAL.Calibration.load_ekp_struct(output_dir_abs, iter)
    tprintln("=== Iteration $iter: EKP struct loaded ($(round(time()-t1, digits=1))s) ===")

    tprintln("=== Iteration $iter: running observation_map ===")
    t2 = time()
    g_ensemble = CAL.observation_map(interface, iter)
    tprintln("=== Iteration $iter: observation_map done ($(round(time()-t2, digits=1))s) ===")

    # Sanity-check G_ensemble before proceeding
    n_nan_cols = sum(all(isnan, g_ensemble, dims = 1))
    n_ens = size(g_ensemble, 2)
    nan_frac = n_nan_cols / n_ens
    tprintln("=== Iteration $iter: G_ensemble NaN columns: $n_nan_cols / $n_ens ($(round(nan_frac*100, digits=1))%) ===")
    if n_nan_cols == n_ens
        error(
            "Observation map failed for ALL $n_ens ensemble members. " *
            "Check simulation output and config consistency (t_end vs g_t_end_sec).",
        )
    end

    tprintln("=== Iteration $iter: saving G_ensemble ===")
    CAL.Calibration.save_G_ensemble(output_dir_abs, iter, g_ensemble)
    tprintln("=== Iteration $iter: G_ensemble saved ===")

    tprintln("=== Iteration $iter: running EKP update_ensemble! ===")
    t3 = time()
    terminate = CAL.Calibration.update_ensemble!(
        ekp,
        g_ensemble,
        output_dir_abs,
        iter,
        prior,
    )
    tprintln("=== Iteration $iter: EKP update done ($(round(time()-t3, digits=1))s) ===")

    tprintln("=== Iteration $iter: COMPLETE ===")
    !isnothing(terminate) && break
end

tprintln("Calibration finished.")
