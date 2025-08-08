import ClimaCalibrate as CAL
import ClimaAtmos as CA
import EnsembleKalmanProcesses.ParameterDistributions as PD
import EnsembleKalmanProcesses as EKP
import YAML

import YAML
import JLD2
using LinearAlgebra

include("helper_funcs.jl")
include("observation_map.jl")
include("get_les_metadata.jl")

#const prior = CAL.get_prior(joinpath(experiment_dir, prior_path))

const prior_liq = PD.constrained_gaussian("τₗ", 500, 50, 100, 1000) # real = 800?
const prior_ice = PD.constrained_gaussian("τᵢ", 2000, 300, 100, 10000) # real = 5000?

const prior = PD.combine_distributions([prior_1, prior_2])

model_config = "diagnostic_edmfx_diurnal_scm_imp_noneq_1M.yml"

# WRITE & PASS IN MY GROUND TRUTH GUYS
τₗ_truth = 800
τᵢ = 5000

# load configs and directories 
#model_config_dict = YAML.load_file(model_config)
atmos_config = CA.AtmosConfig(model_config) # ADD PARAM DICT HERE W TRUTH VALS
zc_model = get_z_grid(atmos_config; z_max)


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

    #include("observation_map.jl")

    experiment_dir = dirname(Base.active_project())
    const model_interface = joinpath(experiment_dir, "model_interface.jl")
    # const experiment_config =
    #     YAML.load_file(joinpath(experiment_dir, "experiment_config.yml"))

    include(model_interface)

end

# ### define minibatcher
# rfs_minibatcher =
#     EKP.FixedMinibatcher(collect(1:experiment_config["batch_size"]))
# observations = EKP.ObservationSeries(obs_vec, rfs_minibatcher, series_names)

# SET UP OBS AND CREATE EKP STRUCT

EnsembleKalmanProcess(
    params::AbstractMatrix{FT},
    observation_series::OS,
    obs_noise_cov::Union{AbstractMatrix{FT}, UniformScaling{FT}},
    process::P;
    scheduler = DefaultScheduler(1),
    Δt = FT(1),
    rng::AbstractRNG = Random.GLOBAL_RNG,
    failure_handler_method::FM = IgnoreFailures(),
    localization_method::LM = NoLocalization(),
    verbose::Bool = false,
) where {FT <: AbstractFloat, P <: Process, FM <: FailureHandlingMethod, LM <: LocalizationMethod, OS <: ObservationSeries}

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
