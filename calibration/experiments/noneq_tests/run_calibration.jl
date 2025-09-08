import ClimaCalibrate as CAL
import ClimaAtmos as CA
import EnsembleKalmanProcesses.ParameterDistributions as PD
import EnsembleKalmanProcesses as EKP
import YAML
import JLD2
using LinearAlgebra
using Distributions
using Distributed

include("observation_map.jl")
include("model_interface.jl")

#const prior = CAL.get_prior(joinpath(experiment_dir, prior_path))
fast_timescale = true

if fast_timescale
    prior_vec = [PD.constrained_gaussian("condensation_evaporation_timescale", 20, 19, 0.1, 100), # real = 5s
                 PD.constrained_gaussian("sublimation_deposition_timescale", 100, 99, 0.1, 500)] # real = 20s
else
    prior_vec = [PD.constrained_gaussian("condensation_evaporation_timescale", 700, 300, 100, 1100), # real = 800s
                PD.constrained_gaussian("sublimation_deposition_timescale", 3000, 2000, 100, 10000)] # real = 5000s
end

const prior = PD.combine_distributions(prior_vec)

ensemble_size = 20
n_iterations = 10
output_dir = "/home/oalcabes/EKI_output/test_14"

run_truth = false

if run_truth

    #model_config = "diagnostic_edmfx_diurnal_scm_exp_noneq_1M.yml"
    model_config = "diagnostic_edmfx_diurnal_scm_imp_noneq_1M_mixed_phase_site.yml"

    config_dict = YAML.load_file(model_config)

    if fast_timescale
        truth_toml = "toml/truth_fast.toml"
    else
        truth_toml = "toml/truth.toml"
    end

    # load configs and directories -- running truth!
    push!(config_dict["toml"], truth_toml)
    @show config_dict["toml"]
    atmos_config = CA.AtmosConfig(config_dict) # ADD PARAM DICT HERE W TRUTH VALS
    diag_sim = CA.AtmosSimulation(atmos_config)
    CA.solve_atmos!(diag_sim)
    truth_out_dir = diag_sim.output_dir
else
    if fast_timescale
        truth_out_dir = "/home/oalcabes/ClimaAtmos.jl/calibration/experiments/noneq_tests/output/output_0061" # mixed phase diag explicit truth -- short
    else
        truth_out_dir = "/home/oalcabes/ClimaAtmos.jl/calibration/experiments/noneq_tests/output/output_0058" # mixed phase diag explicit truth -- long
    end

    #truth_out_dir = "/home/oalcabes/ClimaAtmos.jl/calibration/experiments/noneq_tests/output/output_0060" # mixed phase diag implicit truth -- long
    #truth_out_dir = "/home/oalcabes/ClimaAtmos.jl/calibration/experiments/noneq_tests/output/output_0062" # mixed phase diag implicit truth -- short
    #truth_out_dir = "/home/oalcabes/ClimaAtmos.jl/calibration/experiments/noneq_tests/output/output_0058" # mixed phase diag explicit truth -- long
    #truth_out_dir = "/home/oalcabes/ClimaAtmos.jl/calibration/experiments/noneq_tests/output/output_0061" # mixed phase diag explicit truth -- short
    #truth_out_dir = "/home/oalcabes/ClimaAtmos.jl/calibration/experiments/noneq_tests/output/output_0051" # warm prog truth
    #truth_out_dir = "/home/oalcabes/ClimaAtmos.jl/calibration/experiments/noneq_tests/output/output_0021"
end

# add workers
@info "Starting $ensemble_size workers."
addprocs(
    CAL.SlurmManager(Int(ensemble_size)),
    t = "6:00:00",
    mem_per_cpu = "25G",
    cpus_per_task = 1,
)

@everywhere begin
    using ClimaCalibrate
    import ClimaCalibrate as CAL
    import ClimaAtmos as CA
    import JLD2
    import YAML
    using LinearAlgebra
    using Distributions
    using Distributed

    include("observation_map.jl")
    include("model_interface.jl")

    ensemble_size = 20
    n_iterations = 10
    output_dir = "/home/oalcabes/EKI_output/test_14"

    experiment_dir = dirname(Base.active_project())
    #const model_interface = joinpath(experiment_dir, "..", "model_interface.jl")
    # const experiment_config =
    #     YAML.load_file(joinpath(experiment_dir, "experiment_config.yml"))

    #include(model_interface)

end

observations = process_member_data(SimDir(truth_out_dir))
noise = diagm(ones(2))*1e-7 #Diagonal([0.1*diagm(ones(2)), 0.1*diagm(ones(2))])

observation = EKP.Observation(
                    Dict(
                    "samples" => observations,
                    "covariances" => noise,
                    "names" => "water_paths",
                )
            )

ekp_obj = EKP.EnsembleKalmanProcess(
    EKP.construct_initial_ensemble(prior, ensemble_size),
    observation,
    EKP.Inversion();
    localization_method = EKP.Localizers.NoLocalization(),
    failure_handler_method = EKP.SampleSuccGauss(),
    accelerator = EKP.DefaultAccelerator(),
    scheduler = EKP.DataMisfitController(on_terminate = "continue"),
    verbose= true,
)

eki = CAL.calibrate(CAL.WorkerBackend, ekp_obj, n_iterations, prior, output_dir; failure_rate = 0.9)
