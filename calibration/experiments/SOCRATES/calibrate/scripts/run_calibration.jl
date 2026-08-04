"""
Entry point: build observations and run ClimaCalibrate EKI for SOCRATES.

Usage (from experiment root):
  julia --project=. run_calibration.jl
"""

using ClimaCalibrate: ClimaCalibrate
using EnsembleKalmanProcesses: EnsembleKalmanProcesses as EKP
using JLD2: JLD2
using Random: Random
using YAML: YAML

model_interface_filepath = joinpath(@__DIR__, "model_interface", "model_interface.jl")
include(model_interface_filepath)

exp_cfg_path = joinpath(@__DIR__, "configs", "experiment_config.yml")
exp_cfg = YAML.load_file(exp_cfg_path)

config = joinpath(@__DIR__, exp_cfg["model_config"])
output_dir = joinpath(@__DIR__, exp_cfg["output_dir"])
prior_path = joinpath(@__DIR__, exp_cfg["prior_path"])

interface = SOCRATESAtmosModelInterface(config, output_dir, exp_cfg)

obs_path = save_observations!(interface; overwrite = false)
obs_vec = JLD2.load_object(obs_path)

batch_size = Int(exp_cfg["batch_size"])
obs_series = ClimaCalibrate.observation_series_from_samples(
    obs_vec,
    batch_size,
    getfield.(interface.cases, :name),
)

prior = ClimaCalibrate.get_prior(prior_path)
rng = Random.MersenneTwister(1234)

ensemble_size = Int(exp_cfg["ensemble_size"])
ekp = EKP.EnsembleKalmanProcess(
    EKP.construct_initial_ensemble(rng, prior, ensemble_size),
    obs_series,
    EKP.Inversion();
    verbose = true,
    rng,
    scheduler = EKP.DataMisfitController(terminate_at = 1e6),
)

backend = ClimaCalibrate.JuliaBackend()
n_iterations = Int(exp_cfg["n_iterations"])

eki = ClimaCalibrate.calibrate(
    backend,
    ekp,
    interface,
    n_iterations,
    prior,
    interface.output_dir,
)

@info "Calibration finished" eki
