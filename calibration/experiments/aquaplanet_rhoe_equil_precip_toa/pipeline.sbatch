#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --output=aquaplanet_rhoe_equil_precip_toa_calibration.txt

julia --project=calibration/experiments/aquaplanet_rhoe_equil_precip_toa -e '
import Pkg; Pkg.instantiate(;verbose=true)
import JLD2
using ClimaUtilities.ClimaArtifacts
import ClimaCalibrate: calibrate, ExperimentConfig, ClimaGPUBackend, get_prior, kwargs

experiment_dir = dirname(Base.active_project())
include(joinpath(experiment_dir, "observation_map.jl"))

# Load data and configurations
n_iterations = 1
ensemble_size = 12
observations = JLD2.load_object(joinpath(experiment_dir, "obs_mean.jld2"))
noise = JLD2.load_object(joinpath(experiment_dir, "obs_noise_cov.jld2"))
prior = get_prior(joinpath(experiment_dir, "prior.toml"))
output_dir = joinpath("output", "aquaplanet_rhoe_equil_precip_toa")

experiment_config = ExperimentConfig(; n_iterations, ensemble_size, 
                                       observations, noise, output_dir, prior)
ENV["CLIMACOMMS_DEVICE"] = "CUDA"
ENV["JULIA_MPI_HAS_CUDA"] = "true"
slurm_kwargs = kwargs(time = 180, ntasks = 2, gpus_per_task = 1)
eki = calibrate(experiment_config; slurm_kwargs)

include(joinpath(experiment_dir, "postprocessing.jl"))
prior = get_prior(joinpath(experiment_dir, "prior.toml"))
convergence_plot(eki, prior)
scatter_plot(eki)
'
