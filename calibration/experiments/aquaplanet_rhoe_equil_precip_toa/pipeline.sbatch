#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --output=aquaplanet_rhoe_equil_precip_toa_calibration.txt

julia --project=calibration/experiments/aquaplanet_rhoe_equil_precip_toa -e '
import Pkg; Pkg.instantiate(;verbose=true)
import JLD2
using ClimaUtilities.ClimaArtifacts
import ClimaCalibrate: calibrate, ExperimentConfig, CaltechHPCBackend, get_prior, kwargs

experiment_dir = dirname(Base.active_project())
include(joinpath(experiment_dir, "observation_map.jl"))

# Load data and configurations
n_iterations = 1
ensemble_size = 10
observations = JLD2.load_object(joinpath(experiment_dir, "obs_mean.jld2"))
noise = JLD2.load_object(joinpath(experiment_dir, "obs_noise_cov.jld2"))
prior = get_prior(joinpath(experiment_dir, "prior.toml"))
output_dir = joinpath("output", "aquaplanet_rhoe_equil_precip_toa")

experiment_config = ExperimentConfig(; n_iterations, ensemble_size, 
                                       observations, noise, output_dir, prior)
slurm_kwargs = kwargs(time = 1200, ntasks = 64, mem_per_cpu = "8G")
eki = calibrate(CaltechHPCBackend, experiment_config; slurm_kwargs)

include(joinpath(experiment_dir, "postprocessing.jl"))
prior = get_prior(joinpath(experiment_dir, "prior.toml"))
convergence_plot(eki, prior)
scatter_plot(eki)
'
