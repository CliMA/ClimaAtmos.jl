#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --output=sphere_held_suarez_rhoe_equilmoist_calibration.txt

julia --project=calibration/experiments/sphere_held_suarez_rhoe_equilmoist -e '
import Pkg; Pkg.instantiate(;verbose=true)
import JLD2
using ClimaUtilities.ClimaArtifacts
import ClimaCalibrate: calibrate, ExperimentConfig, CaltechHPCBackend, get_prior, kwargs

experiment_dir = dirname(Base.active_project())
include(joinpath(experiment_dir, "observation_map.jl"))
artifact_path = @clima_artifact("atmos_held_suarez_obs")
model_interface = joinpath(experiment_dir, "model_interface.jl")

# Load data and configurations
n_iterations = 2
ensemble_size = 10
observations = JLD2.load_object(joinpath(artifact_path, "obs_mean.jld2"))
noise = JLD2.load_object(joinpath(artifact_path, "obs_noise_cov.jld2"))
prior = get_prior(joinpath(experiment_dir, "prior.toml"))
output_dir = joinpath("output", "sphere_held_suarez_rhoe_equilmoist")

experiment_config = ExperimentConfig(; n_iterations, ensemble_size, 
                                       observations, noise, output_dir, prior)
slurm_kwargs = kwargs(time = 50, ntasks = 16, mem_per_cpu = "8G")
eki = calibrate(CaltechHPCBackend, experiment_config; model_interface, slurm_kwargs, verbose=true)

include(joinpath(experiment_dir, "postprocessing.jl"))
prior = get_prior(joinpath(experiment_dir, "prior.toml"))
convergence_plot(eki, prior)
scatter_plot(eki)
'
