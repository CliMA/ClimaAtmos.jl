#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=16
#SBATCH --partition=expansion
#SBATCH --output="experiments/sphere_held_suarez_rhoe_equilmoist/truth_simulation/model_log.txt"

# Configure the environment
export MODULEPATH=/groups/esm/modules:$MODULEPATH
module load climacommon/2024_04_05

echo "Generating truth observations."

srun julia --project=experiments/sphere_held_suarez_rhoe_equilmoist -e '
import ClimaAtmos as CA
import YAML
import JLD2
using ClimaAnalysis

experiment_dir = joinpath("experiments", "sphere_held_suarez_rhoe_equilmoist")
output_dir = joinpath(experiment_dir, "truth_simulation")

config_dict = YAML.load_file(joinpath(experiment_dir, "model_config.yml"))
config_dict["t_end"] = "3000days"
config_dict["output_dir"] = output_dir
config_dict["restart_file"] = nothing

config = CA.AtmosConfig(config_dict)
simulation = CA.get_simulation(config)
CA.solve_atmos!(simulation)

include(joinpath(experiment_dir, "observation_map.jl"))
simdir = SimDir(joinpath(output_dir, "output_active"))
(; observation, variance) = process_member_data(simdir; output_variance = true)
JLD2.save_object(joinpath(experiment_dir, "obs_mean.jld2"), observation)
JLD2.save_object(joinpath(experiment_dir, "obs_noise_cov.jld2"), variance)
'
