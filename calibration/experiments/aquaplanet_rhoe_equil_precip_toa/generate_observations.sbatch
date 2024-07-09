#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --ntasks=512
#SBATCH --output="generate_obs.txt"
#SBATCH --mem-per-cpu=8G

EXPERIMENT_DIR="calibration/experiments/aquaplanet_rhoe_equil_precip_toa"

julia --project=$EXPERIMENT_DIR -e 'import Pkg; Pkg.instantiate(;verbose=true)'

srun julia --project=$EXPERIMENT_DIR -e '
import ClimaAtmos as CA
import YAML
using ClimaAnalysis
import ClimaAnalysis.Visualize as viz
import ClimaComms; ClimaComms.@import_required_backends()
import JLD2
import Statistics: var

experiment_dir = dirname(Base.active_project())
config_dict = YAML.load_file(joinpath(experiment_dir, "model_config.yml"))
config_dict["t_end"] = "900days"
config_dict["dt_save_state_to_disk"] = "300days"
output_dir = joinpath(experiment_dir, "generate_observations")
config_dict["output_dir"] = output_dir
config = CA.AtmosConfig(config_dict)
simulation = CA.get_simulation(config)
sol_res = CA.solve_atmos!(simulation)

simdir = SimDir(joinpath(output_dir, "output_active"))
short_names = ("rlut", "rsut", "pr")
period = "60d"

observations = Vector{Float64}(undef, 3)
noise = Matrix{Float64}(undef, 3, 3)
const days = 86400.0

for (i, short_name) in enumerate(short_names)
    output_var = average_lat(average_lon(get(simdir; short_name, period)))
    observations[i] = slice(output_var, time = 360days).data[1]
    noise[i,i] = var(output_var.data)
end

JLD2.save_object(joinpath(experiment_dir, "obs_noise_cov.jld2"), noise)
JLD2.save_object(joinpath(experiment_dir, "obs_mean.jld2"), observations)
'
