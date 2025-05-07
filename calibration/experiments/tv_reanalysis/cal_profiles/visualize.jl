import TOML, YAML
import JLD2
using Distributions
import EnsembleKalmanProcesses as EKP
using EnsembleKalmanProcesses.ParameterDistributions
using EnsembleKalmanProcesses.TOMLInterface
import ClimaCalibrate as CAL
using Plots
using LinearAlgebra
using DataFrames
using Revise
using EnsembleKalmanProcesses

# patch for v2.4.2
includet("postprocessing/ekp_patch.jl")

full_cov_simulation = "/central/groups/esm/jschmitt/calibrations/tv_profiles_29"
var_cov_simulation = "/central/groups/esm/jschmitt/calibrations/tv_profiles_28"
flat_noise_simulation = "/central/groups/esm/jschmitt/calibrations/tv_profiles_24"

output_dir = full_cov_simulation

config_dict =
    YAML.load_file(joinpath(full_cov_simulation, "configs", "experiment_config.yml"))
n_iterations = config_dict["n_iterations"]
iterations = 0:n_iterations

# last eki object
iter_path = CAL.path_to_iteration(output_dir, iterations[end])
eki = JLD2.load_object(joinpath(iter_path, "eki_file.jld2"))

get_obs(eki.obs, build=false)


get_g_final(eki)


