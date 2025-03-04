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
using Statistics


include("helper_funcs.jl")


output_dir = "/groups/esm/cchristo/climaatmos_scm_calibrations/output_ml_mix/exp_43" # path to calibration output
iteration = 10
prefix = ""

write_optimal_toml_dir = "./scm_runner/optimal_tomls"
param_overrides_path = "./scm_tomls/prognostic_edmfx.toml"


const config_dict =
    YAML.load_file(joinpath(output_dir, "configs", "experiment_config.yml"))
const pretrained_nn_path = config_dict["pretrained_nn_path"]


prior_path = joinpath(output_dir, "configs", "prior.toml")
prior = create_prior_with_nn(prior_path, pretrained_nn_path)



iter_path = CAL.path_to_iteration(output_dir, iteration)
eki = JLD2.load_object(joinpath(iter_path, "eki_file.jld2"))
u = EKP.get_u_final(eki)

 
# mean_diff = sum(abs, u .- u_best_mean, dims = 1)
mean_diff = sum(abs, u .- mean(u, dims = 2), dims = 1)
nearest_mean_index = argmin(mean_diff)  
col_index = nearest_mean_index[2]

u_nearest = u[:, col_index]
phi_nearest = EKP.transform_unconstrained_to_constrained(prior, u_nearest)


@info "Ensemble member nearest to the mean for iteration $iteration"
@info "Particle Number: $col_index"
# @info "u values: $u_nearest"
# @info "phi values: $phi_nearest"

param_toml_path = joinpath(CAL.path_to_ensemble_member(output_dir, iteration, col_index), "parameters.toml")
param_nearest = TOML.parsefile(param_toml_path)

param_overrides = TOML.parsefile(param_overrides_path)

merged_params = merge(param_nearest, param_overrides)

# write optimal toml to file 
exp_match = match(r"exp_(\d+)", output_dir)
exp_number = exp_match !== nothing ? "exp" * exp_match.captures[1] : "exp_unknown"
file_name = "parameters_nearest_neig_particle_i$(iteration)_m$(col_index)_$(exp_number)$(prefix).toml"
output_toml_path = joinpath(write_optimal_toml_dir, file_name)
open(output_toml_path, "w") do file
    TOML.print(file, merged_params)
end
@info "Merged parameters written to: $output_toml_path"
