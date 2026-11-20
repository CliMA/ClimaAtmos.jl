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
using BSON


include("helper_funcs.jl")


output_dir = "/resnick/groups/esm/jschmitt/climaatmos_scm_calibrations/output_cf_ml_v1/exp4_hess_cov_nn_prior_tenth"
iteration = 11
prefix = "nn_hess_noise_cf"
EXP_DIR_PATTERN = r"exp(\d+)"


write_optimal_toml_dir = "./calibrated_tomls/pert_pres"
param_overrides_path = "./scm_tomls/prognostic_edmfx.toml"

write_optimal_toml_dir = "./scm_runner/optimal_tomls"
param_overrides_path = "./scm_tomls/prognostic_edmfx.toml"


const config_dict =
    YAML.load_file(joinpath(output_dir, "configs", "experiment_config.yml"))
# const pretrained_nn_path = config_dict["pretrained_nn_path"]


prior_path = joinpath(output_dir, "configs", "prior.toml")
# prior = create_prior_with_nn(prior_path, pretrained_nn_path)

prior = cf_prior_with_nn(
    prior_path,
    "/resnick/groups/esm/jschmitt/ClimaAtmos.jl/calibration/experiments/cf_calibration/prior_network_generator_hess-gauss.bson",
)

# Select prior consistent with calibration setup (NN vs physical)
model_config_path = joinpath(output_dir, "configs", "model_config.yml")
model_config_dict = YAML.load_file(model_config_path)


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

param_toml_path = joinpath(
    CAL.path_to_ensemble_member(output_dir, iteration, col_index),
    "parameters.toml",
)
param_nearest = TOML.parsefile(param_toml_path)

param_overrides = TOML.parsefile(param_overrides_path)

merged_params = merge(param_nearest, param_overrides)

# write optimal toml to file 
exp_match = match(EXP_DIR_PATTERN, output_dir)
exp_number =
    exp_match !== nothing ? "exp" * exp_match.captures[1] : "exp_unknown"
file_name = "parameters_nearest_neig_particle_i$(iteration)_m$(col_index)_$(exp_number)_$(prefix).toml"
output_toml_path = joinpath(write_optimal_toml_dir, file_name)
open(output_toml_path, "w") do file
    TOML.print(file, merged_params)
end
@info "Merged parameters written to: $output_toml_path"

using CairoMakie

nn_cf_old =  [-0.74087787, -0.3999826, 0.74106497, 1.1249413, -0.7692651, -0.3288731, 1.1838305, -0.44306874, 1.2400066, 0.08220636, 4.162183, -4.255733, -4.1470923, -3.5917249, 1.9350492, -3.1265073, 0.62557614, 0.59036386, 0.38260913, -0.04336004, -0.022237444, 0.21320705, -0.73384005, -1.5728604, 0.12632804, 0.25012466, -0.79051197, 2.901747, 0.99446714, -0.28530046, 2.8799458, -3.1699934, -0.008281425, 0.04032145, 0.0022431544, 0.23048602, 0.002109726, -0.0027990476, 0.014690841, 0.015412016, -0.47020674, 0.0911798, 0.5514113, 0.26639637, -0.7011554, 0.048635133, -0.11820925, -0.74253386, -0.6299769, 0.25821522, 0.45762223, -0.24785824, -0.54129744, 0.43782854, 0.32382002, -0.37370422, 1.6636233, 0.42479172, -0.60597306, -0.22591783, 0.34605289, 0.20129378, -0.9169212, -1.7097968, -0.15992814, 0.42109442, 1.4578766, 0.96085256, 1.4828439, -0.71255374, 1.3403219, 1.1625379, -1.1571414, -0.34768724, 0.82053113, 0.6995275, 1.0853457, -0.8751673, 1.4938492, 0.26861444, -1.5908451, -0.90890837, 0.8357902, 0.7919029, 1.0858604, -0.4867975, 0.23034543, 1.3411491, 1.1225301, 0.6950453, -0.81695104, -0.023932422, -1.0674969, -0.29764092, -1.7934479, -1.5139523, -2.155796, 0.56899023, -0.7082227, -0.80488247, -0.28393012, -4.3622384, 0.790751, 0.97193205, 0.14509517, -0.1046927, -0.13245293, -0.17821075, 0.11578787, 0.26055467, -0.31904995, -0.19141985, -1.5523665, -0.5322224, 0.7271137, 0.63075674, 1.1851735, -1.550133, 2.073769, 1.3312619, -0.09809579]
nn_cf_new = [-0.8682884598195689, -0.33516903962740846, 0.7670039058300598, 1.4290562302093883, -0.9954849517708659, -0.4372110416889669, 0.6527811691350767, -0.49334645225903695, 1.2371306591958426, 0.0841798844472313, 4.159101610121286, -4.23858354116815, -4.153737619746136, -3.5951161893914825, 1.905681074479979, -3.1282191766402163, 0.7592199812892576, 0.5618582685015713, -0.08134081829212864, 0.10733658559373292, -0.12655738125289695, 0.1785476350073567, -0.7461399706631531, -1.5577865139897704, 0.1279148272819999, 0.25049469576518, -0.7980938050252959, 2.90496071978161, 0.9936047854718992, -0.2858689975220472, 2.8794032466434887, -3.170333262616446, 0.3507499050311194, -0.12466887445926417, -0.1452411798794418, 0.17372933916058367, -0.08754525598052403, 0.07042495369331678, 0.038950828602612994, 0.17704085348577908, -0.46165498801236265, 0.08953215254663223, 0.5210803208385826, 0.2541370232953319, -0.7972800051826472, 0.1384430780149269, -0.2155895902063, -0.7550122730888524, -0.610289216969961, 0.2592132127678255, 0.4550045307660597, -0.2373961923692655, -0.5811247959607664, 0.489616885303196, 0.3169312409286337, -0.39387779104180864, 1.5709323512987567, 0.4804024660882293, -0.6019870444485985, -0.376652938598611, 0.4473571232974799, 0.1934710678257229, -0.8932571958359236, -1.7172936064750248, -0.29059610931965457, 0.4713338713195026, 1.4257980598336142, 0.7991583802449825, 1.6605848267289642, -0.6642369874338104, 1.359892648501992, 1.0900900243728275, -1.1515309104019762, -0.34789096125835933, 0.8178059348415846, 0.7087608970512826, 1.0533252031320948, -0.8364386816192027, 1.4781568154831253, 0.26170253881793043, -1.588066004770028, -0.9088307659295064, 0.8423653609508381, 0.8051130771090592, 1.0758811855205128, -0.47092941513865666, 0.24307725170843006, 1.3408713657082527, 0.9372211242244086, 0.746745272172156, -0.826479025166701, -0.2005176100786385, -0.8823666526276142, -0.2867709049516592, -1.7949470866113597, -1.5194005623328983, -2.1572459946348657, 0.5688883826761171, -0.702884575658961, -0.8000741369080921, -0.27548375557694627, -4.364583499390056, 0.8067121360473769, 0.9740852021006478, 0.47362234638194467, 0.06632122172276392, -0.1440277817850779, -0.3497590170310852, 0.19068162229536328, 0.4390048212924756, 0.028856521832984927, -0.478525938352815, -1.797853294938206, -1.2777946840848708, 0.6336664957984173, 0.4349747000278921, 1.2334342684753938, -1.5266704055897058, 2.202292088441342, 1.317270449962564, -0.16122488419451264]

fig = Figure(size = (800, 600))
ax = Axis(fig[1,1], xlabel = "CF Parameter Index", ylabel = "Value", xgridvisible = false, ygridvisible = false)
lines!(ax, 1:length(nn_cf_old), nn_cf_old, label = "Offline Learned", color = :blue)
lines!(ax, 1:length(nn_cf_new), nn_cf_new, label = "Online Calibration", color = :red)
axislegend(ax)
save("cf_parameter_comparison_nearest_to_mean.png", fig)
