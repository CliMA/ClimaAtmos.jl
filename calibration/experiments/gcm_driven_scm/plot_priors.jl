import ClimaCalibrate as CAL
import ClimaAtmos as CA
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
import YAML

import YAML
import JLD2
using LinearAlgebra
using Distributions
using Plots

include("helper_funcs.jl")
include("observation_map.jl")
include("get_les_metadata.jl")
    

output_dir = "output_deep_conv_etki_exp/exp_10"

config_dict =
    YAML.load_file(joinpath(output_dir, "configs", "experiment_config.yml"))

model_config_dict =
    YAML.load_file(joinpath(output_dir, "configs", "model_config.yml"))



const prior = CAL.get_prior(joinpath(output_dir, "configs", "prior.toml"))


# [turb_entr_param_vec]
# prior = "VectorOfParameterized([Normal(-3.149, 0.554), Normal(8.228, 0.362)])"
# constraint = "[bounded_below(0.0), bounded_below(0.0)]"
# type = "float"


# p1 = ParameterDistribution(Parameterized(Normal(8.228, 0.45)), bounded_below(0.0), "tes1")
# p2 = ParameterDistribution(Parameterized( Normal(8.228, 0.362)), bounded_below(0.0), "tes1")
p3 = constrained_gaussian("rain_autoconversion_timescale", 5000.0, 3000.0, 0, Inf)
# p4 = constrained_gaussian("entr_inv_tau", 1e-4, 1e-4, 0, Inf)

plot(p3, size=(800, 600))
# plot(p4, size=(800, 600), ylims=(0, 1e3))
# plot(p2)
# prior = "VectorOfParameterized([Normal(-3.149, 0.554), Normal(8.228, 0.362)])"
# constraint = "[bounded_below(0.0), bounded_below(0.0)]"

# p1 = 