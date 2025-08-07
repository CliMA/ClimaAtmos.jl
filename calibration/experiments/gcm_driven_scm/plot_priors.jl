import ClimaCalibrate as CAL
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
import YAML

import YAML
import JLD2
using LinearAlgebra
using Distributions
using Plots

include("helper_funcs.jl")



# const prior = CAL.get_prior("./prior_prognostic_pi_entr_smooth_entr_detr_impl_0M.toml")
# p1 = plot(prior)
# savefig(p1, "prior.pdf")

# usefule for specifing bounded parameters in config like:
# ParameterDistribution(Parameterized(Normal(10.1871, 0.1)), bounded_below(0.0), "tes2")

prior_1 = constrained_gaussian("turb_entr_scale", 0.001, 0.0005, 0, Inf)
transformed_mu_1 = prior_1.distribution[1].distribution.μ
prior_2 = constrained_gaussian("turb_entr_power", 5000.0, 1.000, 0, Inf)
transformed_mu_2 = prior_2.distribution[1].distribution.μ

# p1 = plot(prior)
# p_dist1 = plot(parm_dist1)
# savefig(p_dist1, "./prior_dist1.pdf")

p_dist2 = plot(parm_dist2)
savefig(p_dist2, "./prior_dist2.pdf")

# p4 = constrained_gaussian("entr_inv_tau", 1e-4, 1e-4, 0, Inf)
# p5 = constrained_gaussian("specific_humidity_precipitation_threshold", 5e-6, 3e-7, 0, Inf)
