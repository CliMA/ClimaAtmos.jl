using Revise 
using ClimaAnalysis
import ClimaAtmos as CA
using NCDatasets
import YAML
import Plots
import Glob
using CairoMakie
import ClimaAnalysis.Visualize as viz
import ClimaAnalysis.Utils: kwargs
import JLD2

import Interpolations
using Statistics

import TOML, YAML
import JLD2
using Distributions
import EnsembleKalmanProcesses as EKP
using EnsembleKalmanProcesses.ParameterDistributions
using EnsembleKalmanProcesses.TOMLInterface
import ClimaCalibrate as CAL

# code to get prior from eki object 
sim_path = "/scratch/julian/calibrations/precal_23/"
eki_initial = JLD2.load_object(joinpath(sim_path, "iteration_000/eki_file.jld2"))

eki_final = JLD2.load_object(joinpath(sim_path, "iteration_010/eki_file.jld2"));
prior = CAL.get_prior("../calibrate/toml/prior_1m.toml");
t = TOML.parsefile("../calibrate/toml/prior_1m.toml");


function get_ϕ_mean_final_nn(eki, prior, toml_final = "calibrate/toml/final_1m.toml")
    """
    Get the ensemble member of the final iteration closest to the ensemble mean 
    as measured in unconstrained space wrt 2-norm
    """
    u_final = EKP.get_u_final(eki)
    u_mean_final = EKP.get_u_mean_final(eki)
    best_member = argmin(sum((u_final .- u_mean_final) .^2, dims = 1))[2]

    # pick the best one in physical space
    ϕ_mean_final = EKP.get_ϕ_final(prior, eki)[:, best_member]

    # index and save to toml
    batch_idx = EKP.batch(prior)
    param_names = prior.name

    toml_dict = Dict(zip(param_names, [Dict("value" => length(ϕ_mean_final[idx]) == 1 ? ϕ_mean_final[idx][1] : ϕ_mean_final[idx]) for idx in batch_idx]))

    toml_string = TOML.print(toml_dict)
    open(toml_final, "w") do io 
        TOML.print(io, toml_dict)
    end
end

get_ϕ_mean_final_nn(eki_final, prior, "calibrate/toml/final_1m_nn.toml")


# run final model 
config_dict = YAML.load_file("calibrate/prognostic_edmfx_gcmdriven_column.yml")
config_dict["toml"] = ["calibrate/toml/final_1m_nn.toml"]
config_dict["external_forcing_file"] = "data/era5_monthly_forcing_1.nc"

config = CA.AtmosConfig(config_dict; job_id = "posterior_1m_nn")
simulation = CA.get_simulation(config)
sol_res = CA.solve_atmos!(simulation)

# get prior model config 
config_dict = YAML.load_file("calibrate/prognostic_edmfx_gcmdriven_column.yml")
config_dict["toml"] = ["calibrate/toml/initial.toml"]
config_dict["external_forcing_file"] = ".data/era5_monthly_forcing_1.nc"

config = CA.AtmosConfig(config_dict; job_id = "initial_1m_nn")
simulation = CA.get_simulation(config)
sol_res = CA.solve_atmos!(simulation)
println("done")


