#= End-to-end test
Runs a perfect model calibration, calibrating on the parameter `astronomical_unit`
with top-of-atmosphere radiative shortwave flux in the loss function.

The calibration is run twice, once on the backend obtained via `get_backend()`
and once on the `JuliaBackend`. The output of each calibration is tested individually
and compared to ensure reproducibility.
=#
import ClimaAtmos as CA
import ClimaAnalysis: SimDir, get, slice, average_xy
import CairoMakie
import JLD2
import LinearAlgebra: I
import EnsembleKalmanProcesses as EKP
import Statistics: var, mean
using Test
using Revise

import ClimaCalibrate as CAL
import ClimaAtmos as CA
const experiment_dir = joinpath(pkgdir(CA), "calibration", "experiments", "gcm_driven_scm")
# const experiment_dir = "/home/jschmitt/ClimaAtmos.jl/calibration/experiments/gcm_driven_scm"
const model_interface =
    joinpath(pkgdir(CA), "calibration", "model_interface.jl")
const output_dir = joinpath("output", "gcm_driven_scm")

includet(model_interface)

# load observation map
includet("observation_map.jl")




# # Observation map
# function CAL.observation_map(iteration)
#     ensemble_size = 10
#     single_member_dims = (1,)
#     G_ensemble = Array{Float64}(undef, single_member_dims..., ensemble_size)

#     for m in 1:ensemble_size
#         member_path = CAL.path_to_ensemble_member(output_dir, iteration, m)
#         simdir_path = joinpath(member_path, "output_active")
#         if isdir(simdir_path)
#             simdir = SimDir(simdir_path)
#             G_ensemble[:, m] .= process_member_data(simdir)
#         else
#             G_ensemble[:, m] .= NaN
#         end
#     end
#     return G_ensemble
# end

# function process_member_data(simdir::SimDir)
#     isempty(simdir.vars) && return NaN
#     rsut =
#         get(simdir; short_name = "rsut", reduction = "average", period = "30d")
#     return slice(average_xy(rsut); time = 30).data
# end

# Generate observations
obs_path = joinpath(experiment_dir, "observations.jld2")
if !isfile(obs_path)
    @info "Generating observations"
    config = CA.AtmosConfig(joinpath(experiment_dir, "model_config.yml"))
    simulation = CA.get_simulation(config)
    CA.solve_atmos!(simulation)
    observations = Vector{Float64}(undef, 1)
    observations .= process_member_data(SimDir(simulation.output_dir))
    JLD2.save_object(obs_path, observations)
end

# Initialize experiment data
astronomical_unit = 149_597_870_000
observations = JLD2.load_object(obs_path)
noise = 0.1 * I
n_iterations = 3
ensemble_size = 10
prior = CAL.get_prior(joinpath(experiment_dir, "prior.toml"))
experiment_config = CAL.ExperimentConfig(;
    n_iterations,
    ensemble_size,
    observations,
    noise,
    output_dir,
    prior,
)
# Run calibration
julia_eki = CAL.calibrate(CAL.JuliaBackend, experiment_config)