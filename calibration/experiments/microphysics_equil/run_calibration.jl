using ClimaCalibrate
import ClimaCalibrate as CAL
import ClimaAtmos as CA
import EnsembleKalmanProcesses as EKP
import YAML
import TOML
using Distributions
using Distributed
using Random
using Logging

import JLD2
using LinearAlgebra

include("model_interface.jl")
include("observation_map.jl")

"""
FROM TUTORIAL
"""

ensemble_size = 10
n_iterations = 3

# change?
noise = 0.1 * I

# change this
prior = constrained_gaussian("pow_ice", 5, 4, 0.01, 10)

@info "Generating observations"
parameter_file = CAL.parameter_path(output_dir, 0, 0)
mkpath(dirname(parameter_file))
touch(parameter_file)
simulation = CAL.forward_model(0, 0)

observations = Vector{Float64}(undef, 1)
observations .= process_member_data(SimDir(simulation.output_dir))

@info "Run EKI"
eki = CAL.calibrate(
    CAL.WorkerBackend,
    ensemble_size,
    n_iterations,
    observations,
    noise,
    prior,
    output_dir,
)