import ClimaAtmos as CA
import ClimaCalibrate
import ClimaCalibrate: ObservationRecipe, EnsembleBuilder, Checker
import ClimaAnalysis
import EnsembleKalmanProcesses as EKP

import CairoMakie

import ClimaComms
ClimaComms.@import_required_backends

import Dates
import Random
import Statistics

"""
    PerfectAtmosModelInterface <: ClimaCalibrate.AbstractModelInterface

An interface for conducting a perfect model calibration.

To generate the observations, `make_observation_vec` is called which start an
atmos simulation with `config` and generate `EKP.Observation`s. The variables
of interest are determined by `diagnostic_dicts` whose schema is the same as
the one used by the `AtmosConfig`.

The calibration starts by running `N` forward model simulations via
`ClimaCalibrate.forward_model`. The simulation outputs is transformed into the
G ensemble matrix via `ClimaCalibrate.observation_map`. Any additional post
analysis of the iteration is done by `ClimaCalibrate.analyze_iteration`.

The calibration is saved in `output_dir`.
"""
struct PerfectAtmosModelInterface{DIAGS <: Vector} <: ClimaCalibrate.AbstractModelInterface
    "A filepath to a configuration file for a ClimaAtmos simulation"
    config::String

    "The directory to save the calibration to"
    output_dir::String

    "A vector of dictionaries following the schema specified by AtmosConfig"
    diagnostic_dicts::DIAGS
end

"""
    PerfectAtmosModelInterface(config, output_dir, diagnostic_dicts)

Constructor for `PerfectAtmosModelInterface`.

# Arguments

`config`: A filepath to a configuration file for a `ClimaAtmos` simulation.
`output_dir`: A filepath to a directory for storing the outputs of the
calibration.
`diagnostic_dicts`: A dictionary or vector of dictionary that specify which
diagnostics should be used for calibration.
"""
function PerfectAtmosModelInterface(config, output_dir, diagnostic_dicts)
    # Validation of config file
    ispath(config) || error("$config is not a filepath")
    endswith(config, "yml") || error("$config is not a YAMl file")
    config = abspath(config)

    # Responsible for making output directory
    isdir(output_dir) || mkpath(output_dir)
    output_dir = abspath(output_dir)

    diagnostic_dicts isa Dict && (diagnostic_dicts = [diagnostic_dicts])

    for diag_dict in diagnostic_dicts
        haskey(diag_dict, "short_name") ||
            error("The key (short_name) does not exist in the diagnostic_dict")
        haskey(diag_dict, "period") ||
            error("The key (period) does not exist in the diagnostic_dict")
        haskey(diag_dict, "reduction_time") ||
            error("The key (reduction_time) does not exist in the diagnostic_dict")
    end
    return PerfectAtmosModelInterface(config, output_dir, diagnostic_dicts)
end

"""
    ClimaCalibrate.model_interface_filepath(::PerfectAtmosModelInterface)

This is used by the `ClimaCalibrate.HPCBackend`s to know which file to include
for the ensemble members to create a `PerfectAtmosModelInterface` and its
functions.
"""
function ClimaCalibrate.model_interface_filepath(::PerfectAtmosModelInterface)
    return @__FILE__
end

include("postprocess_data.jl")
include("generate_observations.jl")
include("forward_model.jl")
include("observation_map.jl")
include("postanalyze_iteration.jl")
include("utils.jl")
