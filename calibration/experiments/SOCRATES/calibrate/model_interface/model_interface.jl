using ClimaAtmos: ClimaAtmos
using ClimaCalibrate: ClimaCalibrate
using ClimaAnalysis: ClimaAnalysis
using EnsembleKalmanProcesses: EnsembleKalmanProcesses as EKP

using Dates: Dates
using JLD2: JLD2
using Random: Random
using Statistics: Statistics
using YAML: YAML
using TOML: TOML
using NCDatasets: NCDatasets as NC
using SOCRATESSingleColumnForcings: SOCRATESSingleColumnForcings as SSCF

include(joinpath(@__DIR__, "utils.jl"))
include(joinpath(@__DIR__, "..", "forcing", "generate_climacolumn.jl"))
include(joinpath(@__DIR__, "socrates_setup.jl"))

"""
    SOCRATESAtmosModelInterface <: ClimaCalibrate.AbstractModelInterface

Multi-case SOCRATES column SCM calibration against Atlas LES.

`cases` lists the 11 Obs/ERA cases. Observations are one `EKP.Observation` per
case (hydrometeor profiles + paths). Forward model runs all cases for each
ensemble member.
"""
struct SOCRATESAtmosModelInterface{C} <: ClimaCalibrate.AbstractModelInterface
    config::String
    output_dir::String
    experiment_config::Dict
    cases::Vector{SOCRATESCase}
    diagnostic_dicts::C
end

function SOCRATESAtmosModelInterface(
    config::AbstractString,
    output_dir::AbstractString,
    experiment_config::Dict;
    cases = load_cases(experiment_config),
)
    ispath(config) || error("$config is not a filepath")
    endswith(config, ".yml") || error("$config is not a YAML file")
    config = abspath(config)
    isdir(output_dir) || mkpath(output_dir)
    output_dir = abspath(output_dir)

    y_vars = String.(experiment_config["y_var_names"])
    diagnostic_dicts = [
        Dict(
            "short_name" => y_vars,
            "period" => "10mins",
            "reduction_time" => "average",
        ),
    ]
    foreach(validate_case!, cases)
    return SOCRATESAtmosModelInterface(
        config,
        output_dir,
        experiment_config,
        cases,
        diagnostic_dicts,
    )
end

function ClimaCalibrate.model_interface_filepath(::SOCRATESAtmosModelInterface)
    return abspath(joinpath(@__DIR__, "model_interface.jl"))
end

include(joinpath(@__DIR__, "generate_observations.jl"))
include(joinpath(@__DIR__, "forward_model.jl"))
include(joinpath(@__DIR__, "observation_map.jl"))
