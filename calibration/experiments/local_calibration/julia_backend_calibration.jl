import ClimaAtmos as CA
import ClimaAnalysis: SimDir, get, slice, average_xy
import CairoMakie
import JLD2
import LinearAlgebra: I
import EnsembleKalmanProcesses as EKP
import Statistics: var, mean
using Test
using Revise

# load observation map
include("observation_map.jl")


# add processes
using Distributed
addprocs(3)

@everywhere begin
    using Revise
    import ClimaCalibrate as CAL
    import ClimaAtmos as CA
    const experiment_dir = joinpath(pkgdir(CA), "calibration", "experiments", "local_calibration")
    const model_interface =
        joinpath(pkgdir(CA), "calibration", "model_interface.jl")
    const output_dir = joinpath("output", "local_calibration")
    # include model interface
    include(model_interface)
end


cal_ex_config = CAL.ExperimentConfig("experiment_config.yml")
result_calibration = CAL.calibrate(CAL.JuliaBackend, cal_ex_config, 
    scheduler = EKP.DataMisfitController(terminate_at = 1),
    localization_method = EKP.NoLocalization(),
    failure_handler_method = EKP.SampleSuccGauss(),
    accelerator = EKP.DefaultAccelerator(),
)