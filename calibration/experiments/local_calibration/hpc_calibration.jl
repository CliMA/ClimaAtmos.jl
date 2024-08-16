import ClimaAtmos as CA
import ClimaAnalysis: SimDir, get, slice, average_xy
import CairoMakie
import JLD2
import LinearAlgebra: I
import EnsembleKalmanProcesses as EKP
import Statistics: var, mean
using Test

# load observation map
include("observation_map.jl")

import ClimaCalibrate as CAL
import ClimaAtmos as CA
const experiment_dir = joinpath(pkgdir(CA), "calibration", "experiments", "local_calibration")
const model_interface =
    joinpath(pkgdir(CA), "calibration", "model_interface.jl")
const output_dir = joinpath("output", "local_calibration")
# include model interface
include(model_interface)


cal_ex_config = CAL.ExperimentConfig("experiment_config.yml")
hpc_kwargs = CAL.kwargs(time = 30, mem = "16G") # , reservation = "clima"
result_calibration = CAL.calibrate(CAL.CaltechHPCBackend, cal_ex_config; 
    hpc_kwargs = hpc_kwargs,
    scheduler = EKP.DataMisfitController(terminate_at = 10), # don't terminate at 1 but use adaptive timestepping
    localization_method = EKP.NoLocalization(),
    failure_handler_method = EKP.SampleSuccGauss(),
    accelerator = EKP.DefaultAccelerator(),
    verbose = true,
)