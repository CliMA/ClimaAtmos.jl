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
const model_interface =
    joinpath(pkgdir(CA), "calibration", "model_interface.jl")

# include model interface
include(model_interface)
output_dir  = config_dict["output_dir"]
if !isdir(output_dir)
    mkdir(output_dir)
end
vi_wght = 10 # we'll upweight the importance of radiative vi's to compete with the profiles
norm_factors_dict = Dict(
    "thetaa" => [306.172, 8.07383, 1],
    "hus" => [0.0063752, 0.00471147, 1],
    "husv" => [0.0063752, 0.00471147, 1],
    "clw" => [2.67537e-6, 4.44155e-6, 1],
    "lwp" => [1, .1^2, 1],
    "prw" => [30, 3^2, 1],
    "clwvi" => [1.25, .1^2, 1],
    "clvi" => [1100, 100^2, 1],
    "husvi" => [32, 3^2, 1],
    "hurvi" => [.45, .03^2, 1],
    "rlut" => [279.5, 1^2, vi_wght],
    "rlutcs" => [290, 1^2, vi_wght],
    "rsut" => [283.5, 1^2, vi_wght],
    "rsutcs" => [123, 1^2, vi_wght],
)
try
    JLD2.jldsave(
        joinpath(output_dir, "norm_factors.jld2");
        norm_factors_dict = norm_factors_dict,
    )
catch
    println("Norm Factors already saved: $(joinpath(output_dir, "norm_factors.jld2"))")
end

cal_ex_config = CAL.ExperimentConfig("experiment_config.yml")
hpc_kwargs = CAL.kwargs(time = 60, mem = "16G") # , reservation = "clima"
result_calibration = CAL.calibrate(CAL.CaltechHPCBackend, cal_ex_config; 
    hpc_kwargs = hpc_kwargs,
    reruns = 0,
    scheduler = EKP.DataMisfitController(terminate_at = 10), # DataMisfitController(terminate_at = 10) don't terminate at 1 but use adaptive timestepping
    localization_method = EKP.NoLocalization(),
    failure_handler_method = EKP.SampleSuccGauss(),
    accelerator = EKP.DefaultAccelerator(),
    verbose = true,
)