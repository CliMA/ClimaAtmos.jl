"""
    SocratesCalibration

Calibrate the SOCRATES column model against the Atlas LES with EKI.

This layer adds the EKP/ClimaCalibrate machinery on top of the model and scoring layers. It runs
cases only through `SocratesModel.run_case` and scores only through `SocratesScoring`, so there is
one way to run a case and one definition of the misfit.

```julia
include(".../src/calibration/SocratesCalibration.jl")
const SC = SocratesCalibration

interface = SC.SocratesInterface(; output_dir = "calibrations")
prior = SC.default_prior()
ekp = SC.build_ekp(interface, prior; ensemble_size = 10, T_stops = [1.0, 10.0, 100.0])

SC.calibrate(backend, ekp, interface; prior, n_iterations = 20, T_stops = [1.0, 10.0, 100.0])
```

Loading this file also loads `SocratesScoring` and `SocratesModel`; the reverse is not true, so
running or scoring a case costs nothing from this layer.
"""
module SocratesCalibration

using ClimaAnalysis: ClimaAnalysis
using ClimaCalibrate: ClimaCalibrate
using EnsembleKalmanProcesses: EnsembleKalmanProcesses as EKP
using JLD2: JLD2
using LinearAlgebra: LinearAlgebra
using NaNStatistics: NaNStatistics
using Random: Random
using Statistics: Statistics

include(joinpath(@__DIR__, "..", "scoring", "SocratesScoring.jl"))
using .SocratesScoring: SocratesScoring as SS

# no exports, qualified uses only

include("observations.jl")
include("gmap.jl")
include("interface.jl")
include("prior.jl")
include("driver.jl")
include("ekp.jl")
include("postprocess.jl")

end # module