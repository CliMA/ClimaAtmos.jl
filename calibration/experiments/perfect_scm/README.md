# Perfect SCM experiment

This directory contains a perfect SCM calibration experiment which recovers the
known parameters from data generated from a CliMA simulation. You may find it
helpful to read the documentation of `EnsembleKalmanProcesses` (`EKP`) which
performs the optimization algorithms for calibration, `ClimaCalibrate` which
orchestrates the calibration, and `ClimaAnalysis` which performs the
postprocessing for calibration.

In general, a calibration consists of
- observational data,
- a prior parameter distribution,
- a forward model, which uses input parameters to return diagnostic output,
- an observation map, which maps the forward model's diagnostic output to a
  vector comparable to the observations.

The rest of the README.md will provides introductory information to the core
components of the calibration.

## Interface

The `PerfectAtmosModelInterface` object is passed to all functions in the
calibration experiment and `ClimaCalibrate`. As of now, the
`PerfectAtmosModelInterface` object consists of a config file used for
generating the simulation and the forward model, the output directory for
storing the results of the calibration, and a dictionary of diagnostics which
specifies which observations to use for the calibration. The
`PerfectAtmosModelInterface` struct is defined in
`model_interface/model_interface.jl`

This object should be constant for the calibration experiment when it is first
initialized and it is not suitable for mutation.

In the future, if you want to support different calibrations, you should use
multiple dispatch and create another interface type. It would be helpful to
define an abstract interface type and have the `PerfectAtmosInterface` subtype
it to reuse most of the behavior of the `PerfectAtmosModelInterface`.

## Observational data

The observational data is generated from a CliMA simulation. The functionality
to begin a simulation, generate diagnostics, and create a vector of
`EKP.Observation`s is in `model_interface/generate_observations.jl". To make
this step faster and simplify the steps for the observation map, only the
diagnostics used for the calibration are created. The parameters used for
generating the simulation data is the default parameters used for the config in
the `PerfectAtmosModelInterface` object. The postprocessing of the diagnostics
are done in `create_ekp_observations` which create the `EKP.Observation` which
includes the observational data as a vector and the corresponding covariance
matrix. ClimaAnalysis is used for the postprocessing of the diagnostics.
The diagnostics are preprocessed by `preprocess` in
`model_interface/postprocess_data.jl`.

The final result is a vector of `EKP.Observation`s that is saved to disk as a
JLD2 file. This leads to the possibility of caching the results of the
simulation if the next steps of the calibration fail, but this feature is not
yet supported.

## Prior parameter distribution

The prior parameter distribution is specified in the `run_calibration.jl`
script. This determines what parameters are used for the calibration.

## Forward model

For each iteration of the calibration, an ensemble of forward model evaluations
are ran by `ClimaCalibrate`. This is implemented by
`ClimaCalibrate.forward_model` in `model_interface/forward_model.jl`. The config
is preprocessed by
- updating the output directory,
- disabling the default diagnostics,
- disabling the checkpoints,
- updating the parameters with the parameters provided by `EKP`.

## Observation map

The observation map transform the simulation outputs from each of the ensemble
member and stores the results in the G ensemble matrix. This step is done
by `ClimaCalibrate.observation_map` in `model_interface/observation_map.jl`.
This step consists of loading the simulation outputs as
`ClimaAnalysis.OutputVar`s, calling `preprocess` on each `OutputVar`, and
relying on the metadata of the `EKP.Observation`s to automatically create the
G ensemble matrix.

## Postanalysis of iteration

At the end of each iteration, `ClimaCalibrate.analyze_iteration` is called.
This plots how the parameters and loss vary across the iterations. In addition,
plots of the columns of the G ensemble matrix ± standard deviation (from the
diagonal covariance matrix), and the true model data are made. This
functionality is in `model_interface/postanalyze_iteration.jl`.
