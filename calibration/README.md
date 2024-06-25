# ClimaAtmos Calibration Experiments

This directory contains the model interface (`model_interface.jl`) for ClimaCalibrate and folders (within `experiments/`) for reproducing calibration experiments.

## Current Experiments

### sphere_held_suarez_rhoe_equilmoist

A perfect-model calibration, using ClimaAtmos v0.24.0.

- Configuration: equilmoist, 0-moment microphysics scheme, spherical with Held-Suarez forcing.
- Observational data: zonal and 60-day time average air temperature at 242m altitude. Provided by [ClimaArtifacts](https://github.com/CliMA/ClimaArtifacts/tree/main/atmos_held_suarez_obs).
- Parameter being calibrated: `equator_pole_temperature_gradient_wet`

For more details, see the experiment directory.

To reproduce the results, on the Caltech central cluster run

```sbatch calibration/experiments/sphere_held_suarez_rhoe_equilmoist/pipeline.sbatch```

## New Experiments

To set up your own experiment, please follow the [setup guide](https://clima.github.io/ClimaCalibrate.jl/dev/atmos_setup_guide/).
