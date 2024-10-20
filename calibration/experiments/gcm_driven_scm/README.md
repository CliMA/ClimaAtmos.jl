# Overview of Calibration Pipeline for GCM-Driven Single Column Model (EDMF)

This setup provide tools for calibrating both prognostic and diagnostic EDMF variants to LES profiles, given the same forcings and boundary conditions. The gcm-driven EDMF setup is employed in single-column mode, which uses both interactive radiation and surface fluxes. Forcing profiles include resolved eddy advection, horizontal advection, subsidence, and GCM state relaxation. The setup is run to the top of the atmosphere to compute radiation, but calibrations statistics are computed only on the lower 4km (`z_max`), where LES output is available.

LES profiles are available for different geolocations ("cfsites"), spanning seasons, forcing host models, and climates (AMIP, AMIP4K). A given LES simulation is referred to as a "configuration". Calibrations employ batching by default and stack multiple configurations (a number equal to the `batch_size`) in a given iteration. The observation vector for a single configuration is formed by concatenating profiles across calibration variables, where each geophysical variable is normalized to have approximately unit variance and zero mean. These variable-by-variable normalization factors are precomputed (`norm_factors_dict`) and applied to all observations. Following this operation, the spatiotemporal calibration window is applied and temporal means are computed to form the observation vector `y`. Because variables are normalized to have 0 mean and unit variance, a constant diagonal noise matrix is used (configurable as `const_noise`).


## Getting Started

### Define calibration and model configurations:
- `experiment_config.yml` - Configuration of calibration settings, including spatiotemporal calibration window and required pipeline file paths.
- `run_calibration.jl` - run script for calibration pipeline. EKI settings and hyperparameters can be modified where CAL.initialize is called.
- `model_config_**.yml` - Config file for underlying ClimaAtmos single column model
- `get_les_metadata.jl` - (Re)Define `get_les_calibration_library()` to specify which LES configurations to use. Set `batch_size` in the `experiment_config.yml` accordingly (<= the number of cases).

### Run with:
- `sbatch run_calibration.sbatch` -  schedules and runs calibration pipeline end-to-end using HPC resources
- `julia --project run_calibration.jl` - interactively runs calibration end-to-end using HPC resources, streaming to a Julia REPL

### Analyze output with:
- `julia --project plot_ensemble.jl` - plots vertical profiles of all ensemble members in a given iteration, given path to calibration output
- `julia --project edmf_ensemble_stats.jl` - computes and plots metrics offline [i.e., root mean squared error (RMSE)] as a function of iteration, given path to calibration output.
- `julia --project plot_eki.jl` - plot eki metrics [loss, var-weighted loss] and `y`, `g` vectors vs iteration, display best particles


