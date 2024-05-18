# Overview of Calibration Pipeline for GCM-Driven Single Column Model (EDMF)

This setup provide tools for calibrating both prognostic and diagnostic EDMF variants to LES profiles, given the same forcings and boundary conditions. The gcm-driven EDMF setup is employed in single-column mode, which uses both interactive radiation and surface fluxes. Forcing profiles include resolved eddy advection, horizontal advection, subsidence, and GCM state relaxation. The setup is run to the top of the atmosphere to compute radiation, but calibrations statistics are computed only on the lower 4km (`z_max`), where LES output is available.

LES profiles are available for different geolocations ("cfsites"), spanning seasons, forcing host models, and climates (AMIP, AMIP4K). A given LES simulation is referred to as a "configuration". Calibrations employ batching by default and stack multiple configurations (a number equal to the `batch_size`) in a given iteration. The observation vector for a single configuration is formed by concatenating profiles across calibration variables, where each geophysical variable is normalized to have approximately unit variance and zero mean. These variable-by-variable normalization factors are precomputed (`norm_factors_dict`) and applied to all observations. Following this operation, the spatiotemporal calibration window is applied and temporal means are computed to form the observation vector `y`. Because variables are normalized, a constant, diagonal noise matrix is used (configurable as `const_noise`).


## Getting Started

### Define calibration and model configurations:
- `experiment_config.yml` - Configuration of EKI hyperparameters and settings, spatiotemporal calibration window, required pipeline file paths
- `model_config_**.yml` - Config file for underlying ClimaAtmos single column model
- `get_les_metadata.jl` - (Re)Define `get_les_calibration_library()` to specify which LES configurations to use

### Run with:
- `run_calibration.jl` - runs calibration end-to-end using HPC resources

### Analyze output with:
- `plot_ensemble.jl` - plots vertical profiles of all ensemble members in a given iteration.


