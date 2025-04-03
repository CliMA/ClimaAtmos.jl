# Overview of Calibration Pipeline for GCM-Driven Single Column Model (EDMF)

## Pipeline Components

### Configuration Files
- `experiment_config.yml` - Configuration of calibration settings
  - Defines spatiotemporal calibration window
  - Specifies required pipeline file paths
  - Controls batch processing parameters

- `model_config_**.yml` - Configuration for ClimaAtmos single column model
  - Defines model-specific parameters
  - Allows for multiple model configuration variants

## Best Practices
- Ensure `batch_size` matches available LES configurations
- Verify normalization factors for each variable
- Monitor ensemble convergence using provided plotting tools

This setup provide tools for calibrating both prognostic and diagnostic EDMF variants to LES profiles, given the same forcings and boundary conditions. The gcm-driven EDMF setup is employed in single-column mode, which uses both interactive radiation and surface fluxes. Forcing profiles include resolved eddy advection, horizontal advection, subsidence, and GCM state relaxation. The setup is run to the top of the atmosphere to compute radiation, but calibration statistics are only computed on the calibration grid `z_cal_grid`.

LES profiles are available for different geolocations ("cfsites"), spanning seasons, forcing host models, and climates (AMIP, AMIP4K). A given LES simulation is referred to as a "configuration". Calibrations employ batching by default and stack multiple configurations (a number equal to the `batch_size`) in a given iteration. The observation vector for a single configuration is formed by concatenating profiles across calibration variables, where each geophysical variable is normalized to have approximately unit variance and zero mean. These variable-by-variable normalization factors are precomputed (`norm_factors_dict`) and applied to all observations. Following this operation, the spatiotemporal calibration window is applied and temporal means are computed to form the observation vector `y`. Because variables are normalized to have 0 mean and unit variance, a constant diagonal noise matrix is used (configurable as `const_noise`).


### Observation Map
1. **Time-mean**: Time-mean of profiles taken between [`y_t_start_sec`, `y_t_end_sec`] for `y` and [`g_t_start_sec`, `g_t_end_sec`] for `G`.
2. **Interpolation**: Case-specific (i.e., "shallow", "deep") interpolation to the calibration grid, defined with stretch-grid parameters in `z_cal_grid`.
3. **Normalization**: Variable-specific normalization using the mean and standard deviation defined in `norm_factors_by_var`. Optionally, take log of variables using `log_vars` before normalization.
4. **Concatenation**: Stack across cases in a batch, forming `y`, `G`.

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
- `julia --project plot_eki.jl` - plot eki metrics [loss, variance-weighted loss] and `y`, `G` vectors vs iteration, display best particles

## Troubleshooting

- **Memory Issues**: If you encounter out of memory errors, increase the memory allocation in both `run_calibration.sbatch` and the `experiment_config.yml` file. This is particularly important when working with larger batch sizes. Example error message:
  ```
  srun: error: hpc-92-10: task 9: Out Of Memory
  ```