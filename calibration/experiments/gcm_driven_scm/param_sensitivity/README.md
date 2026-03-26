# SCM Parameter Sensitivity Analysis

This folder contains scripts for running parameter sensitivity analysis on the GCM-driven SCM simulations.

## Overview

The sensitivity analysis sweeps over parameters **one at a time**, keeping all other parameters at their baseline values. This allows you to understand how each parameter independently affects the simulation outputs.

## Files

- `sensitivity_config.toml` - Configuration file specifying parameters to sweep and their values
- `run_sensitivity.jl` - Main runner using Slurm-managed distributed workers
- `run_sensitivity_simple.jl` - Simpler runner using local workers (good for testing)
- `sensitivity_helpers.jl` - Helper functions for running simulations
- `analyze_sensitivity.jl` - Post-processing and plotting script
- `submit_sensitivity.sbatch` - Slurm submission script

## Quick Start

### 1. Configure the sweep

Edit `sensitivity_config.toml` to specify:
- Output directory
- Model configuration file
- Base parameter TOML file
- Parameters to sweep and their values
- Number of workers and Slurm settings

### 2. Run in dry-run mode first

```bash
cd calibration/experiments/gcm_driven_scm
julia --project=. param_sensitivity/run_sensitivity_simple.jl --config param_sensitivity/sensitivity_config.toml --dry-run
```

This will show you the simulation configurations without running anything.

### 3. Run the sensitivity analysis

**Option A: Simple local workers (good for testing)**
```bash
julia --project=. param_sensitivity/run_sensitivity_simple.jl \
    --config param_sensitivity/sensitivity_config.toml \
    --workers 8
```

**Option B: Slurm-managed workers (for production)**
```bash
sbatch param_sensitivity/submit_sensitivity.sbatch
```

### 4. Analyze results

```bash
julia --project=. param_sensitivity/analyze_sensitivity.jl \
    --output-dir /path/to/sensitivity/output \
    --vars thetaa,hus,clw \
    --save-plots
```

## Configuration Reference

### `[paths]` section

```toml
[paths]
model_config = "model_config_prognostic_v2.yml"  # Model config relative to gcm_driven_scm
base_toml = "scm_tomls/prognostic_edmfx_dev.toml"  # Base parameters
output_dir = "/path/to/output"  # Where to save results
```

### `[run]` section

```toml
[run]
num_workers = 10  # Number of parallel workers
dry_run = false   # Set true to only generate configs
max_cases = 3     # Limit number of LES cases (null = all)
cfsite_cases = null  # Specific cfsites to run (null = all)
```

### `[forcing]` section

Specifies the external forcing files to use. The forcing file path is constructed as:
`{base_path}/{model}_{experiment}.2004-2008.{MM}.nc`

```toml
[forcing]
months = [7]              # Month(s) to run: 7 = July, [7, 10] = July and October
model = "HadGEM2-A"       # GCM model name
experiment = "amip"       # Experiment type
base_path = "/resnick/groups/esm/zhaoyi/GCMForcedLES/forcing/corrected"
```

This generates cases as the cross-product of `cfsite_cases × months`. For example:
- `cfsite_cases = [2, 3, 4]` and `months = [7, 10]` → 6 cases total

### `[forcing_toml_files]` section

Specifies forcing-type-specific TOML files for shallow vs deep convection cases:

```toml
[forcing_toml_files]
shallow = "scm_tomls/gcmdriven_relaxation_shallow_forcing.toml"
deep = "scm_tomls/gcmdriven_relaxation_deep_forcing.toml"
```

The script automatically determines cfsite type (shallow/deep) from the cfsite number and applies the appropriate relaxation settings.

### `[parameters.*]` sections

Each parameter to sweep gets its own section:

```toml
[parameters.mixing_length_diss_coeff]
values = [0.1, 0.15, 0.22, 0.3, 0.4]  # Values to sweep
type = "float"  # Parameter type: "float", "int", or "vector"
```

For vector parameters:
```toml
[parameters.entr_param_vec]
values = [
    [-1.0, -1.0, 0.1],
    [0.0, 0.0, 0.2],
    [1.0, 1.0, 0.3],
]
type = "vector"
```

### `[groups.*]` sections (coupled parameters)

To make parameters change together:

```toml
[groups.inv_tau_group]
parameters = ["entr_inv_tau", "detr_inv_tau"]
```

The first parameter is the "driver" - its sweep values are used. All other parameters in the group take on the same values.

## Output Structure

```
output_dir/
├── configs/
│   ├── sensitivity_config.toml      # Copy of config
│   ├── model_config_prognostic.yml  # Copy of model config
│   ├── base_parameters.toml         # Baseline parameters
│   ├── baseline_parameters.toml
│   └── params/                       # Generated parameter override files
│       ├── mixing_length_diss_coeff_0.1.toml
│       ├── mixing_length_diss_coeff_0.22.toml
│       └── ...
├── manifest.toml                     # List of all configurations
├── results_summary.toml              # Run results and status
├── baseline/                         # Baseline simulation (all params at default)
│   ├── cfsite_17_HadGEM2-A_amip_07/
│   ├── cfsite_18_HadGEM2-A_amip_07/
│   └── ...
└── mixing_length_diss_coeff/         # One folder per parameter
    ├── val_0.1/                      # Descriptive value in folder name
    │   ├── cfsite_17_HadGEM2-A_amip_07/
    │   └── ...
    ├── val_0.15/
    ├── val_0.22/
    └── val_0.3/
```

Each cfsite folder contains the simulation output with standard ClimaAtmos diagnostics.

## Tips

1. **Start small**: Use `max_cases = 1` or `--baseline-only` to test the setup
2. **Test single params**: Use `--single-param mixing_length_diss_coeff` to test one parameter
3. **Check logs**: Worker logs are in the output `logs/` directory
4. **Resume runs**: The manifest tracks all configs; you can re-run failed configs individually

## Computing Sensitivity Metrics

The analysis script computes RMSE from baseline for each parameter value, allowing you to rank parameters by their sensitivity:

```bash
julia --project=. param_sensitivity/analyze_sensitivity.jl \
    --output-dir /path/to/output \
    --vars thetaa,hus
```

This prints a table ranking parameters by their impact on each output variable.
