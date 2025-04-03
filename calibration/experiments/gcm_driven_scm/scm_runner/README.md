## GCM-Driven SCM Runner

### Purpose
- **What it does**: Runs the ClimaAtmos single column model (SCM) for every LES configuration in the library using a specified parameter TOML, writing outputs per case.

### Prerequisites
- **Configs present**:
  - `../experiment_config.yml` with at least `output_dir`, and optionally `forcing_toml_files` mapping.
  - `model_config_prognostic_runner.yml` in this directory.
- **Parameter TOML**: Path to a TOML file with EDMF parameters (e.g., from calibration output).
- **Environment**: Same Julia project used for calibration; on HPC, the module stack in `scm_runner.sbatch` loads `climacommon`.

### Usage
- **Submit on HPC**:
```bash
sbatch scm_runner.sbatch \
  --run_output_dir=/absolute/path/to/scm_runs/<run_name> \
  --parameter_path=/absolute/path/to/parameters.toml
```
- **Run interactively**:
```bash
julia --project scm_runner.jl \
  --run_output_dir=/absolute/path/to/scm_runs/<run_name> \
  --parameter_path=/absolute/path/to/parameters.toml
```

### Arguments
- **`--run_output_dir`**: Base directory for all SCM outputs. The runner creates one subdirectory per case.
- **`--parameter_path`**: Absolute or relative path to the parameter TOML to apply to all cases.

### What the runner does
- Builds per-case configs from `get_les_calibration_library_runner()` and `model_config_prognostic_runner.yml`.
- Sets `output_dir` per case as:
  - ``<run_output_dir>/cfsite_<N>_<forcing_model>_<experiment>_<month>/``
- Adds forcing-specific TOML if configured in `../experiment_config.yml` under `forcing_toml_files`.
  - Files are expected under ``<experiment_config.output_dir>/configs/``.
- Runs cases in parallel via `Distributed.pmap` with `NUM_WORKERS` defined in `scm_runner.jl`.

### Output layout
- One folder per case under `--run_output_dir`, containing the standard ClimaAtmos outputs for that case.

### Tips
- **Parallelism**: Keep `NUM_WORKERS` in `scm_runner.jl` aligned with `#SBATCH --ntasks` in `scm_runner.sbatch` and node resources.
- **Resources**: Adjust `--mem`, `--time`, and `--nodes` in `scm_runner.sbatch` based on case count and model resolution.
- **Best particle**: To find the nearest-to-mean particle from calibration, see `get_nearest_particle.jl` in the parent directory.

