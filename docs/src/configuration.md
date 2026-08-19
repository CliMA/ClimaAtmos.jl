# Creating custom configurations

To create a custom configuration, first make a .yml file.
In the file, you can set configuration arguments as `key: value` pairs to override the default config.
Each value is coerced to the type of the corresponding default in
[`config/default_configs/default_config.yml`](https://github.com/CliMA/ClimaAtmos.jl/blob/main/config/default_configs/default_config.yml), so quoting does not
matter (a quoted `"true"` still becomes a `Bool`); a value that cannot be coerced
raises an error naming the key and the expected type. Keys that are not in the
default configuration produce a warning, or an error when `strict_config: true`
is set.

To start the model with a custom configuration, run:

```julia
import ClimaAtmos as CA

config = CA.AtmosConfig("path/to/config.yaml")
simulation = CA.AtmosSimulation(config)
CA.solve_atmos!(simulation)
```

## Example

Below is the default BOMEX configuration
([`config/model_configs/prognostic_edmfx_bomex_column.yml`](https://github.com/CliMA/ClimaAtmos.jl/blob/main/config/model_configs/prognostic_edmfx_bomex_column.yml)):

```yaml
initial_condition: "Bomex"
turbconv: "prognostic_edmfx"
implicit_diffusion: true
approximate_linear_solve_iters: 2
edmfx_entr_model: "Generalized"
edmfx_detr_model: "Generalized"
edmfx_sgs_mass_flux: true
edmfx_sgs_diffusive_flux: true
edmfx_nh_pressure: true
edmfx_vertical_diffusion: true
edmfx_filter: true
prognostic_tke: true
microphysics_model: "1M"
config: "column"
z_max: 4200
z_elem: 60
z_stretch: false
perturb_initstate: false
dt: "120secs"
t_end: "6hours"
dt_save_state_to_disk: "10mins"
toml: [toml/prognostic_edmfx_1M.toml]
netcdf_interpolation_num_points: [2, 2, 60]
ode_algo: "ARS222"
```

Keys can also point to artifacts. As artifacts are folders, we specify both the artifact name, as we would from the REPL, and file to read from, separated by a `/`. For example, to drive a single
column model with an external forcing file from GCM output, we include the following lines in the
configuration:

```yaml
initial_condition: "GCM"
external_forcing_file: artifact"cfsite_gcm_forcing"/HadGEM2-A_amip.2004-2008.07.nc
```

To learn more about artifacts and how they're used in CliMA, visit [ClimaArtifacts.jl](https://github.com/CliMA/ClimaArtifacts).

To add a new configuration argument/key, open [`config/default_configs/default_config.yml`](https://github.com/CliMA/ClimaAtmos.jl/blob/main/config/default_configs/default_config.yml).
Add an entry with the following format:

```yaml
<argument_name>:
  help: <help string>
  value: <argument_value>
```

The `help` field is optional if you don't plan on making a permanent change to the configuration argument.

The full list of configuration arguments is in
[Configuration options](configuration_options.md).

## Overriding parameters

Physical constants and calibratable parameters are managed by
[ClimaParams.jl](https://github.com/CliMA/ClimaParams.jl), which stores the
default values and lets you override them without touching source code. To
override a parameter, create a TOML file with one block per parameter:

```toml
[gravitational_acceleration]
value = 9.81
type = "float"
```

The `type` field (`bool`, `float`, `integer`, `string`, or `datetime`) is
optional; the
[ClimaParams TOML documentation](https://clima.github.io/ClimaParams.jl/dev/toml/)
describes the full format. Then list the file under the `toml` key of your
configuration:

```yaml
toml: [parameters.toml]
```

and run as usual. Three behaviors to know about:

  - The `toml` key accepts several files, but a given parameter may appear in
    only one of them; a duplicate entry across files raises a
    `Duplicate TOML entry` error.
  - A `toml:` key in a later configuration file *replaces* the earlier list
    rather than appending to it. Many shipped model configurations already
    carry one (e.g.
    [`toml/prognostic_edmfx_1M.toml`](https://github.com/CliMA/ClimaAtmos.jl/blob/main/toml/prognostic_edmfx_1M.toml)
    in the BOMEX example above), so to combine your overrides with such a
    configuration, list both files.
  - Overriding a parameter that no component of the current model uses raises
    an error; set `strict_params: false` to allow unused overrides.

## Environment variables

A few behaviors are controlled by environment variables rather than the
configuration file:

  - **`CI`**: when set (as it is on our continuous-integration tests), the
    default output directory is `<job_id>` instead of `output/<job_id>`. Set by
    the CI system; you normally do not need to set it yourself. (See
    `setup_output_dir` in `src/simulation/restart.jl`.)

  - **`CLIMAATMOS_GC_NSTEPS`**: the garbage-collection interval for distributed
    runs, described under
    [Running on GPUs and MPI](gpu_and_mpi.md).

## Common Configurations

ClimaAtmos provides a set of common numerical configurations that can be used as building blocks for different types of simulations. These configurations are located in `config/common_configs/` and contain standardized settings for grid resolution, time stepping, numerical schemes, and diagnostics.

### Available Common Configurations

#### Column Configurations

  - **[`numerics_column_ze63.yml`](https://github.com/CliMA/ClimaAtmos.jl/blob/main/config/common_configs/numerics_column_ze63.yml)**: Single column configuration with 63 vertical levels

#### Sphere Configurations

  - **[`numerics_sphere_he6ze10.yml`](https://github.com/CliMA/ClimaAtmos.jl/blob/main/config/common_configs/numerics_sphere_he6ze10.yml)**: Spherical configuration with 6 horizontal elements (550km), 10 vertical levels, 30km domain top, no sponge, explicit vertical diffusion

  - **[`numerics_sphere_he6ze31.yml`](https://github.com/CliMA/ClimaAtmos.jl/blob/main/config/common_configs/numerics_sphere_he6ze31.yml)**: Spherical configuration with 6 horizontal elements (550km), 31 vertical levels, 60km domain top, rayleigh and viscous sponges, implicit vertical diffusion

  - **[`numerics_sphere_he16ze63.yml`](https://github.com/CliMA/ClimaAtmos.jl/blob/main/config/common_configs/numerics_sphere_he16ze63.yml)**: Spherical configuration with 16 horizontal elements (206km), 63 vertical levels, 60km domain top, rayleigh and viscous sponges, implicit vertical diffusion

  - **[`numerics_sphere_he30ze43.yml`](https://github.com/CliMA/ClimaAtmos.jl/blob/main/config/common_configs/numerics_sphere_he30ze43.yml)**: Spherical configuration with 30 horizontal elements (110km), 43 vertical levels, 30km domain top, no sponge, explicit vertical diffusion

  - **[`numerics_sphere_he30ze63.yml`](https://github.com/CliMA/ClimaAtmos.jl/blob/main/config/common_configs/numerics_sphere_he30ze63.yml)**: Spherical configuration with 30 horizontal elements (110km), 63 vertical levels, 60km domain top, rayleigh and viscous sponges, implicit vertical diffusion

#### Diagnostics Configurations for PROPHET Columns

Common diagnostics sets for PROPHET single-column runs. Each file defines a `diagnostics:` block mostly at 10-minute output frequency; individual model configs can add case-specific diagnostics on top.

  - **[`diagnostics_column_progedmf_0M.yml`](https://github.com/CliMA/ClimaAtmos.jl/blob/main/config/common_configs/diagnostics_column_progedmf_0M.yml)**: Standard diagnostics for PROPHET columns with 0-moment microphysics (`microphysics_model: "0M"`). Includes atmospheric state, surface fluxes and precipitation, updraft/environment profiles, and entrainment/detrainment variables.

  - **[`diagnostics_column_progedmf_1M.yml`](https://github.com/CliMA/ClimaAtmos.jl/blob/main/config/common_configs/diagnostics_column_progedmf_1M.yml)**: Standard diagnostics for PROPHET columns with 1-moment microphysics (`microphysics_model: "1M"`). Mirrors the 0M set (without the static-energy variables `ha`/`haup`/`haen`) and adds rain/snow specific humidities, supersaturations, updraft/environment precipitation variables, and the full suite of 1M bulk microphysics process rates for the grid mean, updraft, and environment (`mp1m_*`, `mp1mup_*`, `mp1men_*`).

### Using Common Configurations

Common configurations are designed to be combined with model-specific configurations. In the CI pipeline and when running simulations, you can specify multiple configuration files:

```julia
import ClimaAtmos as CA

config = CA.AtmosConfig([
    "config/common_configs/numerics_sphere_he16ze63.yml",
    "config/model_configs/your_model_config.yml",
])
simulation = CA.AtmosSimulation(config)
CA.solve_atmos!(simulation)
```

For PROPHET single-column runs, a diagnostics common config is prepended before the model config:

```julia
import ClimaAtmos as CA

config = CA.AtmosConfig(
    [
        "config/common_configs/diagnostics_column_progedmf_1M.yml",
        "config/model_configs/prognostic_edmfx_bomex_column.yml",
    ]; job_id = "prognostic_edmfx_bomex_column")
simulation = CA.AtmosSimulation(config)
CA.solve_atmos!(simulation)
```

The common configuration provides the numerical setup (grid, time stepping, etc.), or common diagnostics, while the model configuration provides the physical setup (physics schemes, initial conditions, etc.). The model configuration overrides any conflicting settings from the common configuration. Please modify them only if you are certain of the implications.
