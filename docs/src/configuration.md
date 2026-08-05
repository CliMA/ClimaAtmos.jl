# Creating custom configurations

To create a custom configuration, first make a .yml file.
In the file, you can set configuration arguments as `key: value` pairs to override the default config.
YAML parsing is forgiving -- values generally parse to the correct type.
One caveat: unquoted `true`/`false` are parsed to `Bool`s; if a configuration argument
expects the literal string `"true"` or `"false"`, put quotes around it.

To start the model with a custom configuration, run:

```julia
import ClimaAtmos as CA

config = CA.AtmosConfig("path/to/config.yaml")
simulation = CA.AtmosSimulation(config)
CA.solve_atmos!(simulation)
```

## Example

Below is the default BOMEX configuration
(`config/model_configs/prognostic_edmfx_bomex_column.yml`):

```
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

```
initial_condition: "GCM"
external_forcing_file: artifact"cfsite_gcm_forcing"/HadGEM2-A_amip.2004-2008.07.nc
```

To learn more about artifacts and how they're used in CliMA, visit [ClimaArtifacts.jl](https://github.com/CliMA/ClimaArtifacts).

To add a new configuration argument/key, open `config/default_configs/default_config.yml`.
Add an entry with the following format:

```
<argument_name>:
    value: <argument_value>
    help: <help string>
```

The `help` field is optional if you don't plan on making a permanent change to the configuration argument.

The full list of configuration arguments is in
[Configuration options](configuration_options.md).

## Environment variables

A few behaviors are controlled by environment variables rather than the
configuration file:

  - **`CI`**: when set (as it is on our continuous-integration tests), the
    default output directory is `<job_id>` instead of `output/<job_id>`. Set by
    the CI system; you normally do not need to set it yourself. (See
    `setup_output_dir` in `src/simulation/restart.jl`.)

  - **`CLIMAATMOS_GC_NSTEPS`**: number of steps between manual garbage-collection
    calls for distributed (MPI) runs. Defaults to `1000`. Only has an effect when
    running with more than one process. (See `gc_callback` in
    `src/callbacks/get_callbacks.jl`.)

## Common Configurations

ClimaAtmos provides a set of common numerical configurations that can be used as building blocks for different types of simulations. These configurations are located in `config/common_configs/` and contain standardized settings for grid resolution, time stepping, numerical schemes, and diagnostics.

### Available Common Configurations

#### Column Configurations

  - **`numerics_column_ze63.yml`**: Single column configuration with 63 vertical levels

#### Sphere Configurations

  - **`numerics_sphere_he6ze10.yml`**: Spherical configuration with 6 horizontal elements (550km), 10 vertical levels, 30km domain top, no sponge, explicit vertical diffusion

  - **`numerics_sphere_he6ze31.yml`**: Spherical configuration with 6 horizontal elements (550km), 31 vertical levels, 60km domain top, rayleigh and viscous sponges, implicit vertical diffusion

  - **`numerics_sphere_he16ze63.yml`**: Spherical configuration with 16 horizontal elements (206km), 63 vertical levels, 60km domain top, rayleigh and viscous sponges, implicit vertical diffusion

  - **`numerics_sphere_he30ze43.yml`**: Spherical configuration with 30 horizontal elements (110km), 43 vertical levels, 30km domain top, no sponge, explicit vertical diffusion

  - **`numerics_sphere_he30ze63.yml`**: Spherical configuration with 30 horizontal elements (110km), 63 vertical levels, 60km domain top, rayleigh and viscous sponges, implicit vertical diffusion

#### Diagnostics Configurations for PROPHET Columns

Common diagnostics sets for PROPHET (prognostic EDMF) single-column runs. Each file defines a `diagnostics:` block mostly at 10-minute output frequency; individual model configs can add case-specific diagnostics on top.

  - **`diagnostics_column_progedmf_0M.yml`**: Standard diagnostics for PROPHET columns with 0-moment microphysics (`microphysics_model: "0M"`). Includes atmospheric state, surface fluxes and precipitation, updraft/environment profiles, and entrainment/detrainment variables.

  - **`diagnostics_column_progedmf_1M.yml`**: Standard diagnostics for PROPHET columns with 1-moment microphysics (`microphysics_model: "1M"`). Extends the 0M set with rain/snow specific humidities, updraft/environment precipitation variables, and the full suite of 1M bulk microphysics process rates for the grid mean, updraft, and environment (`mp1m_*`, `mp1mup_*`, `mp1men_*`).

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
