
# Creating custom configurations
To create a custom configuration, first make a .yml file.
In the file, you can set configuration arguments as `key: value` pairs to override the default config.
YAML parsing is fairly forgiving -- values will generally be parsed to the correct type.
The only exception is true/false strings. These need quotes around them, or they will be parsed to `Bool`s.

To start the model with a custom configuration, run:

`julia --project=.buildkite .buildkite/ci_driver.jl --config_file <yaml>`

### Example
Below is the default Bomex configuration:
```
initial_condition: "Bomex"
subsidence: "Bomex"
scm_coriolis: "Bomex"
ls_adv: "Bomex"
surface_setup: "Bomex"
turbconv: "prognostic_edmfx"
edmfx_upwinding: first_order
edmfx_entr_model: "Generalized"
edmfx_detr_model: "Generalized"
edmfx_sgs_mass_flux: true
edmfx_sgs_diffusive_flux: true
edmfx_nh_pressure: true
prognostic_tke: false
moist: "equil"
config: "box"
hyperdiff: Hyperdiffusion
x_max: 1e8
y_max: 1e8
z_max: 3e3
x_elem: 2
y_elem: 2
z_elem: 60
z_stretch: false
perturb_initstate: false
dt: "5secs"
t_end: "6hours"
dt_save_state_to_disk: "10mins"
toml: [toml/prognostic_edmfx.toml]
```

Keys can also point to artifacts. As artifacts are folders, we specify both the artifact name, as we would from the REPL, and file to read from, separated by a `/`. For example, to drive a single
column model with an external forcing file from GCM output, we include the following lines in the
configuration:
```
insolation: "gcmdriven"
external_forcing_file: artifact"cfsite_gcm_forcing"/HadGEM2-A_amip.2004-2008.07.nc
```
To learn more about artifacts and how they're used in CliMA, visit [ClimaArtifacts.jl](https://github.com/CliMA/ClimaArtifacts).

To add a new configuration argument/key, open `.buildkite/default_config.yml`.
Add an entry with the following format:
```
<argument_name>:
    value: <argument_value>
    help: <help string>
```
The `help` field is optional if you don't plan on making a permanent change to the configuration argument.

See below for the full list of configuration arguments.

# Common Configurations

ClimaAtmos provides a set of common numerical configurations that can be used as building blocks for different types of simulations. These configurations are located in `config/common_configs/` and contain standardized settings for grid resolution, time stepping, and numerical schemes.

## Available Common Configurations

### Column Configurations
- **`numerics_column_ze63.yml`**: Single column configuration with 63 vertical levels

### Sphere Configurations
- **`numerics_sphere_he6ze10.yml`**: Spherical configuration with 6 horizontal elements (550km), 10 vertical levels, 30km domain top, no sponge, explicit vertical diffusion

- **`numerics_sphere_he6ze31.yml`**: Spherical configuration with 6 horizontal elements (550km) , 31 vertical levels, 60km domain top, rayleigh and viscous sponges, implicit vertical diffusion

- **`numerics_sphere_he16ze63.yml`**: Spherical configuration with 16 horizontal elements (206km), 63 vertical levels, 60km domain top, rayleigh and viscous sponges, implicit vertical diffusion

- **`numerics_sphere_he30ze43.yml`**: Spherical configuration with 30 horizontal elements (110km), 43 vertical levels, 30km domain top, no sponge, explicit vertical diffusion

- **`numerics_sphere_he30ze63.yml`**: Spherical configuration with 30 horizontal elements (110km), 63 vertical levels, 60km domain top, rayleigh and viscous sponges, implicit vertical diffusion

## Using Common Configurations

Common configurations are designed to be used in combination with model-specific configurations. In the CI pipeline and when running simulations, you can specify multiple configuration files:

```bash
julia --project=.buildkite .buildkite/ci_driver.jl \
  --config_file config/common_configs/numerics_sphere_he16ze63.yml \
  --config_file config/model_configs/your_model_config.yml
```

The common configuration provides the numerical setup (grid, time stepping, etc.), while the model configuration provides the physical setup (physics schemes, initial conditions, etc.). The model configuration will override any conflicting settings from the common configuration. Please modify them only if you are certain of the implications.

# Default Configuration
