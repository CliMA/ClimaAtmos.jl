
# Creating custom configurations
To create a custom configuration, first make a .yml file.
In the file, you can set configuration arguments as `key: value` pairs to override the default config.
YAML parsing is fairly forgiving -- values will generally be parsed to the correct type.
The only exception is true/false strings. These need quotes around them, or they will be parsed to `Bool`s.

To start the model with a custom configuration, run:

`julia --project=examples examples/hybrid/driver.jl --config_file <yaml>`

### Example
Below is the default Bomex configuration:
```
job_id: "prognostic_edmfx_bomex_box"
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
hyperdiff: true
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


# Default Configuration

