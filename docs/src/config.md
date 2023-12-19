
# Creating custom configurations
To create a custom configuration, first make a .yml file.
In the file, you can set configuration arguments as `key: value` pairs to override the default config.
YAML parsing is fairly forgiving -- values will generally be parsed to the correct type.
The only exception is true/false strings. These need quotes around them, or they will be parsed to `Bool`s.

To start the model with a custom configuration, run: 

`julia --project=examples examples/driver.jl --config_file <yaml>`

### Example
Below is the default Bomex configuration:
```
job_id: "prognostic_edmfx_bomex_box"
initial_condition: "Bomex"
subsidence: "Bomex"
edmf_coriolis: "Bomex"
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
hyperdiff: "true"
kappa_4_vorticity: 1.0e12
kappa_4_tracer: 1.0e12
x_max: 1e5
y_max: 1e5
z_max: 3e3
x_elem: 2
y_elem: 2
z_elem: 60
z_stretch: false
perturb_initstate: false
dt: "5secs"
t_end: "6hours"
dt_save_to_disk: "10mins"
toml: [toml/prognostic_edmfx_box.toml]
```

To add a new configuration argument/key, open `.buildkite/default_config.yml`.
Add an entry with the following format:
```
<argument_name>:
    value: <argument_value>
    help: <help string>
```
The `help` field is optional if you don't plan on making a permanent change to the configuration argument.
If adding a configuration used in EDMF, add it to `.buildkite/default_edmf_config.yml`.

See below for the full list of configuration arguments.


# Default Configuration
```@example
include("config_table.jl");
```
