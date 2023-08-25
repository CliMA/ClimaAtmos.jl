
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
edmf_coriolis: Bomex
dt_save_to_disk: 5mins
hyperdiff: "false"
z_elem: 60
dt: 20secs
debugging_tc: true
surface_setup: Bomex
turbconv_case: Bomex
t_end: 6hours
turbconv: edmf
z_stretch: false
config: column
subsidence: Bomex
FLOAT_TYPE: Float64
z_max: 3000.0
apply_limiter: false
regression_test: true
ls_adv: Bomex
dt_save_to_sol: 5mins
job_id: edmf_bomex
moist: equil
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
