# Script vs Config Interface

ClimaAtmos provides two ways to set up and run simulations. Both produce
an `AtmosSimulation` that is run with `solve_atmos!`.

## Script API

Build a model on a grid, then wrap it in a simulation:

```julia
import ClimaAtmos as CA

model = CA.AtmosModel(
    CA.SphereGrid(Float64; z_elem = 45, h_elem = 6);
    setup = CA.Setups.DecayingProfile(; perturb = true),
)
simulation = CA.AtmosSimulation(model;
    dt = "10mins",
    t_end = "10days",
    job_id = "my_run",
)
CA.solve_atmos!(simulation)
```

The model owns the grid, the physical parameters, and the setup; the
simulation owns run control (timestep, duration, callbacks, output).

**Best for:** interactive exploration, notebooks, custom scripts, programmatic
parameter sweeps.

See [`AtmosSimulation`](@ref ClimaAtmos.AtmosSimulation) for more information on how to customize your simulation.

## Config API

Define the simulation in a YAML file, then load it:

```yaml
# config.yml
initial_condition: "DecayingProfile"
perturb_initstate: true
config: "sphere"
z_elem: 45
h_elem: 6
dt: "600secs"
t_end: "10days"
```

```julia
import ClimaAtmos as CA

config = CA.AtmosConfig("config.yml"; job_id = "my_run")
simulation = CA.get_simulation(config)
CA.solve_atmos!(simulation)
```

**Best for:** reproducible runs, CI pipelines, and sharing configurations.

See the [Creating custom configurations](config.md) reference for the complete list of
YAML options.

## Comparison

|             | Script API                             | Config API                     |
|:----------- |:-------------------------------------- |:------------------------------ |
| Entry point | `AtmosSimulation(model; kwargs...)`    | `AtmosConfig("file.yml")`      |
| Model       | `AtmosModel(grid; params, setup, ...)` | Built from YAML keys           |
| Grid        | Positional argument of `AtmosModel`    | `config` key + grid parameters |
| Setup       | `setup` kwarg of `AtmosModel`          | `initial_condition` string     |
| Timestep    | `dt = 600` (number)                    | `dt: "600secs"` (string)       |
| Duration    | `t_end = 864000` (number)              | `t_end: "10days"` (string)     |

## Common mappings

Everything physical is a keyword argument of `AtmosModel(grid; ...)`;
run control is a keyword argument of
[`AtmosSimulation`](@ref ClimaAtmos.AtmosSimulation) (`dt`, `t_end`,
`job_id`, `checkpoint_frequency`, `diagnostics`, ...).

| Script kwarg                                 | YAML key                                 |
|:-------------------------------------------- |:---------------------------------------- |
| `AtmosModel(ColumnGrid(...); ...)`           | `config: "column"` + `z_max`, `z_elem`   |
| `AtmosModel(SphereGrid(...); ...)`           | `config: "sphere"` + `h_elem`, `z_elem`  |
| `AtmosModel(BoxGrid(...); ...)`              | `config: "box"` + `x_max`, `y_max`, etc. |
| `AtmosModel(grid; setup = Setups.Bomex())`   | `initial_condition: "Bomex"`             |
| `AtmosModel(grid; aerosol_names = ("SO4",))` | `prescribed_aerosols: ["SO4"]`           |
| `dt = 5`                                     | `dt: "5secs"`                            |
| `t_end = 21600`                              | `t_end: "6hours"`                        |
| `diagnostics = DiagnosticsConfig(...)`       | `output_default_diagnostics: true`       |
| `checkpoint_frequency = 3600`                | `dt_save_state_to_disk: "1hours"`        |

`job_id` is not a YAML key: in the config workflow set it with the `--job_id` flag
(or it defaults to the config file name); in the script workflow it is an
`AtmosSimulation` keyword argument.
