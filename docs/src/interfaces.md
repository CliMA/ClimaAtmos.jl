# Script vs Config Interface

ClimaAtmos provides two ways to set up and run simulations. Both produce an
`AtmosSimulation` that is run with `solve_atmos!`; they differ in where the
configuration is written.

The **script API** builds a model on a grid and then wraps it in a simulation:
`AtmosSimulation(AtmosModel(grid; params, setup, ...); dt, t_end, ...)`. The
model owns the grid, the parameters, and the setup; the simulation owns run
control. It is best for interactive exploration, notebooks, and programmatic
parameter sweeps.
[Your First Simulation](first_simulation.md) walks through it, and
[Scripting Simulations](scripting_simulations.md) covers each component.

The **config API** reads the simulation from a YAML file:
`AtmosSimulation(AtmosConfig("config.yml"))`. Every key overrides a default
from [`config/default_configs/default_config.yml`](https://github.com/CliMA/ClimaAtmos.jl/blob/main/config/default_configs/default_config.yml), which makes runs
reproducible and shareable; it is what the CI pipelines use. See
[Creating custom configurations](configuration.md) for writing configuration
files and [Configuration options](configuration_options.md) for the complete
key list.

## How the options map

Everything physical is a keyword argument of `AtmosModel(grid; ...)`, with the
grid as its positional argument; run control is a keyword argument of
[`AtmosSimulation`](@ref ClimaAtmos.AtmosSimulation) (`dt`, `t_end`, `job_id`,
`checkpoint_frequency`, `diagnostics`). The `setup` and `diagnostics` options
take objects built by their own constructors (`Setups.*` and
`DiagnosticsConfig`) before being handed to `AtmosModel` or `AtmosSimulation`.

|             | Script API                              | YAML key                                 |
|:----------- |:--------------------------------------- |:---------------------------------------- |
| Entry point | `AtmosSimulation(model; kwargs...)`     | `AtmosSimulation(AtmosConfig("f.yml"))`  |
| Grid        | `AtmosModel(ColumnGrid(...); ...)`      | `config: "column"` + `z_max`, `z_elem`   |
|             | `AtmosModel(SphereGrid(...); ...)`      | `config: "sphere"` + `h_elem`, `z_elem`  |
|             | `AtmosModel(BoxGrid(...); ...)`         | `config: "box"` + `x_max`, `y_max`, etc. |
| Model       | `AtmosModel(grid; params, setup, ...)`  | physics keys (`turbconv`, `rad`, ...)    |
| Setup       | `setup = Setups.Bomex()`                | `initial_condition: "Bomex"`             |
| Timestep    | `dt = 5` or `dt = "5secs"`              | `dt: "5secs"`                            |
| Duration    | `t_end = 21600` or `"6hours"`           | `t_end: "6hours"`                        |
| Diagnostics | `DiagnosticsConfig(; default = ...)`    | `output_default_diagnostics:`            |
|             | `DiagnosticsConfig(; additional = ...)` | `diagnostics:` block                     |
| Checkpoints | `checkpoint_frequency = 3600`           | `dt_save_state_to_disk: "1hours"`        |

`job_id` is not in [`default_config.yml`](https://github.com/CliMA/ClimaAtmos.jl/blob/main/config/default_configs/default_config.yml). In a script it is an
`AtmosSimulation` keyword argument; with a configuration file, pass it to
`AtmosConfig` (`AtmosConfig("f.yml"; job_id = "my_run")`) or set a `job_id:`
key in the file itself. Given neither, it is derived from the configuration
file names.

Three caveats. First, the mapping covers each option in isolation: the shipped
YAML case configurations also set model physics keys (`turbconv`,
`microphysics_model`, `rad`, ...), the grid and timestep, and sometimes
parameter overrides (`toml`), so `setup = Setups.Bomex()` alone is not
equivalent to running [`prognostic_edmfx_bomex_column.yml`](https://github.com/CliMA/ClimaAtmos.jl/blob/main/config/model_configs/prognostic_edmfx_bomex_column.yml); in a script, the
corresponding physics is chosen through `AtmosModel` (or a `CA.Presets`
constructor). Second, the two entry points do not share every default: script
defaults are the keyword defaults of `AtmosModel` and
[`AtmosSimulation`](@ref ClimaAtmos.AtmosSimulation), YAML defaults come from
`default_config.yml`, and the two can differ (for example,
`update_cache_every` defaults to `"stage"` in the script API and `"step"` in
YAML). Third, `CA.AtmosSimulation(config)` and
`CA.get_simulation(config)` are the same operation, the former being an alias
for the latter; the documentation uses `AtmosSimulation(config)` throughout.
