# Script vs Config Interface

ClimaAtmos provides two ways to set up and run simulations. Both produce an
`AtmosSimulation` that is run with `solve_atmos!`; they differ in where the
configuration lives.

The **script API** builds a simulation from Julia keyword arguments:
`AtmosSimulation{FT}(; grid, model, setup, dt, t_end, ...)`. It is best for
interactive exploration, notebooks, and programmatic parameter sweeps.
[Your First Simulation](first_simulation.md) walks through it, and
[Scripting Simulations](scripting_simulations.md) covers each component.

The **config API** reads the simulation from a YAML file:
`AtmosSimulation(AtmosConfig("config.yml"))`. Every key overrides a default
from `config/default_configs/default_config.yml`, which makes runs
reproducible and shareable; it is what the CI pipelines use. See
[Creating custom configurations](configuration.md) for writing configuration
files and [Configuration options](configuration_options.md) for the complete
key list.

## How the options map

All script options are keyword arguments of
[`AtmosSimulation`](@ref ClimaAtmos.AtmosSimulation). Most are passed directly
(`dt`, `t_end`, `job_id`, `checkpoint_frequency`, `diagnostics`); `grid`,
`model`, and `setup` instead take objects built by their own constructors (the
grid constructors, `AtmosModel`, and `Setups.*`) before being handed to
`AtmosSimulation`.

|             | Script API                              | YAML key                                 |
|:----------- |:--------------------------------------- |:---------------------------------------- |
| Entry point | `AtmosSimulation{FT}(; kwargs...)`      | `AtmosSimulation(AtmosConfig("f.yml"))`  |
| Grid        | `grid = ColumnGrid(...)`                | `config: "column"` + `z_max`, `z_elem`   |
|             | `grid = SphereGrid(...)`                | `config: "sphere"` + `h_elem`, `z_elem`  |
|             | `grid = BoxGrid(...)`                   | `config: "box"` + `x_max`, `y_max`, etc. |
| Model       | `model = AtmosModel(...)`               | physics keys (`turbconv`, `rad`, ...)    |
| Setup       | `setup = Setups.Bomex()`                | `initial_condition: "Bomex"`             |
| Timestep    | `dt = 5` or `dt = "5secs"`              | `dt: "5secs"`                            |
| Duration    | `t_end = 21600` or `"6hours"`           | `t_end: "6hours"`                        |
| Diagnostics | `DiagnosticsConfig(; default = ...)`    | `output_default_diagnostics:`            |
|             | `DiagnosticsConfig(; additional = ...)` | `diagnostics:` block                     |
| Checkpoints | `checkpoint_frequency = 3600`           | `dt_save_state_to_disk: "1hours"`        |

`job_id` is not in `default_config.yml`. In a script it is an
`AtmosSimulation` keyword argument; with a configuration file, pass it to
`AtmosConfig` (`AtmosConfig("f.yml"; job_id = "my_run")`) or set a `job_id:`
key in the file itself. Given neither, it is derived from the configuration
file names.

Two caveats. First, the mapping covers each option in isolation: the shipped
YAML case configurations also set model physics keys (`turbconv`,
`microphysics_model`, `rad`, ...), so `setup = Setups.Bomex()` alone is not
equivalent to running `prognostic_edmfx_bomex_column.yml`; in a script, the
corresponding physics is chosen through `AtmosModel` (or a `CA.Presets`
constructor). Second, `CA.AtmosSimulation(config)` and
`CA.get_simulation(config)` are the same operation, the former being an alias
for the latter; the documentation uses `AtmosSimulation(config)` throughout.
