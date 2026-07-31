# Script vs Config Interface

ClimaAtmos provides two ways to set up and run simulations. Both produce
an `AtmosSimulation` that is run with `solve_atmos!`.

## Script API

Build a simulation directly from Julia keyword arguments:

```julia
import ClimaAtmos as CA

simulation = CA.AtmosSimulation{Float64}(;
    model = CA.AtmosModel(),
    grid = CA.SphereGrid(Float64; z_elem = 45, h_elem = 6),
    setup = CA.Setups.DecayingProfile(;
        perturb = true,
        params = CA.ClimaAtmosParameters(Float64),
    ),
    dt = "10mins",
    t_end = "10days",
    job_id = "my_run",
)
CA.solve_atmos!(simulation)
```

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
simulation = CA.AtmosSimulation(config)
CA.solve_atmos!(simulation)
```

**Best for:** reproducible runs, CI pipelines, and sharing configurations.

See [Configuration options](configuration_options.md) for the complete list of
YAML options.

## Comparison

|             | Script API                         | Config API                     |
|:----------- |:---------------------------------- |:------------------------------ |
| Entry point | `AtmosSimulation{FT}(; kwargs...)` | `AtmosConfig("file.yml")`      |
| Model       | Pass `AtmosModel()` directly       | Built from YAML keys           |
| Grid        | Pass grid object                   | `config` key + grid parameters |
| Setup       | Pass setup instance                | `initial_condition` string     |
| Timestep    | `dt = 600` or `dt = "10mins"`      | `dt: "600secs"`                |
| Duration    | `t_end = 864000` or `"10days"`     | `t_end: "10days"`              |

## Common mappings

All script options are keyword arguments of
[`AtmosSimulation`](@ref ClimaAtmos.AtmosSimulation). Most are passed directly (`dt`,
`t_end`, `job_id`, `checkpoint_frequency`, `diagnostics`); `grid`, `model`, and `setup`
instead take objects built by their own constructors (the grid constructors, `AtmosModel`,
and `Setups.*`) before being handed to `AtmosSimulation`.

| Script kwarg                           | YAML key                                 |
|:-------------------------------------- |:---------------------------------------- |
| `grid = ColumnGrid(...)`               | `config: "column"` + `z_max`, `z_elem`   |
| `grid = SphereGrid(...)`               | `config: "sphere"` + `h_elem`, `z_elem`  |
| `grid = BoxGrid(...)`                  | `config: "box"` + `x_max`, `y_max`, etc. |
| `setup = Setups.Bomex()`               | `initial_condition: "Bomex"`             |
| `dt = 5`                               | `dt: "5secs"`                            |
| `t_end = 21600`                        | `t_end: "6hours"`                        |
| `diagnostics = DiagnosticsConfig(...)` | `output_default_diagnostics: true`       |
| `checkpoint_frequency = 3600`          | `dt_save_state_to_disk: "1hours"`        |

`job_id` is not in `default_config.yml`, but `AtmosConfig` reads a `job_id:` key
if one is present in a configuration file; otherwise set it with the `--job_id`
flag (or it defaults to the config file name). In the script workflow it is an
`AtmosSimulation` keyword argument.

Two caveats. First, the mapping covers each option in isolation: the shipped
YAML case configurations also set model physics keys (`turbconv`,
`microphysics_model`, `rad`, ...), so `setup = Setups.Bomex()` alone is not
equivalent to running `prognostic_edmfx_bomex_column.yml`; in a script, the
corresponding physics is chosen through `AtmosModel` (or a `CA.Presets`
constructor). Second, `CA.AtmosSimulation(config)` and the lower-level
`CA.get_simulation(config)` are the same operation; the documentation uses
`AtmosSimulation(config)` throughout.
