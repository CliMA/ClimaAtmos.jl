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
    initial_condition = CA.Setups.DecayingProfile(; perturb = true),
    dt = 600,
    t_end = 86400 * 10,
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
job_id: "my_run"
```

```julia
import ClimaAtmos as CA

config = CA.AtmosConfig("config.yml")
simulation = CA.get_simulation(config)
CA.solve_atmos!(simulation)
```

**Best for:** reproducible runs, CI pipelines, and sharing configurations.

See the [Creating custom configurations](config.md) reference for the complete list of
YAML options.

## Comparison

| | Script API | Config API |
|---|---|---|
| Entry point | `AtmosSimulation{FT}(; kwargs...)` | `AtmosConfig("file.yml")` |
| Model | Pass `AtmosModel()` directly | Built from YAML keys |
| Grid | Pass grid object | `config` key + grid parameters |
| Setup | Pass setup instance | `initial_condition` string |
| Timestep | `dt = 600` (number) | `dt: "600secs"` (string) |
| Duration | `t_end = 864000` (number) | `t_end: "10days"` (string) |

## Common mappings

| Script kwarg | YAML key |
|---|---|
| `grid = ColumnGrid(...)` | `config: "column"` + `z_max`, `z_elem` |
| `grid = SphereGrid(...)` | `config: "sphere"` + `h_elem`, `z_elem` |
| `grid = BoxGrid(...)` | `config: "box"` + `x_max`, `y_max`, etc. |
| `initial_condition = Setups.Bomex()` | `initial_condition: "Bomex"` |
| `dt = 5` | `dt: "5secs"` |
| `t_end = 21600` | `t_end: "6hours"` |
| `default_diagnostics = true` | `output_default_diagnostics: true` |
| `job_id = "my_run"` | `job_id: "my_run"` |
| `checkpoint_frequency = 3600` | `dt_save_state_to_disk: "1hours"` |
