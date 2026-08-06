# Scripting Simulations

YAML configurations (see
[Creating custom configurations](configuration.md)) suit reproducible,
batch-style runs. The scripting interface builds the same simulations from
Julia code, which fits parameter sweeps, interactive exploration in the REPL
or in notebooks, and customization beyond what the YAML schema exposes. For a
side-by-side comparison of the two interfaces, see
[Script vs Config Interface](interfaces.md).

## The `AtmosSimulation` object

A simulation is an [`AtmosSimulation`](@ref ClimaAtmos.AtmosSimulation) value.
Its keyword arguments set the grid, the physics, the initial state, and the
run parameters:

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
    job_id = "my_script_run",
)
```

Scalar options such as the timestep `dt` and the run length `t_end` accept a
number in seconds or a string with a unit (`secs`, `mins`, `hours`, `days`,
`weeks`), the same syntax the YAML interface uses. The composite pieces are
objects constructed separately and passed in, as the following sections show.

The `grid`, `setup`, and `model` arguments are not independent: a setup
supplies the initial state, the model supplies the physics, and a case only
makes sense when they match. Running the BOMEX setup with the default dry
model, for instance, leaves no moisture to convect. The presets below pair
them correctly; when assembling the pieces yourself, choose them together.

## Grid

Construct a [`ColumnGrid`](@ref ClimaAtmos.ColumnGrid),
[`BoxGrid`](@ref ClimaAtmos.BoxGrid),
[`PlaneGrid`](@ref ClimaAtmos.PlaneGrid), or
[`SphereGrid`](@ref ClimaAtmos.SphereGrid) and pass it as `grid`:

```julia
# A single column with 60 vertical levels up to 40 km
grid = CA.ColumnGrid(Float64; z_elem = 60, z_max = 40000.0)

# The same, with uniform spacing instead of the default stretching
grid = CA.ColumnGrid(Float64; z_elem = 60, z_max = 3000.0, z_stretch = false)

# A global cubed-sphere grid
grid = CA.SphereGrid(Float64; z_elem = 45, h_elem = 6)
```

The [Grids](grids.md) reference page lists the constructors and their
options, including vertical stretching and topography.

## Model

An `AtmosModel` collects the physics and numerics choices: microphysics,
turbulence and convection, radiation, surface fluxes, and so on. Keyword
arguments override the defaults one at a time:

```julia
model = CA.AtmosModel(;
    microphysics_model = CA.EquilibriumMicrophysics0M(),
    radiation_mode = CA.RRTMGPI.AllSkyRadiation(),
)
```

Preset constructors in `CA.Presets` assemble common combinations, so most
scripts start from one of them instead of from bare `AtmosModel` keywords.
The model presets are `dry`, `equil_moist_0m`, `nonequil_moist_1m`,
`prognostic_edmf`, and `prognostic_edmf_1m`; the simulation presets
`aquaplanet`, `baroclinic_wave`, and `bomex` return a ready-to-run
`AtmosSimulation`. Each forwards keyword arguments for further overrides:

```julia
# PROPHET turbulence-convection with 0-moment microphysics, plus radiation
model = CA.Presets.prognostic_edmf(
    Float64;
    radiation_mode = CA.RRTMGPI.AllSkyRadiation(),
)
```

## Setup (initial conditions)

A setup defines the initial state and, for column cases, the case forcing:

```julia
# The BOMEX shallow-cumulus case
setup = CA.Setups.Bomex()

# An idealized decaying temperature profile for global runs
setup = CA.Setups.DecayingProfile(;
    perturb = true,
    params = CA.ClimaAtmosParameters(Float64),
)
```

The [Setups](setups.md) reference page lists the available cases and the
interface for defining new ones.

## Diagnostics

Output is specified by a `DiagnosticsConfig` passed as `diagnostics`; the
output location is the simulation's `output_dir` (derived from `job_id` when
not given):

```julia
diagnostics = CA.DiagnosticsConfig(;
    default = true,   # the built-in diagnostic set for this model
    additional = ["ua" => (; period = "30mins", reduction = "average")],
)

simulation = CA.AtmosSimulation{Float64}(;
    model, grid, setup, diagnostics,
    output_dir = "my_output_directory",
)
```

See [Computing and saving diagnostics](diagnostics.md) for the accepted entry
forms and the output layout.

## Running the integration

`solve_atmos!` runs the integration loop:

```julia
sol_res = CA.solve_atmos!(simulation)
```

After the run, the final prognostic state is `simulation.integrator.u`.

## Interactivity and debugging

The scripting interface works well for inspecting the model state
interactively. `solve_atmos!` advances the simulation in place, and the
integrator can also be stepped manually:

```julia
import ClimaTimeSteppers: step!

step!(simulation.integrator)    # advance a single timestep
Y = simulation.integrator.u     # prognostic state
p = simulation.integrator.p     # cache: precomputed fields, parameters
```

The same loop works when starting from a YAML configuration: build the
config, adjust its parsed arguments, and rebuild the simulation to pick up
the change:

```julia
config = CA.AtmosConfig("config/model_configs/prognostic_edmfx_bomex_column.yml")
config.parsed_args["t_end"] = "1hours"
simulation = CA.AtmosSimulation(config)
CA.solve_atmos!(simulation)
```

For development,
[Revise.jl](https://timholy.github.io/Revise.jl/stable/) makes source edits
take effect in the running session, and
[Infiltrator.jl](https://github.com/JuliaDebug/Infiltrator.jl) sets
breakpoints inside tendency functions.
