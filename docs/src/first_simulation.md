# Your First Simulation

## Minimal example

The simplest ClimaAtmos simulation uses all defaults. It solves the dry
compressible equations (with hyperdiffusion and surface fluxes) on a global
cubed-sphere grid, starting from a hydrostatically balanced, slightly perturbed
state with a vertically decaying temperature profile:

```@example first_sim
using Logging # hide
Logging.disable_logging(Logging.Info) # hide
import ClimaAtmos as CA

simulation = CA.AtmosSimulation{Float32}(; t_end = "1days")
nothing # hide
```

This builds the simulation but does not run it. [Running the simulation](@ref)
advances it in time. The first construction and solve in a session compile a
large amount of code and can take several minutes; later calls are fast.

`AtmosSimulation{FT}(...)` accepts keyword arguments for every aspect of
the simulation. When omitted, defaults are used (see
[Script vs Config Interface](@ref) for the full list).

## Customizing the simulation

### Change the grid

Run a single-column model instead of the default global cubed-sphere:

```@example first_sim
grid = CA.ColumnGrid(Float32; z_elem = 30, z_max = 30000.0)
simulation = CA.AtmosSimulation{Float32}(; grid, t_end = "6hours")
nothing # hide
```

See the [Grids](api.md#Grids) section of the API for all grid types and their options.

### Change the timestep and duration

`dt` is the timestep and `t_end` the total simulation time. Each accepts either a number
of seconds, or a duration string with a unit (`secs`, `mins`, `hours`, `days`, `weeks`).
This is the same syntax used by the [config interface](@ref "Script vs Config Interface"):

```@example first_sim
simulation = CA.AtmosSimulation{Float32}(;
    dt = "5mins",     # equivalently, dt = 300
    t_end = "10days", # equivalently, t_end = 86400 * 10
)
nothing # hide
```

### Change the setup

A *setup* defines the initial conditions, boundary conditions, and (optionally)
forcing for a simulation case. A setup supplies the initial state only; the
physics comes from the `model`, and the two must be chosen together. For
example, the BOMEX shallow-cumulus case needs a moist model (with the default
dry model it would have no moisture to convect):

```@example first_sim
simulation = CA.AtmosSimulation{Float32}(;
    grid = CA.ColumnGrid(Float32; z_elem = 60, z_max = 3000.0, z_stretch = false),
    setup = CA.Setups.Bomex(),
    model = CA.Presets.equil_moist_0m(),
    dt = 5,
    t_end = 3600,
    job_id = "my_bomex",
)
nothing # hide
```

See the [Setups](@ref) page for the full list of available setups and how to create
your own.

## Presets

Common configurations are available as one-line presets in `CA.Presets`:

```@example first_sim
simulation = CA.Presets.bomex(Float32; t_end = "10mins")
nothing # hide
```

See the [Presets](api.md#Presets) section of the API for the full list of
simulation and model presets.

## Running the simulation

Constructing an `AtmosSimulation` sets everything up but does not advance it in
time. Call `solve_atmos!` to integrate the simulation forward to `t_end`:

```julia
CA.solve_atmos!(simulation)
```

## Inspecting results

After a simulation completes, access the prognostic state through the
integrator:

!!! note

    `Y` is the state vector: `Y.c` holds the cell-center variables (such as
    the density `ρ` and the total energy `ρe_tot`) and `Y.f` the cell-face
    variables (such as the vertical velocity `u₃`). The `integrator` is the
    ODE integrator, from ClimaTimeSteppers, that advances the state in time.
    See the [Glossary](@ref) for these and other recurring names.

```@example first_sim
Y = simulation.integrator.u

# Center (cell-center) variables
propertynames(Y.c)  # (:ρ, :uₕ, :ρe_tot, :ρq_tot) for the moist BOMEX preset
# above; the dry default has (:ρ, :uₕ, :ρe_tot)

# Face (cell-interface) variables
propertynames(Y.f)  # e.g., (:u₃,)
```

Output is written to `simulation.output_dir`, which defaults to
`output/<job_id>` under the directory Julia was started in (here
`output/my_bomex`). Each run writes to a numbered subdirectory, and
`output_active` links to the most recent one. Two formats appear there, each
with a distinct role:

  - **NetCDF** (`.nc`) files hold the **diagnostics** -- derived (and often interpolated)
    output variables such as temperature or precipitation. See
    [Computing and saving diagnostics](@ref) for how to configure them.
  - **HDF5** (`.hdf5`) files hold full-resolution **model-state checkpoints**, written when
    `checkpoint_frequency` is set. These are the files a simulation reads to
    [restart](@ref "Restarting Simulations in ClimaAtmos").

## Terminology

The state vector `Y`, the cache `p`, the simulation time `t`, and other recurring
symbols and terms are defined in the [Glossary](@ref).

## Using the config-based interface

The same kind of simulation can be set up with a YAML configuration file instead
of a script. The path below is relative to the repository root, so this
assumes a clone (see [Running from a cloned repository](@ref)) with Julia
started there. For example, to build and run the default BOMEX configuration:

```julia
import ClimaAtmos as CA

config = CA.AtmosConfig("config/model_configs/prognostic_edmfx_bomex_column.yml")
simulation = CA.AtmosSimulation(config)
CA.solve_atmos!(simulation)
```

Every key in the YAML file overrides a default from
`config/default_configs/default_config.yml`. See
[Script vs Config Interface](@ref) for how the two workflows relate, and
[Creating custom configurations](configuration.md) for writing your own configuration files.

## Next steps

  - [Script vs Config Interface](@ref) -- detailed comparison of the two workflows
  - [Single Column Models](@ref) -- BOMEX, DYCOMS, RICO, and more
  - [Computing and saving diagnostics](@ref) -- configure output variables and formats
