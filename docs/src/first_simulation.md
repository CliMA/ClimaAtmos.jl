# Your First Simulation

## Minimal example

The simplest ClimaAtmos simulation uses all defaults -- it solves the dry
compressible Euler equations on a global cubed-sphere grid, starting from a
hydrostatically balanced state with a vertically decaying temperature profile:

```@example first_sim
using Logging # hide
Logging.disable_logging(Logging.Info) # hide
import ClimaAtmos as CA

simulation = CA.AtmosSimulation{Float32}(; t_end = "1days")
nothing # hide
```

This builds the simulation but does not run it. [Running the simulation](@ref)
advances it in time.

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
of seconds, or a duration string with a unit (`secs`, `mins`, `hours`, `days`, `weeks`) --
the same syntax used by the [config interface](@ref "Script vs Config Interface"):

```@example first_sim
simulation = CA.AtmosSimulation{Float32}(;
    dt = "5mins",     # equivalently, dt = 300
    t_end = "10days", # equivalently, t_end = 86400 * 10
)
nothing # hide
```

### Change the setup

A *setup* defines the initial conditions, boundary conditions, and (optionally)
forcing for a simulation case. For example, the BOMEX shallow cumulus case:

```@example first_sim
simulation = CA.AtmosSimulation{Float32}(;
    grid = CA.ColumnGrid(Float32; z_elem = 60, z_max = 3000.0, z_stretch = false),
    setup = CA.Setups.Bomex(),
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

This advances the model to `t_end`.


## Inspecting results

After a simulation completes, access the prognostic state through the
integrator:

```@example first_sim
Y = simulation.integrator.u

# Center (cell-center) variables
propertynames(Y.c)  # e.g., (:ρ, :uₕ, :ρe_tot, :ρq_tot)

# Face (cell-interface) variables
propertynames(Y.f)  # e.g., (:u₃,)
```

Output is written to `simulation.output_dir` in two formats, each with a distinct role:

- **NetCDF** (`.nc`) files hold the **diagnostics** -- derived (and often interpolated)
  output variables such as temperature or precipitation. See
  [Computing and saving diagnostics](@ref) for how to configure them.
- **HDF5** (`.h5`) files hold full-resolution **model-state checkpoints**, written when
  `checkpoint_frequency` is set. These are the files a simulation reads to
  [restart](@ref "Restarting Simulations in ClimaAtmos").

## Terminology

The state vector `Y`, the cache `p`, the simulation time `t`, and other recurring
symbols and terms are defined in the [Glossary](@ref).

## Using the config-based interface

The same simulation can be set up with a YAML file.

## Next steps

- [Script vs Config Interface](@ref) -- detailed comparison of the two workflows
- [Single Column Models](@ref) -- BOMEX, DYCOMS, RICO, and more
- [Computing and saving diagnostics](@ref) -- configure output variables and formats
