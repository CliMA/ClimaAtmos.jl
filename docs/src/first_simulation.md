# Your First Simulation

## Minimal example

The simplest ClimaAtmos simulation uses all defaults — it solves the dry
compressible Euler equations on a global cubed-sphere grid, starting from a
hydrostatically balanced state with a vertically decaying temperature profile:

```@example first_sim
using Logging # hide
Logging.disable_logging(Logging.Info) # hide
import ClimaAtmos as CA

simulation = CA.AtmosSimulation{Float32}(; t_end = 86400)  # 1 day
CA.solve_atmos!(simulation)
nothing # hide
```

`AtmosSimulation{FT}(...)` accepts keyword arguments for every aspect of
the simulation. When omitted, defaults are used (see
[Script vs Config Interface](@ref) for the full list).

## Customizing the simulation

### Change the grid

Run a single-column model instead of a global sphere:

```@example first_sim
grid = CA.ColumnGrid(Float32; z_elem = 30, z_max = 30000.0)
simulation = CA.AtmosSimulation{Float32}(; grid, t_end = 3600 * 6)
nothing # hide
```

Other grid types: [`SphereGrid`](@ref), [`BoxGrid`](@ref), [`PlaneGrid`](@ref).

### Change the setup

A *setup* defines the initial conditions, boundary conditions, and (optionally)
forcing for a simulation case. For example, the BOMEX shallow cumulus case:

```julia
simulation = CA.AtmosSimulation{Float32}(;
    grid = CA.ColumnGrid(Float32; z_elem = 60, z_max = 3000.0),
    initial_condition = CA.Setups.Bomex(),
    dt = 5,
    t_end = 3600 * 6,
    job_id = "my_bomex",
)
CA.solve_atmos!(simulation)
```

See the [Setups](setups.md) page for the full list of available setups and how to create
your own.

### Change the timestep and duration

`dt` is the timestep in seconds. `t_end` is the total simulation time in
seconds:

```julia
simulation = CA.AtmosSimulation{Float32}(;
    dt = 300,           # 5-minute timestep
    t_end = 86400 * 10, # 10 days
)
```

## Inspecting results

After a simulation completes, access the prognostic state through the
integrator:

```julia
Y = simulation.integrator.u

# Center (cell-center) variables
propertynames(Y.c)  # e.g., (:ρ, :ρe_tot, :uₕ, ...)

# Face (cell-interface) variables
propertynames(Y.f)  # e.g., (:w, ...)
```

Output files (NetCDF, HDF5) are written to `simulation.output_dir`.

## Terminology

These terms appear throughout the codebase and documentation:

- `Y` — the prognostic state vector. `Y.c` holds cell-center variables (density, energy, tracers), `Y.f` holds face variables (vertical velocity).
- `p` — the cache. Contains parameters, precomputed fields, and model configuration.
- `t` — current simulation time (seconds from `start_date`).
- `Yₜ` — the tendency (time derivative of `Y`).

## Using the config-based interface

The same simulation can be set up with a YAML file.

## Next steps

- [Script vs Config Interface](@ref) — detailed comparison of the two workflows
- [Single Column Models](@ref) — BOMEX, DYCOMS, RICO, and more
- [Computing and saving diagnostics](@ref) — configure output variables and formats
