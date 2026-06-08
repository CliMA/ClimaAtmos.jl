# ClimaAtmos.jl

ClimaAtmos.jl is a Julia library for building atmospheric circulation models. It
supports global configurations, Cartesian geometries (e.g., doubly periodic
boxes used for large-eddy simulations), and single-column configurations. It
runs on CPUs and GPUs and is designed to work with data assimilation and machine
learning tools.

## Quickstart

```julia
import ClimaAtmos as CA

simulation = CA.AtmosSimulation{Float32}(; t_end = "1days")
CA.solve_atmos!(simulation)
```

This runs a 1-day global simulation with default settings (cubed-sphere grid, decaying temperature profile, IMEX timestepping).

## Capabilities

- **Global simulations** on cubed-sphere grids with topography
- **Single-column models** for BOMEX, DYCOMS, RICO, and other standard cases
- **GPU acceleration** via CUDA
- **ERA5 and GCM-driven** initial conditions and forcing
- **EDMF turbulence** (prognostic and diagnostic)
- **Microphysics** (0-moment, 1-moment, 2-moment)
- **Configurable diagnostics** with NetCDF and HDF5 output
- **Restarts and checkpointing** for long simulations

## Finding your way around

Each section of these docs answers a different need -- head for the one that
matches what you're doing:

- **Getting Started** -- new to ClimaAtmos? Start here.
    - [Installation](installation.md) -- install the package, or run from a clone
    - [Your First Simulation](first_simulation.md) -- run and customize a simulation
    - [Script vs Config Interface](interfaces.md) -- the two ways to configure a run
- **How-to Guides** -- task recipes for running and configuring simulations.
    - Running simulations: [single-column cases](single_column_prospect.md), [radiative equilibrium](radiative_equilibrium.md), [restarts](restarts.md), [REPL debugging](repl_scripts.md)
    - Configuration: [custom configurations](config.md), [parameters](parameters.md)
    - [Computing and saving diagnostics](diagnostics.md)
- **Explanation** -- the science and numerics behind the model.
    - Dynamics & numerics: [governing equations](equations.md), [implicit solver](implicit_solver.md), [integer time (ITime)](itime.md)
    - Physics & parameterizations: [microphysics](microphysics.md), EDMF ([prognostic](edmf_equations.md), [diagnostic](diagnostic_edmf_equations.md)), [gravity-wave drag](gravity_wave.md), [ocean surface albedo](surface_albedo.md), [topography](topography.md)
- **Reference** -- look-up material.
    - [API](api.md), [Glossary](glossary.md), [Grids](grids.md), [Setups](setups.md), [Surface conditions](surface_conditions.md), [Trace gases](tracers.md), [Available diagnostics](available_diagnostics.md), [Bibliography](references.md)
- **Developer Guide** -- [contributing](contributor_guide.md) and the [Buildkite longrun jobs](longruns.md)

New here? Start with [Installation](@ref) and [Your First Simulation](@ref), then pick
the workflow that suits you in [Script vs Config Interface](@ref). For column experiments,
see [Single Column Models](@ref).
