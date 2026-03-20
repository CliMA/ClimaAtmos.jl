# ClimaAtmos.jl

ClimaAtmos.jl is a Julia library for building atmospheric circulation models. It
supports global configurations, Cartesian geometries (e.g., doubly periodic
boxes used for large-eddy simulations), and single-column configurations. It
runs on CPUs and GPUs and is designed to work with data assimilation and machine
learning tools.

## Quickstart

```julia
import ClimaAtmos as CA

simulation = CA.AtmosSimulation{Float64}(; t_end = 86400)
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

## Next steps

- [Installation](@ref) — install Julia and ClimaAtmos
- [Your First Simulation](@ref) — run and customize a simulation
- [Script vs Config Interface](@ref) — choose the right workflow
- [Single Column Models](@ref) — SCM cases and tutorials
