<div align="center">
  <img src="logo.svg" alt="ClimaAtmos.jl Logo" width="140">
</div>

# ClimaAtmos.jl

The atmosphere model of the CliMA Earth System Model: a GPU-capable global atmosphere model designed for calibration with data assimilation and machine learning.

ClimaAtmos.jl solves the compressible equations of atmospheric motion on cubed-sphere and column grids, with physics parameterizations for turbulence and convection (PROPHET, an extended prognostic EDMF scheme), cloud microphysics, and radiation. It is built on [ClimaCore.jl](https://github.com/CliMA/ClimaCore.jl) and runs on CPUs and GPUs from a single codebase.

|                   |                                                                                                                                                                                                                                                                                                                                                                      |
| -----------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Documentation** | [![stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://CliMA.github.io/ClimaAtmos.jl/stable/) [![dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://CliMA.github.io/ClimaAtmos.jl/dev/)                                                                                                                                                   |
| **Version**       | [![version](https://juliahub.com/docs/ClimaAtmos/version.svg)](https://juliahub.com/ui/Packages/General/ClimaAtmos)                                                                                                                                                                                                                                                  |
| **License**       | [![license](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/CliMA/ClimaAtmos.jl/blob/main/LICENSE)                                                                                                                                                                                                                                   |
| **Tests**         | [![gha ci](https://github.com/CliMA/ClimaAtmos.jl/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/CliMA/ClimaAtmos.jl/actions/workflows/ci.yml?query=branch%3Amain) [![buildkite](https://badge.buildkite.com/2a31b42d67409c27660a0dcce65b49294cd9c6b9f14c12f21e.svg?branch=main)](https://buildkite.com/clima/climaatmos-ci/builds?branch=main) |
| **Code Coverage** | [![codecov](https://codecov.io/gh/CliMA/ClimaAtmos.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/CliMA/ClimaAtmos.jl)                                                                                                                                                                                                                                       |
| **Downloads**     | [![Downloads](https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Ftotal_downloads%2FClimaAtmos&query=total_requests&label=Downloads)](https://juliapkgstats.com/pkg/ClimaAtmos)                                                                                                                                                |

<p align="center">
  <img src="https://github.com/user-attachments/assets/778b0c14-a5d7-4907-82db-6d1f8a0c5b07" alt="Condensed water path from a global ClimaAtmos simulation">
</p>

Condensed water path from a global simulation initialized with ERA5 on 8-31-25 00Z. Output every 30 minutes; ran for ~4 days.

## Features

  - **Global and single-column configurations**: cubed-sphere grids with topography for global simulations; boxes, planes, and columns for process studies (BOMEX, DYCOMS, RICO, and other standard cases)
  - **Turbulence and convection**: TKE-based eddy diffusion and the PROPHET scheme (an extended, prognostic eddy-diffusivity mass-flux (EDMF) scheme), designed for calibration with data assimilation and machine learning
  - **Cloud microphysics**: 0-moment to 2-moment bulk schemes, plus the P3 ice scheme, via [CloudMicrophysics.jl](https://github.com/CliMA/CloudMicrophysics.jl)
  - **Radiation**: RRTMGP radiative transfer
  - **ERA5 and GCM-driven** initial conditions and forcing
  - **Configurable diagnostics** with NetCDF and HDF5 output
  - **Restarts and checkpointing** for long simulations
  - **GPU support**: runs on CPUs and NVIDIA GPUs from the same codebase
  - **Composable configuration**: script and YAML-config interfaces for every aspect of a simulation

## Installation

ClimaAtmos.jl is a registered Julia package (recommended Julia: v1.11):

```julia
using Pkg
Pkg.add("ClimaAtmos")
```

## Quick Example

The simplest simulation uses all defaults: it solves the dry compressible equations on a global cubed-sphere grid from a hydrostatically balanced, slightly perturbed state:

```julia
import ClimaAtmos as CA

simulation = CA.AtmosSimulation{Float32}(; t_end = "1days")
CA.solve_atmos!(simulation)
```

Every aspect of the simulation can be customized through keyword arguments, for example a single-column model:

```julia
grid = CA.ColumnGrid(Float32; z_elem = 30, z_max = 30000.0)
simulation = CA.AtmosSimulation{Float32}(; grid, t_end = "6hours")
```

See [Your First Simulation](https://CliMA.github.io/ClimaAtmos.jl/dev/first_simulation/) in the documentation for a guided introduction.

## Documentation

  - **[Stable docs](https://CliMA.github.io/ClimaAtmos.jl/stable/)**: equations, parameterizations, configuration reference, and API
  - **[Dev docs](https://CliMA.github.io/ClimaAtmos.jl/dev/)**: latest development version
  - **[Available diagnostics](https://CliMA.github.io/ClimaAtmos.jl/dev/available_diagnostics/)**: output variables

## Integration with CliMA models

ClimaAtmos.jl is a component of the [CliMA](https://github.com/CliMA) Earth System Model:

  - [ClimaCore.jl](https://github.com/CliMA/ClimaCore.jl): dynamical core and discretization tools
  - [ClimaCoupler.jl](https://github.com/CliMA/ClimaCoupler.jl): coupling to ocean, land, and sea ice components
  - [Thermodynamics.jl](https://github.com/CliMA/Thermodynamics.jl): moist thermodynamics, shared across all CliMA components for energetic consistency
  - [ClimaParams.jl](https://github.com/CliMA/ClimaParams.jl): the single source of truth for all model parameters

See [The CliMA Ecosystem](https://CliMA.github.io/ClimaAtmos.jl/dev/ecosystem/) in the documentation for the full architectural overview, including [Insolation.jl](https://github.com/CliMA/Insolation.jl), [RRTMGP.jl](https://github.com/CliMA/RRTMGP.jl), [SurfaceFluxes.jl](https://github.com/CliMA/SurfaceFluxes.jl), and [CloudMicrophysics.jl](https://github.com/CliMA/CloudMicrophysics.jl).

## Contributing

If you're interested in contributing to ClimaAtmos, we welcome contributions of any size! Let us know by [opening an issue](https://github.com/CliMA/ClimaAtmos.jl/issues/new) if you'd like to work on a new feature.

Contributors should follow the shared CliMA engineering standards in [`docs/dev-guides/`](docs/dev-guides/), which cover architecture, performance, code quality, documentation, and workflows. These are vendored from [CliMA/DeveloperGuides](https://github.com/CliMA/DeveloperGuides). The repo's [`AGENTS.md`](AGENTS.md) is a starting point for AI agents with repo-specific guidance. See also the [contributor's guide](https://clima.github.io/ClimaAtmos.jl/dev/contributor_guide/).
