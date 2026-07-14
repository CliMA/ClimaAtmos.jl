<div align="center">
  <img src="docs/src/assets/logo.svg" alt="ClimaAtmos.jl Logo" width="128" height="128">

# ClimaAtmos.jl

</div>

<p align="center">
  <strong>Atmosphere components of the CliMA software stack.</strong>
</p>

|||
|------------------:|:------------------------------------------------------------|
| **Documentation** | [![stable][docs-stable-img]][docs-stable-url] [![dev][docs-dev-img]][docs-dev-url] |
| **Tests**         | [![gha ci][gha-ci-img]][gha-ci-url] [![buildkite][bk-ci-img]][bk-ci-url] |
| **Code Coverage** | [![codecov][codecov-img]][codecov-url]                      |
| **Downloads**     | [![Downloads][dlt-img]][dlt-url]                            |
| **Contributing**  | [![colprac][colprac-img]][colprac-url]                      |

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://CliMA.github.io/ClimaAtmos.jl/stable/

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://CliMA.github.io/ClimaAtmos.jl/dev/

[gha-ci-img]: https://github.com/CliMA/ClimaAtmos.jl/actions/workflows/ci.yml/badge.svg
[gha-ci-url]: https://github.com/CliMA/ClimaAtmos.jl/actions/workflows/ci.yml

[bk-ci-img]: https://badge.buildkite.com/2a31b42d67409c27660a0dcce65b49294cd9c6b9f14c12f21e.svg?branch=main
[bk-ci-url]: https://buildkite.com/clima/climaatmos-ci

[codecov-img]: https://codecov.io/gh/CliMA/ClimaAtmos.jl/branch/main/graph/badge.svg
[codecov-url]: https://codecov.io/gh/CliMA/ClimaAtmos.jl

[dlt-img]: https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Ftotal_downloads%2FClimaAtmos&query=total_requests&suffix=%2Ftotal&label=Downloads
[dlt-url]: http://juliapkgstats.com/pkg/ClimaAtmos

[colprac-img]: https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet?style=flat-square
[colprac-url]: https://github.com/SciML/ColPrac

ClimaAtmos.jl is the atmosphere components of the CliMA software stack. We strive for a user interface that makes ClimaAtmos.jl as friendly and intuitive to use as possible, allowing users to focus on the science. It runs on CPUs and GPUs and is designed to work with data assimilation and machine learning tools.

<p align="center">
  <img src="https://github.com/user-attachments/assets/778b0c14-a5d7-4907-82db-6d1f8a0c5b07" alt="animation (1)">
</p>

Condensed water path from a global simulation using diagnostic EDMF and 0M microphysics, initialized with ERA5 on 8-31-25 00Z. Output every 30 minutes; ran for ~4 days.

## Features

- **Global simulations** on cubed-sphere grids with topography
- **Single-column models** for BOMEX, DYCOMS, RICO, and other standard cases
- **GPU acceleration** via CUDA
- **ERA5 and GCM-driven** initial conditions and forcing
- **Turbulence and convection** (TKE-based diffusion and prognostic EDMF)
- **Microphysics** (0-moment, 1-moment, 2-moment)
- **Configurable diagnostics** with NetCDF and HDF5 output
- **Restarts and checkpointing** for long simulations

## Quick Example

```julia
import ClimaAtmos as CA

simulation = CA.AtmosSimulation{Float32}(; t_end = "1days")
CA.solve_atmos!(simulation)
```

This runs a 1-day global simulation with default settings (cubed-sphere grid, decaying temperature profile, IMEX timestepping). See the [documentation](https://CliMA.github.io/ClimaAtmos.jl/dev/) for how to customize a simulation, including the script vs. config interfaces.

## Installation instructions

Recommended Julia: Stable release v1.11.6

ClimaAtmos.jl is a [registered Julia package](https://julialang.org/packages/). To install

```julia
julia> using Pkg

julia> Pkg.add("ClimaAtmos")

```

Alternatively, download the `ClimaAtmos`
[source](https://github.com/CliMA/ClimaAtmos.jl) with:

```
$ git clone https://github.com/CliMA/ClimaAtmos.jl.git
```

Now change into the `ClimaAtmos.jl` directory with

```
$ cd ClimaAtmos.jl
```

To use ClimaAtmos, you need to instantiate all dependencies with:

```
$ julia --project
julia> ]
(ClimaAtmos) pkg> instantiate
```

## Running instructions

Currently, the simulations are stored in the `test` folder. Run all the test cases with the following commands.

First, launch Julia from the `ClimaAtmos.jl/` directory with the project active:

```
$ julia --project
```

Then, in the Julia REPL, switch to the package manager by pressing `]` and run the tests:

```pkg
(ClimaAtmos) pkg> test
```

Or run from the command line:

```
$ julia --project -e 'using Pkg; Pkg.test()'
```

If you run into issues when running the test suite this way, please open an issue.

## Contributing

If you're interested in contributing to the development of ClimaAtmos we want your help no matter how big or small a contribution you make! It's always great to have new people look at the code with fresh eyes: you will see errors that other developers have missed.

Let us know by [opening an issue](https://github.com/CliMA/ClimaAtmos.jl/issues/new) if you'd like to work on a new feature.

Contributors should follow the shared CliMA engineering standards in [`docs/dev-guides/`](docs/dev-guides/), which cover architecture, performance, code quality, documentation, and workflows. These are vendored from [CliMA/DeveloperGuides](https://github.com/CliMA/DeveloperGuides). The repo's [`AGENTS.md`](AGENTS.md) is a starting point for AI agents with repo-specific guidance.

For more information, check out our [contributor's guide](https://CliMA.github.io/ClimaAtmos.jl/dev/contributor_guide/).
