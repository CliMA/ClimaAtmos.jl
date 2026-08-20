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
  - **Calibration-ready**: every parameter is overridable, and the model plugs into [ClimaCalibrate.jl](https://github.com/CliMA/ClimaCalibrate.jl) and [EnsembleKalmanProcesses.jl](https://github.com/CliMA/EnsembleKalmanProcesses.jl) to learn parameters from LES and observations
  - **Coupled Earth system simulations**: the atmosphere component of [ClimaCoupler.jl](https://github.com/CliMA/ClimaCoupler.jl), coupled to land, ocean, and sea ice

## Installation

ClimaAtmos.jl is a registered Julia package (recommended Julia: v1.11):

```julia
using Pkg
Pkg.add("ClimaAtmos")
```

## Quick start

A complete global simulation is three lines. Presets bundle a grid, an initial
state, and matching physics, so nothing has to be assembled by hand:

```julia
import ClimaAtmos as CA

simulation = CA.Presets.aquaplanet(Float32; t_end = "10days")
CA.solve_atmos!(simulation)
```

That is a moist aquaplanet on the cubed sphere, with 0-moment microphysics,
prescribed zonally symmetric SSTs, and idealized insolation. NetCDF diagnostics
and HDF5 checkpoints are written under `simulation.output_dir`.

Nothing in the script changes when the hardware does: ClimaComms.jl reads the
device and the communication context from the environment, so the same file runs
on a laptop, on a GPU, and across nodes.

```bash
julia --project my_run.jl                                     # CPU
CLIMACOMMS_DEVICE=CUDA julia --project my_run.jl              # one GPU
CLIMACOMMS_CONTEXT=MPI CLIMACOMMS_DEVICE=CUDA \
    srun --ntasks=4 julia --project my_run.jl                 # four GPUs
```

The first construction and solve in a session compile a large amount of code
and take a few minutes; later calls are fast.
[Your First Simulation](https://CliMA.github.io/ClimaAtmos.jl/dev/first_simulation/)
walks through the same run step by step, and
[Scripting Simulations](https://CliMA.github.io/ClimaAtmos.jl/dev/scripting_simulations/)
covers each argument in turn.

## Examples

The examples below follow the experiments papers are usually built from: single
columns evaluated against LES and field campaigns, global runs, parameter
sensitivity, calibration against data, and coupled Earth system simulations.

### Single-column cases against LES and field campaigns

Parameterization work usually starts in a column. The canonical
intercomparison cases (BOMEX, DYCOMS, RICO, SOARES, GABLS, TRMM) are built in,
and a preset pairs the case's initial state, forcings, and surface conditions
with a matching column grid. Because the physics is a separate argument, the
same case runs with or without a turbulence-convection scheme -- the comparison
most parameterization papers report:

```julia
import ClimaAtmos as CA

simulation = CA.Presets.bomex(
    Float32;
    # PROPHET, the prognostic eddy-diffusivity mass-flux (EDMF) scheme
    model = CA.Presets.prognostic_edmf(Float32),
    dt = "10secs",
    t_end = "6hours",
    diagnostics = CA.DiagnosticsConfig(;
        default = false,
        additional = [(;
            # profiles to plot against LES: thermodynamics, cloud, and the
            # updraft properties the scheme is judged on
            short_name = ["ta", "hus", "clw", "cl", "tke", "arup", "entr", "detr"],
            period = "10mins",
            reduction = "average",
        )],
    ),
    job_id = "bomex_prophet",
)
CA.solve_atmos!(simulation)
```

The published version of each case, with the diagnostics used in CI, is a
configuration file in `config/model_configs/`
(`prognostic_edmfx_bomex_column.yml`, `prognostic_edmfx_dycoms_rf01_column.yml`,
`prognostic_edmfx_rico_column.yml`, and so on).

Columns can also be driven by data rather than by an idealized case: reanalysis
at any site (`ReanalysisTimeVarying`, including a monthly-mean composite diurnal
cycle), LES-matched GCM forcing at cfSites (`GCMDriven`), and the ARM SGP
variational analysis. These are how a scheme is confronted with observations
before it goes global:

```julia
config = CA.AtmosConfig(
    "config/model_configs/prognostic_edmfx_tv_era5driven_column.yml";
    job_id = "era5_column",
)
CA.solve_atmos!(CA.AtmosSimulation(config))
```

See [Running Single-Column Cases](https://CliMA.github.io/ClimaAtmos.jl/dev/single_column/)
for the case list and the externally-driven setups.

### A global simulation with topography, clouds, and radiation

Each piece of a global run is an argument: the grid (resolution, vertical
stretching, orography), the model (microphysics, turbulence and convection,
radiation), and the diagnostics. Monthly means are written straight out of the
run, aligned to calendar months, so climatologies need no post-processing:

```julia
import ClimaAtmos as CA
using Dates

simulation = CA.AtmosSimulation{Float32}(;
    grid = CA.SphereGrid(
        Float32;
        h_elem = 30,                        # ~110 km horizontal resolution
        z_elem = 63,
        z_max = 55000.0,
        topography = CA.EarthTopography(),  # ETOPO2022 orography
    ),
    model = CA.Presets.prognostic_edmf_1m(  # PROPHET + 1-moment microphysics
        Float32;
        radiation_mode = CA.RRTMGPI.AllSkyRadiation(),
    ),
    dt = "120secs",
    start_date = DateTime(2010, 1, 1),
    t_end = "360days",
    diagnostics = CA.DiagnosticsConfig(;
        default = true,
        additional = [(;
            short_name = ["ta", "hus", "cl", "clwvi", "pr", "rlut", "rsut", "hfls", "hfss"],
            period = "monthly",
            reduction = "average",
        )],
    ),
    checkpoint_frequency = "1months",  # aligned with the monthly averages
    job_id = "global_run",
)
CA.solve_atmos!(simulation)
```

Long runs on a cluster are split across jobs by adding
`detect_restart_file = true`: relaunching the same script picks up the most
recent checkpoint instead of starting over.

Initializing from reanalysis instead of an idealized profile takes one
argument, `setup = CA.Setups.WeatherModel("20250831")` for an ERA5 snapshot (as
in the figure above), or `CA.Setups.AMIPFromERA5(...)` for an AMIP-style start.
Both read pre-processed ERA5 files distributed as ClimaArtifacts, currently
staged on CliMA-managed clusters.

### Perturbed-parameter ensembles

Reporting sensitivity to a parameterization constant means running the same
simulation many times with one number changed. Every parameter lives in a
ClimaParams TOML dictionary, so an override is a dictionary entry (or a TOML
file committed with the paper) and the ensemble is a loop:

```julia
import ClimaAtmos as CA
import ClimaParams as CP

for entr_coeff in (0.1, 0.2, 0.4)
    params = CA.ClimaAtmosParameters(
        CP.create_toml_dict(
            Float32;
            override_file = Dict(
                "entr_coeff" => Dict("value" => entr_coeff, "type" => "float"),
            ),
        ),
    )
    simulation = CA.Presets.bomex(
        Float32;
        params,
        model = CA.Presets.prognostic_edmf(Float32),
        t_end = "6hours",
        job_id = "bomex_entr_$(entr_coeff)",
    )
    CA.solve_atmos!(simulation)
end
```

### Calibrating parameters against data

ClimaAtmos is designed to be *learned from data*, not only run. The same forward
model plugs into [ClimaCalibrate.jl](https://github.com/CliMA/ClimaCalibrate.jl),
which drives ensembles with
[EnsembleKalmanProcesses.jl](https://github.com/CliMA/EnsembleKalmanProcesses.jl)
to fit parameters to LES, field campaigns, or reanalysis. The method is
derivative-free, so it calibrates the model exactly as it is run, with no
adjoint or differentiable rewrite, and each iteration is embarrassingly
parallel over ensemble members, on a laptop or through Slurm:

```julia
import ClimaCalibrate
import EnsembleKalmanProcesses as EKP
import EnsembleKalmanProcesses.ParameterDistributions as PD

# Priors for the parameters being learned
prior = EKP.combine_distributions([
    PD.constrained_gaussian("entr_coeff", 0.2, 0.1, 0.0, Inf),
    PD.constrained_gaussian("detr_coeff", 0.002, 0.001, 0.0, Inf),
])

# `observations` is a vector of EKP.Observation built from LES or field data,
# mapped onto model diagnostics with ClimaAnalysis
ekp = EKP.EnsembleKalmanProcess(
    EKP.ObservationSeries(observations),
    EKP.TransformUnscented(prior; impose_prior = true),
)

ClimaCalibrate.calibrate(
    ClimaCalibrate.JuliaBackend(),
    ekp,
    interface,
    n_iterations,
    prior,
    output_dir,
)
```

Every ensemble member is one `solve_atmos!` behind
`ClimaCalibrate.forward_model`, implemented in
[`calibration/model_interface.jl`](calibration/model_interface.jl).
[`calibration/experiments/`](calibration/) holds runnable end-to-end
experiments, including a perfect-model single-column calibration that recovers
known parameters.

### Coupling to land, ocean, and sea ice

ClimaAtmos is the atmosphere component of a full Earth system model.
[ClimaCoupler.jl](https://github.com/CliMA/ClimaCoupler.jl) handles the coupled
time stepping and the exchange of boundary fluxes between components, and its
`experiments/ClimaEarth/` driver runs AMIP-style simulations with ClimaAtmos and
[ClimaLand.jl](https://github.com/CliMA/ClimaLand.jl):

```bash
git clone https://github.com/CliMA/ClimaCoupler.jl && cd ClimaCoupler.jl
julia --project=experiments/ClimaEarth -e "using Pkg; Pkg.instantiate()"
julia --project=experiments/ClimaEarth experiments/ClimaEarth/run_amip.jl \
    --config_file config/ci_configs/amip_default.yml --job_id amip_default
```

The driver builds its atmosphere through `CA.AtmosConfig` and
`CA.get_simulation`, so the configuration keys from a column or aquaplanet study
carry over to the coupled system unchanged.

### Analyzing the output

Diagnostics are NetCDF files with CF-style short names, read with
[ClimaAnalysis.jl](https://github.com/CliMA/ClimaAnalysis.jl) and plotted through
Makie. Because reanalysis and observational NetCDF files load as the same
`OutputVar` objects, model-versus-observations comparisons use the same few
functions:

```julia
using ClimaAnalysis
import ClimaAnalysis.Visualize as viz
import CairoMakie

simdir = SimDir("output/global_run/output_active")
ta = get(simdir, "ta")                     # air temperature
ta_zonal = average_lon(average_time(ta))   # time mean, then zonal mean

fig = CairoMakie.Figure()
viz.plot!(fig, ta_zonal)                   # latitude-height section
CairoMakie.save("zonal_mean_temperature.png", fig)
```

### Reproducible configurations

Most of what the examples above set from Julia is also a YAML key, and a
configuration file is a single committable record of a numerical experiment.
It is how the CI cases, the production runs, and the calibration experiments
are specified:

```julia
config = CA.AtmosConfig("config/longrun_configs/amip_target.yml"; job_id = "amip")
CA.solve_atmos!(CA.AtmosSimulation(config))
```

`config/model_configs/` holds the process-study and test cases (one per
single-column case, plus aquaplanets and baroclinic waves);
`config/longrun_configs/` holds the production configurations. Both interfaces
build the same simulation object; see
[Script vs Config Interface](https://CliMA.github.io/ClimaAtmos.jl/dev/interfaces/).

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
