# The CliMA Ecosystem

ClimaAtmos composes focused, independently developed and tested packages
from the [CliMA](https://github.com/CliMA) ecosystem. Each package owns one
aspect of the model (thermodynamics, radiative transfer, surface fluxes,
microphysics, parameters), and ClimaAtmos wires them together into an atmosphere model. This page explains what each
package contributes, why the decomposition matters physically, and where each
package enters the ClimaAtmos source code.

Two consistency principles motivate the decomposition:

  - **Energetic consistency.** All components of the model (and of the wider
    Earth system model) must use the same thermodynamic formulation (for
    example, the same saturation vapor pressure) or energy and water budgets
    do not close. A single shared thermodynamics package guarantees this by
    construction.
  - **A single source of truth for parameters.** All physical constants and
    calibratable parameters live in one place. The model can be adapted to
    past climates or other planetary configurations just by changing
    parameters there, without touching model code; calibration tools can
    target any parameter uniformly.

## How the packages fit together

```
┌──────────────────────────────────────────────────────────────┐
│ ClimaAtmos.jl — Equations, Parameterizations, Configuration  │
├──────────────────────────────┬───────────────────────────────┤
│ Physics Libraries            │ Numerics & Infrastructure     │
│   Insolation.jl              │   ClimaCore.jl                │
│   RRTMGP.jl                  │   ClimaTimeSteppers.jl        │
│   SurfaceFluxes.jl           │   ClimaComms.jl               │
│   CloudMicrophysics.jl       │   ClimaDiagnostics.jl         │
│                              │   ClimaUtilities.jl           │
├──────────────────────────────┴───────────────────────────────┤
│ Shared foundation, used by every package above:              │
│   Thermodynamics.jl  ·  ClimaParams.jl                       │
└──────────────────────────────────────────────────────────────┘
```

The physics libraries are themselves clients of Thermodynamics.jl and
ClimaParams.jl; this shared foundation, not ClimaAtmos, keeps the
formulations consistent across packages.

## Shared foundation

### Thermodynamics.jl — one thermodynamic formulation for all of CliMA

[Thermodynamics.jl](https://clima.github.io/Thermodynamics.jl/stable/) is the
unified moist thermodynamics library for all CliMA code. Every package that
needs a thermodynamic quantity (ClimaAtmos itself, CloudMicrophysics.jl,
SurfaceFluxes.jl, the land and ocean models) computes it from the same
formulation, e.g., for saturation vapor pressure, latent heats, or internal
energies. This is necessary for energetic consistency: if the atmosphere and
the surface used different saturation vapor pressures, spurious energy and
water fluxes would appear at their interface. The formulation treats moist air
as a calorically perfect ideal mixture of dry air, water vapor, and condensates
(see [Yatunin2026](@cite), Section 2, and the [governing equations](equations.md)).

*Where it enters ClimaAtmos:* everywhere a thermodynamic state is constructed
or queried, typically as `TD.PhaseEquil_ρeq(...)`, `TD.air_temperature(...)`,
etc., with parameters accessed via `CAP.thermodynamics_params(params)`.

### ClimaParams.jl — single source of truth for parameters

[ClimaParams.jl](https://clima.github.io/ClimaParams.jl/stable/) stores every
model parameter in one central TOML database shared across all CliMA
packages: fixed physical constants such as gas constants and latent heats, as
well as calibratable closure parameters such as entrainment coefficients and
mixing-length parameters.
Because packages never hard-code parameter values, the model can be adapted to
past climates or other planetary configurations just by changing parameters
(e.g., a different solar constant, rotation rate, gravity, or CO2
concentration), and calibration frameworks can override any parameter through
the same [TOML interface](parameters.md) used for manual experimentation.

*Where it enters ClimaAtmos:* `src/parameters/create_parameters.jl` assembles
the `ClimaAtmosParameters` struct from the ClimaParams TOML database, including
the parameter sets handed to Thermodynamics.jl, CloudMicrophysics.jl,
SurfaceFluxes.jl, RRTMGP.jl, and Insolation.jl; YAML configurations override
parameters via `toml: [...]`.

## Physics libraries

### Insolation.jl — solar forcing at the top of the atmosphere

[Insolation.jl](https://clima.github.io/Insolation.jl/stable/) sets the
insolation at the top of the atmosphere: it computes the incoming solar flux
and solar zenith angle from time, location, and the planet's orbital parameters
(eccentricity, obliquity, precession). Because the orbital parameters are just
parameters, paleoclimate (Milankovitch) configurations and idealized insolation
experiments require no code changes.

*Where it enters ClimaAtmos:* the radiation callback
(`src/callbacks/callbacks.jl`) calls `Insolation.insolation` to update the
zenith angle and top-of-atmosphere flux that RRTMGP consumes; the available
insolation modes (idealized, time-varying, RCEMIP-II, GCM-driven, externally
driven, and Larcform1) are selected by the `insolation` configuration argument.

### RRTMGP.jl — radiative transfer in the atmosphere

[RRTMGP.jl](https://clima.github.io/RRTMGP.jl/stable/) handles radiative
transfer within the atmosphere: a GPU-capable Julia implementation of the
RRTMGP correlated k-distribution model that computes longwave and shortwave
fluxes from the atmospheric state, trace-gas concentrations, clouds, and
aerosols. ClimaAtmos supports gray, clear-sky, and all-sky radiation modes.

*Where it enters ClimaAtmos:*
`src/parameterized_tendencies/radiation/RRTMGPInterface.jl` wraps the RRTMGP
solvers; `radiation.jl` and the radiation callback keep the inputs (state,
[trace gases](trace_gases.md), clouds, insolation) up to date. The mode is
selected by the `rad` configuration argument.

### SurfaceFluxes.jl — turbulent exchange with the surface

[SurfaceFluxes.jl](https://clima.github.io/SurfaceFluxes.jl/stable/) provides
the turbulent fluxes of energy, momentum, water, and tracers at ocean, ice, and
land surfaces, based on Monin–Obukhov similarity theory. It is shared with the
land and ocean models, so that both sides of each interface compute the same
exchange from the same theory.

*Where it enters ClimaAtmos:* `src/surface_conditions/` builds surface states
and evaluates SurfaceFluxes.jl to fill the lower boundary conditions; see
[Surface conditions](surface_conditions.md) for the full user and developer
guide. In coupled simulations, ClimaCoupler.jl supplies the surface states
instead.

### CloudMicrophysics.jl — cloud and precipitation processes

[CloudMicrophysics.jl](https://clima.github.io/CloudMicrophysics.jl/stable/)
provides the microphysical process rates (condensation/evaporation,
autoconversion, accretion, sedimentation velocities, ice nucleation, aerosol
activation) for the 0-moment to 2-moment bulk schemes. ClimaAtmos calls these
rates pointwise and handles their coupling to the dynamics (advection,
sedimentation fluxes, energy sinks); see [Microphysics](microphysics.md).

*Where it enters ClimaAtmos:* `src/parameterized_tendencies/microphysics/`,
selected by the `microphysics_model` configuration argument.

## Numerical and computational foundation

  - [ClimaCore.jl](https://clima.github.io/ClimaCore.jl/stable/) provides the
    spatial discretization: spectral-element/finite-difference grids, fields,
    and the [discrete operators](equations.md) used to express the equations.
  - [ClimaTimeSteppers.jl](https://clima.github.io/ClimaTimeSteppers.jl/stable/)
    provides the IMEX time integrators used together with the
    [implicit solver](implicit_solver.md).
  - [ClimaComms.jl](https://clima.github.io/ClimaComms.jl/stable/) abstracts
    the compute device and communication (CPU threads, CUDA, MPI), so the same
    code runs on laptops, clusters, and GPUs.
  - [ClimaDiagnostics.jl](https://clima.github.io/ClimaDiagnostics.jl/stable/)
    schedules, reduces, and writes [diagnostics](diagnostics.md).
  - [ClimaUtilities.jl](https://clima.github.io/ClimaUtilities.jl/stable/)
    provides shared infrastructure: file readers and regridders for input data,
    time management ([ITime](itime.md)), and output-directory handling.

## Input data

  - [ClimaArtifacts](https://github.com/CliMA/ClimaArtifacts) hosts the
    versioned input datasets (topography, ozone and trace-gas concentrations,
    aerosol climatologies, ERA5-derived forcing files) that simulations
    download on demand as Julia artifacts, so runs are reproducible down to
    the input data.
  - [AtmosphericProfilesLibrary.jl](https://github.com/CliMA/AtmosphericProfilesLibrary.jl)
    provides the published reference profiles (BOMEX, DYCOMS, GABLS, …) from
    which the [single-column setups](setups.md) build their initial conditions.

## Calibration

ClimaAtmos is designed to be calibrated against data, and the parameter
architecture above makes this work: because every closure parameter
lives in ClimaParams.jl, a calibration only has to write TOML overrides.

  - [ClimaCalibrate.jl](https://clima.github.io/ClimaCalibrate.jl/stable/)
    orchestrates calibration-with-data workflows: it runs ensembles of
    ClimaAtmos simulations (locally or on HPC clusters), maps observations to
    model diagnostics, and iterates the parameter ensemble.
  - [EnsembleKalmanProcesses.jl](https://clima.github.io/EnsembleKalmanProcesses.jl/stable/)
    provides the gradient-free ensemble Kalman methods used to update the
    parameters, which require only forward model runs, not adjoints or
    derivatives of the model.

*Where it enters ClimaAtmos:* the `calibration/` directory provides the
ClimaCalibrate model interface (`calibration/model_interface.jl`) and example
experiments; see `calibration/README.md`.

## Coupling

In stand-alone simulations, ClimaAtmos computes its own surface conditions. In
Earth-system-model configurations,
[ClimaCoupler.jl](https://clima.github.io/ClimaCoupler.jl/stable/) mediates the
exchange between ClimaAtmos and the land, ocean, and sea-ice models: the
coupler supplies surface states (temperature, albedo, roughness) and receives
the SurfaceFluxes.jl-computed fluxes, with Thermodynamics.jl and ClimaParams.jl
guaranteeing that all components agree on the underlying formulation and
constants.
