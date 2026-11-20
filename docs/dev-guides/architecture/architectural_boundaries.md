# Architectural Boundaries

This guide defines the layered architecture used across CliMA model repositories and the rules that maintain the boundaries between layers. Each repo's `*_specific.md` (linked from [AGENTS.md](../AGENTS.md)) maps these layers to its concrete directories.

## 1. Layer hierarchy

CliMA packages form a directed acyclic dependency graph. The canonical layering is:

| Layer | Packages | Role |
| :---- | :------- | :--- |
| L0 | [ClimaParams.jl](https://github.com/CliMA/ClimaParams.jl), [RootSolvers.jl](https://github.com/CliMA/RootSolvers.jl), [UnrolledUtilities.jl](https://github.com/CliMA/UnrolledUtilities.jl), [LazyBroadcast.jl](https://github.com/CliMA/LazyBroadcast.jl), [MultiBroadcastFusion.jl](https://github.com/CliMA/MultiBroadcastFusion.jl), [EnsembleKalmanProcesses.jl](https://github.com/CliMA/EnsembleKalmanProcesses.jl), [NullBroadcasts.jl](https://github.com/CliMA/NullBroadcasts.jl), [RandomFeatures.jl](https://github.com/CliMA/RandomFeatures.jl) | Physical constants, calibratable parameters, generic math, and generic Julia utilities. No CliMA-internal dependencies. |
| L1 | [ClimaComms.jl](https://github.com/CliMA/ClimaComms.jl), [ClimaCore.jl](https://github.com/CliMA/ClimaCore.jl), [Thermodynamics.jl](https://github.com/CliMA/Thermodynamics.jl), [CubedSphere.jl](https://github.com/CliMA/CubedSphere.jl), [ClimaAnalysis.jl](https://github.com/CliMA/ClimaAnalysis.jl), [ClimaUtilities.jl](https://github.com/CliMA/ClimaUtilities.jl), [ClimaInterpolations.jl](https://github.com/CliMA/ClimaInterpolations.jl), [SeawaterPolynomials.jl](https://github.com/CliMA/SeawaterPolynomials.jl), [ClimaDiagnostics.jl](https://github.com/CliMA/ClimaDiagnostics.jl) | Foundations: device abstraction + MPI, fields & discretization, thermodynamic primitives, shared software tooling |
| L2 | [ClimaTimeSteppers.jl](https://github.com/CliMA/ClimaTimeSteppers.jl), [CloudMicrophysics.jl](https://github.com/CliMA/CloudMicrophysics.jl), [SurfaceFluxes.jl](https://github.com/CliMA/SurfaceFluxes.jl), [Insolation.jl](https://github.com/CliMA/Insolation.jl), [RRTMGP.jl](https://github.com/CliMA/RRTMGP.jl), [AtmosphericProfilesLibrary.jl](https://github.com/CliMA/AtmosphericProfilesLibrary.jl), [Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl) | Higher-level libraries built on L1: time integration and physics parameterisations. |
| L3 | [ClimaAtmos.jl](https://github.com/CliMA/ClimaAtmos.jl), [ClimaLand.jl](https://github.com/CliMA/ClimaLand.jl), [ClimaOcean.jl](https://github.com/CliMA/ClimaOcean.jl), [CalibrateEmulateSample.jl](https://github.com/CliMA/CalibrateEmulateSample.jl), [ClimaSeaIce.jl](https://github.com/CliMA/ClimaSeaIce.jl) | Model repos: compose L1 / L2 to integrate state variables. |
| L4 | [ClimaCoupler.jl](https://github.com/CliMA/ClimaCoupler.jl), [ClimaCalibrate.jl](https://github.com/CliMA/ClimaCalibrate.jl) | Couples multiple L3 models, toolkit for calibration pipelines |

**The rule**: a package at layer N may depend on packages at layers ≤ N, but never on packages at layers > N. Concretely:

- Physics libraries (Thermodynamics, CloudMicrophysics, SurfaceFluxes) must not depend on ClimaCore, grid types, or any model repo.
- Infrastructure libraries (ClimaCore, ClimaComms, ClimaTimeSteppers) must not depend on physics libraries or model repos.
- Model repos must not depend on ClimaCoupler.

If in doubt, verify against actual dependencies with `Pkg.dependencies`. A new dependency that creates an upward edge in this DAG is a design smell.

### Role of each layer

- **Physics libraries** (Thermodynamics.jl, CloudMicrophysics.jl, SurfaceFluxes.jl): public functions accept scalars or tuples and return scalars or `NamedTuple`s. They must not allocate `Array` or `Field` objects internally.
- **Infrastructure libraries** (ClimaCore.jl, ClimaTimeSteppers.jl, ClimaComms.jl): own the data structures, discretization, and parallelism. They define the types that model repos compose.
- **Parameter library** ([ClimaParams.jl](https://github.com/CliMA/ClimaParams.jl)): the central source of truth for physical constants and adjustable parameters that may be calibrated. All physics libraries read their constants from `ClimaParams`-derived parameter structs (see [ecosystem_conventions.md §4](ecosystem_conventions.md)).
- **Model repos** (ClimaAtmos.jl, ClimaLand.jl, ClimaOcean.jl): compose physics and infrastructure. Tendency functions call into physics libraries with extracted scalar values and write results back to fields via broadcasting.
- **Coupler** (ClimaCoupler.jl): composes multiple model repos into a coupled Earth system model.

When adding new code, place it in the layer that owns the relevant concern. Do not embed broadcasting, field allocation, or IO inside physics functions, and do not re-implement numerical algorithms inside model-level tendency code.

## 2. Parameter container design

- Keep each container focused on one physical or mathematical domain, holding physical constants and model parameters. Configuration flags, output options, and diagnostic metadata belong in the model's infrastructure layer.
- When a refactor removes the last caller of a field, remove the field too. A field with no current user is dead weight.

## 3. Avoid hidden dependencies on internals

Do not access internal or undocumented fields, functions, or modules of a dependency directly (for example, `cm2p.internal_field`). Use the documented public accessor or the primary parameter source. Anything a package neither `export`s nor documents is internal and may change without a breaking release; code that stays on the documented API survives those refactors.

Bad:

```julia
# Brittle: depends on the *internal* field names of a microphysics scheme
# struct that are not part of its documented API.
rain_terminal_velocity_coeff = cm1m_internal.rtv_coeff
```

Preferred:

```julia
# The documented public field of the unified terminal-velocity container
# (CloudMicrophysics.Parameters.TerminalVelocityParams).
rain_velocity_params = tv_params.chen2022.rain
```

## 4. Module import rules

See [SDP 2](../code-quality/software_design_patterns.md) for the rule on cross-submodule imports inside `src/`.

## Self-correction

If this guide is discovered to be stale or missing a pattern, update it.
