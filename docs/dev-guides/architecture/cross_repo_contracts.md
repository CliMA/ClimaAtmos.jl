# Cross-Repo Contracts

This guide documents the conventions for calling ecosystem packages from CliMA model repositories. Rules are at the call-site level; internal package APIs are not documented here.

## General principle

Always pass the package's *parameter container* (for example, `thermodynamics_params`, `surface_flux_params`) into physics functions; [ecosystem_conventions.md §4](ecosystem_conventions.md) describes how constants reach it. This keeps model components consistent and makes calibration transparent.

## How to find the current API of a CliMA dependency

The dependency's source is usually not checked out next to the model repo. Use this order:

1. **`NEWS.md`** of the dependency, if accessible: it lists API changes per release.
2. **The dev'd path under `~/.julia/dev/<Package>.jl`**, if the user has the package dev'd locally.
3. **The package's `docs/src/`**, which usually documents the supported API.
4. **Existing call sites in this repo**: grep for the package's module alias in `src/` to see how it is already used.

Treat anything not in the package's `docs/` as internal and unstable (see [architectural_boundaries.md §3](architectural_boundaries.md)).

## Thermodynamics.jl

- Pass the thermodynamics parameter container (e.g. `p.params.thermodynamics_params` in ClimaAtmos) into thermodynamic functions.
- The public API is functional and stateless: functions take a parameter container and the relevant scalar arguments directly.
- Many functions are dispatched on a *formulation type* that names the independent variables. The available formulations (subtypes of `IndepVars`) are `TD.ρe()`, `TD.pe()`, `TD.ph()`, `TD.pρ()`, `TD.pθ_li()`, `TD.ρθ_li()`. For example: `TD.air_temperature(thp, TD.ph(), h, q_tot, q_liq, q_ice)`.
- For iterative phase-equilibrium calculations inside GPU kernels, prefer the fixed-iteration `saturation_adjustment` variants (the GPU default is a 2-iteration safeguarded Newton solve with no stopping test) to avoid thread divergence. See [SDP 19](../code-quality/software_design_patterns.md) and the [Branchless Code Guide §4–5](../performance/branchless_code.md).

## CloudMicrophysics.jl

- The microphysics scheme is passed as a singleton type (e.g. `Microphysics0Moment()`, `Microphysics1Moment()`, `Microphysics2Moment()`); dispatch on it eliminates dead branches at compile time.
- The bulk-tendency wrappers (e.g. `bulk_microphysics_tendencies`) return `NamedTuple`s; materialize them with the scratch pattern in the "Materialization" and "Multi-field updates" subsections of [GPU Performance Guide §3](../performance/gpu_performance.md). Process-rate primitives (e.g. `accretion`, `terminal_velocity`, `conv_q_icl_to_q_sno`) return scalars and can be broadcast directly.
- The terminal-velocity parameters live in a unified container `CMP.TerminalVelocityParams` with fields `stokes`, `chen2022`, `blk1m`. Use these documented fields (see [architectural_boundaries.md §3](architectural_boundaries.md)).

## SurfaceFluxes.jl

- Pass a `SurfaceFluxes.Parameters.SurfaceFluxesParameters` container (the concrete subtype of `AbstractSurfaceFluxesParameters`).
- Surface flux computation is expensive (root-finding on the Monin–Obukhov length); call it once per stage in the infrastructure layer.
- Public entry points include `surface_fluxes` (the bulk solver), `sensible_heat_flux`, `latent_heat_flux`, and `compute_profile_value` (for recovering profile values at a given height). Round-trip tests against these validate flux changes. See [testing_and_validation.md §2 Round-trip tests](../infrastructure/testing_and_validation.md#2-round-trip-inverse-tests).

## ClimaParams.jl

Extract the specific parameter sub-struct (e.g. `thp = p.params.thermodynamics_params`) to a local variable before any `@.` broadcast. Capturing the full `p.params` container can push a large struct into GPU kernel parameter memory and exceed hardware limits. See [SDP 20](../code-quality/software_design_patterns.md) for the rule and a worked example.

## General cross-repo guidance

Treat every function whose name ends in `_deprecated` or that is annotated `@deprecate` as absent; use the replacement.

## Self-correction

If this guide is discovered to be stale or missing a pattern, update it.
