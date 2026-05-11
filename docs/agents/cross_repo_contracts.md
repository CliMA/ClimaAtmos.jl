# Cross-Repo Contracts

This guide documents the conventions for calling ecosystem packages from CliMA model repositories. Rules are at the call-site level; internal package APIs are not documented here.

## General principle

Always pass the package's *parameter container* (for example, `thermodynamics_params`, `surface_flux_params`) into physics functions rather than individual constants. This ensures consistency across model components and makes calibration transparent.

## How to find the current API of a CliMA dependency

You typically will not have the dependency's source checked out next to the model repo. Use this order:

1. **`NEWS.md`** of the dependency, if accessible — it lists API changes per release.
2. **The dev'd path under `~/.julia/dev/<Package>.jl`**, if the user has the package dev'd locally.
3. **The package's `docs/src/`**, which usually documents the supported call surface.
4. **Existing call sites in this repo** — `grep -rn "TD\." src/` is faster and more reliable than guessing.

Treat anything not in the package's `docs/` as internal and unstable.

## Thermodynamics.jl

- Pass `thermo_params` from the model parameter store; do not hard-code thermodynamic constants.
- Use functional constructors from the current public API (for example, `TD.air_temperature`). Confirm the call site against the sources listed above before writing new code.
- For iterative phase-equilibrium calculations inside GPU kernels, prefer variants that accept a fixed iteration count to avoid thread divergence. See [SDP 19](software_design_patterns.md).

## CloudMicrophysics.jl

- The microphysics scheme is passed as a singleton type; dispatch on it eliminates dead branches at compile time.
- Return values are `NamedTuple`s. Materialize them into a pre-allocated `NamedTuple`-of-`Field`s scratch slot in the cache and then issue one `@.` per target field — see the "Materialization" and "Multi-field updates" subsections in [GPU Performance Guide §3](gpu_performance.md).
- Float32 pollution: verify that integer-base exponentiation and float literals inside CloudMicrophysics functions are written with the element type (for example, `FT(6)^x`, not `6^x`).

## SurfaceFluxes.jl

- Pass a fully-typed `SurfaceFluxes.Parameters` container; do not hard-code flux constants.
- Surface flux computation is expensive; call it once per stage in the infrastructure layer, not inside tendency hot paths.

## ClimaParams.jl

- Parameters used inside `@.` broadcasts should be extracted to a named local variable before the broadcast ([SDP 20](software_design_patterns.md)) to keep broadcast shapes unambiguous.

## General cross-repo guidance

- Before writing a new call site, check the target package's `NEWS.md` for recent API changes.
- Treat every function whose name ends in `_deprecated` or that is annotated `@deprecate` as absent; use the replacement.

## Self-correction

If this guide is discovered to be stale or missing a pattern, update it.
