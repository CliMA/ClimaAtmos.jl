# Glossary

Short definitions of the symbols and terms that recur across the ClimaAtmos
source code and documentation. Each entry links to the page that covers it in
depth.

## State and cache

ClimaAtmos integrates a prognostic state vector forward in time. A handful of
single-letter names appear throughout the code:

| Symbol | Meaning |
|:--|:--|
| `Y`  | The **prognostic state vector** — the quantities integrated in time. `Y.c` holds cell-center variables (e.g. density `ρ`, total energy `ρe_tot`, horizontal velocity `uₕ`); `Y.f` holds face variables (e.g. vertical velocity `u₃`). |
| `Yₜ` | The **tendency vector**, i.e. the time derivative `∂Y/∂t`. `Yₜ.sfc` holds surface tendencies. |
| `p`  | The **cache**: parameters, precomputed fields (radiation fluxes, surface conditions, precipitation fluxes), model configuration, and slab/surface model properties. |
| `t`  | The **current simulation time**, in seconds from `start_date`. |

These are accessed through the integrator after a run, e.g.
`Y = simulation.integrator.u` (see [Your First Simulation](@ref)).

## Common terms

- **Setup** — the initial conditions, boundary conditions, and optional forcing
  that define a simulation *case* (BOMEX, DYCOMS, RICO, Held–Suarez, …). See [Setups](@ref).
- **Preset** — a one-line shortcut (`CA.Presets`) that bundles a grid, setup, and
  model choices for a common configuration.
- **Driver** — `.buildkite/ci_driver.jl`, the standard entry point that builds and
  runs a simulation from a configuration.
- **Integrator** — the time-stepping object that advances `Y`; its `.u` field is
  the current state vector.
- **IMEX time stepping** — the implicit–explicit scheme used by default; the
  implicit part is handled by the [Implicit Solver](@ref).
- **ITime** — the integer time type used for reproducible, exactly representable
  simulation times. See [ITime](@ref).
- **EDMF** — the Eddy-Diffusivity Mass-Flux turbulence/convection scheme, in
  [prognostic](@ref "Sub-grid scale equations") and
  [diagnostic](@ref "Diagnostic EDMF equations") forms.
- **Diagnostics** — derived output variables, as opposed to the prognostic state
  `Y`. See [Computing and saving diagnostics](@ref) and the catalog of
  [available diagnostic variables](@ref "Available diagnostic variables").
