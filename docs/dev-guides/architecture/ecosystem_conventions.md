# CliMA Ecosystem Conventions

Conventions for reading and writing code that span every CliMA model and library repo. This guide is a *glossary first, rule book second*: things that look obvious to a long-time contributor but trip up newcomers (human and AI alike).

## 1. Module aliases used across the ecosystem

When working in a CliMA model repo you will see these aliases repeatedly. Match them in new code so call sites stay grep-able. Do **not** invent a different alias for the same package.

| Alias | Package / module                          | Where it dominates |
|:------|:------------------------------------------|:-------------------|
| `TD`  | `Thermodynamics`                          | every model repo |
| `TDP` | `Thermodynamics.Parameters`               | ClimaAtmos, ClimaLand |
| `SF`  | `SurfaceFluxes`                           | ClimaAtmos, ClimaCoupler |
| `SFP` | `SurfaceFluxes.Parameters`                | ClimaAtmos |
| `UF`  | `SurfaceFluxes.UniversalFunctions`        | ClimaAtmos |
| `CM`  | `CloudMicrophysics`                       | ClimaAtmos |
| `CM0` / `CM1` / `CM2` | `CloudMicrophysics.Microphysics{0,1,2}M` | ClimaAtmos microphysics |
| `CMNe`| `CloudMicrophysics.MicrophysicsNonEq`     | ClimaAtmos non-equilibrium microphysics |
| `BMT` | `CloudMicrophysics.BulkMicrophysicsTendencies` | ClimaAtmos bulk microphysics wrapper |
| `CMP` | `CloudMicrophysics.Parameters`            | ClimaAtmos |
| `CC`  | `ClimaCore`                               | parameter / utility code |
| `CAP` | `ClimaAtmos.Parameters`                   | ClimaAtmos internal |
| `CAD` | `ClimaAtmos.Diagnostics`                  | ClimaAtmos diagnostics |
| `CP`  | `ClimaParams`                             | any repo that reads TOML parameters |
| `IP`  | `Insolation.Parameters`                   | ClimaAtmos parameter wiring |
| `SA`  | `StaticArrays`                            | hot-path / kernel code |
| `RS`  | `RootSolvers`                             | Thermodynamics / CloudMicrophysics |
| `RRTMGPI` | `ClimaAtmos.RRTMGPInterface`          | ClimaAtmos radiation |

In docstrings, use the same prefix you would use in code (`TD.air_temperature`, not `Thermodynamics.air_temperature`) — readers grep on the prefix.

## 2. Prognostic state, tendencies, and the cache

Model repos (ClimaAtmos, ClimaLand) follow a common state layout:

- **`Y`** — the prognostic state vector. A `ClimaCore.Fields.FieldVector` whose top-level fields name solution regions (`Y.c` for cell-centered, `Y.f` for face-centered in ClimaAtmos; `Y.soil`, `Y.canopy` in ClimaLand). Anything timestepped lives in `Y`.
- **`Yₜ`** — the tendency, same shape as `Y`. Tendency functions write into `Yₜ` and never allocate new fields; they read from `Y`.
- **`p`** — the parameter and cache *bundle* passed to every right-hand-side function. It is not just parameters: it carries pre-allocated scratch (`p.scratch`), precomputed quantities (`p.precomputed`), atmosphere settings (`p.atmos`), and the numeric parameter struct (`p.params`, e.g. `ClimaAtmosParameters`).
- **`t`** — the current simulation time (in seconds, `FT`).

Rules implied by this layout:

1. **Never allocate `Field`s inside a tendency or precomputed-quantity setter.** Every scratch field must already exist in `p.scratch` (allocated once during model construction in `src/cache/`). See the "Materialization" section of [GPU Performance Guide §3](../performance/gpu_performance.md).
2. **`Yₜ` may only be written into, not read from.** Reading `Yₜ` inside a tendency function couples stages of the time integrator and breaks reproducibility.
3. **`p` must be treated as effectively immutable from the integrator's point of view.** You can write to `p.precomputed` and `p.scratch` *as part of refreshing the cache for the current stage*, but you must not mutate `p` in ways that change behavior on a subsequent call with the same `(Y, t)`.

## 3. Cell-center vs cell-face notation (`ᶜ` / `ᶠ`)

ClimaAtmos uses two Unicode prefixes to distinguish where a field lives on the staggered vertical grid:

- `ᶜ` (U+1D9C) — cell-center field. Typed in most editors as a Unicode escape or with `\^c<TAB>`.
- `ᶠ` (U+1DA0) — cell-face field. Typed as `\^f<TAB>`.

Examples: `ᶜρ`, `ᶠu³`, `ᶜT`, `ᶠw`. These prefixes are part of the variable name, not decoration — `ᶜρ` and `ᶠρ` are different fields living in different `ClimaCore.Spaces.AbstractSpace`s. Operators (`ᶜgradᵥ`, `ᶠinterp`, etc.) follow the same convention: the prefix tells you which space the *result* lives in.

When introducing a new field, copy the level prefix from an analogous existing field. Mismatched prefixes are one of the more common causes of `BroadcastInferenceError` on ClimaCore field operations.

## 4. Parameter wiring (`ClimaParams` → physics libraries → model)

Every CliMA package follows the same pattern for physical constants:

```
TOML files in ClimaParams/src/parameters.toml
        │
        ▼ CP.create_toml_dict(FT)
toml_dict :: CP.ParamDict
        │
        ▼ Constructor: ThermodynamicsParameters(toml_dict)
        ▼              SurfaceFluxesParameters(toml_dict, UF.GryanikParams)
        ▼              CMP.TerminalVelocityParams(toml_dict)
Library-specific parameter struct (immutable, isbits-after-adapt)
        │
        ▼ Bundled by the model
ClimaAtmosParameters / ClimaLandParameters / ...
        │
        ▼ Passed at every RHS call
        p.params
```

Implications:

- **Do not hard-code physical constants** anywhere downstream. Add them to ClimaParams (or, if scheme-specific, to the relevant library's TOML), then read them through the parameter struct.
- **Constructors take a `CP.ParamDict`, not individual values.** When you add a new parameter, register a name-map entry in the library's parameter constructor.
- **`FT` is fixed by `CP.float_type(toml_dict)`.** Always derive `FT` from the parameter container or the live input, not by hard-coding.

## 5. CI and Buildkite

Most CliMA model repos run both GitHub Actions (formatter, unit tests, docs) and Buildkite (GPU-backed integration and reproducibility runs).

- **GitHub Actions** lives in `.github/workflows/`. The common workflows are `JuliaFormatter.yml` (or `julia_formatter.yml`), `UnitTests.yml`, `Documentation.yml`, and `Downstream.yml`.
- **Buildkite** lives in `.buildkite/` with a top-level `pipeline.yml` and a runner script (e.g. `ci_driver.jl` in ClimaAtmos). Jobs are keyed by `job_id` strings that also drive output paths and reproducibility tests.
- **Allocation benchmarks** (typically in `perf/`) are *not* run in CI. Allocation regressions must be caught at review time via the `@allocated` pattern in [GPU Performance Guide §9](../performance/gpu_performance.md).

When a CI job is mentioned by name in a guide or review, it is almost always a Buildkite job name; resolve it by searching `.buildkite/pipeline.yml`.

## 6. Reproducibility tests (model repos)

Model repos (notably ClimaAtmos) maintain a `reproducibility_tests/` directory that pins the simulation output of a fixed set of canonical jobs to within bit tolerance. The reference output is keyed off a numeric counter (`reproducibility_tests/ref_counter.jl`).

Rules for agents:

- **Do not change `ref_counter.jl`, MSE tolerance files, or recorded reference values without an explicit user instruction.** Bumping the counter is how reproducibility-breaking changes are *deliberately* shipped, and it requires human review. See [agent_autonomy.md](../workflow/agent_autonomy.md).
- A code change that flips a reproducibility test is itself a finding: report it in the PR description with the job name and the diff in MSE, not as a hidden side effect.

## 7. Diagnostics

`Diagnostics` is the user-facing name for any quantity written to output (NetCDF, HDF5) at a configurable cadence. Diagnostic *names* and *units* are public API.

- In ClimaAtmos, diagnostics are wired up in `src/diagnostics/` (aliased as `CAD`). Each diagnostic has a short name, long name, units, comments, and a compute function.
- Renaming a diagnostic, changing its units, or removing a default diagnostic is a breaking change and requires a `NEWS.md` entry under `![][badge-💥breaking]` (see [changelogs_and_versions.md](../code-quality/changelogs_and_versions.md)).
- Adding a *new* diagnostic does not require a breaking-change badge, but does need a `NEWS.md` entry under `![][badge-✨feature/enhancement]`.

## 8. Self-correction

If this guide is discovered to be stale or missing a pattern, update it.
