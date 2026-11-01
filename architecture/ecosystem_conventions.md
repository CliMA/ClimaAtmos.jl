# CliMA Ecosystem Conventions

Conventions for reading and writing code that span every CliMA model and library repo. This guide is a *glossary first, rule book second*: it collects what looks obvious to a long-time contributor and trips up newcomers (human and AI alike).

## 1. Module aliases used across the ecosystem

Prefer the functions and types a package `export`s, reached by `using` the package or `import`-ing specific symbols. Where a package's API is not yet finalized, model repos reach unexported quantities through an alias: `import Thermodynamics as TD` makes any quantity in `Thermodynamics.jl` available as `TD.<name>`. Such quantities can change without a breaking release; see [architectural_boundaries.md §3](architectural_boundaries.md).

Match these aliases in new code so call sites stay grep-able.

| Alias   | Package / module                               | Where it dominates                      |
|:--------|:-----------------------------------------------|:----------------------------------------|
| `TD`    | `Thermodynamics`                               | every model repo                        |
| `TDP`   | `Thermodynamics.Parameters`                    | ClimaAtmos, ClimaLand                   |
| `SF`    | `SurfaceFluxes`                                | ClimaAtmos, ClimaCoupler                |
| `SFP`   | `SurfaceFluxes.Parameters`                     | ClimaAtmos                              |
| `UF`    | `SurfaceFluxes.UniversalFunctions`             | ClimaAtmos                              |
| `CM`    | `CloudMicrophysics`                            | ClimaAtmos                              |
| `CM0` / `CM1` / `CM2` | `CloudMicrophysics.Microphysics{0,1,2}M` | ClimaAtmos microphysics                 |
| `CMNe`  | `CloudMicrophysics.MicrophysicsNonEq`          | ClimaAtmos non-equilibrium microphysics |
| `BMT`   | `CloudMicrophysics.BulkMicrophysicsTendencies` | ClimaAtmos bulk microphysics wrapper    |
| `CMP`   | `CloudMicrophysics.Parameters`                 | ClimaAtmos                              |
| `CC`    | `ClimaCore`                                    | parameter / utility code                |
| `CAP`   | `ClimaAtmos.Parameters`                        | ClimaAtmos internal                     |
| `CAD`   | `ClimaAtmos.Diagnostics`                       | ClimaAtmos diagnostics                  |
| `CP`    | `ClimaParams`                                  | any repo that reads TOML parameters     |
| `IP`    | `Insolation.Parameters`                        | ClimaAtmos parameter wiring             |
| `SA`    | `StaticArrays`                                 | hot-path / kernel code                  |
| `RS`    | `RootSolvers`                                  | Thermodynamics / CloudMicrophysics      |
| `RRTMGPI` | `ClimaAtmos.RRTMGPInterface`                 | ClimaAtmos radiation                    |

In docstrings, use the same prefix you would use in code (`TD.air_temperature`, not `Thermodynamics.air_temperature`). Readers grep on the prefix.

## 2. Prognostic state, tendencies, and the cache

Model repos (ClimaAtmos, ClimaLand) follow a common state layout:

- **`Y`**: the prognostic state vector. A `ClimaCore.Fields.FieldVector` whose top-level fields name solution regions (`Y.c` for cell-centered, `Y.f` for face-centered in ClimaAtmos; `Y.soil`, `Y.canopy` in ClimaLand). Anything timestepped lives in `Y`.
- **`Yₜ`**: the tendency, same shape as `Y`. Tendency functions write into `Yₜ` and never allocate new fields; they read from `Y`.
- **`p`**: the cache *bundle* passed to every right-hand-side function. In ClimaAtmos, this contains pre-allocated scratch (`p.scratch`), precomputed quantities (`p.precomputed`), atmosphere settings (`p.atmos`), and the numeric parameter struct (`p.params`, e.g. `ClimaAtmosParameters`). In ClimaLand, the cache contains scratch space and physical quantities computed once or several times per step; parameters are passed in via the model, not the cache.
- **`t`**: the current simulation time (seconds as a `Float64`, or an `ITime`).

Rules implied by this layout:

1. **Never allocate `Field`s inside a tendency or cache setter.** Use a scratch field from `p.scratch` (allocated during model construction in `src/cache/` for ClimaAtmos), or use lazy broadcasting. See the "Materialization" section of [GPU Performance Guide §3](../performance/gpu_performance.md).
2. **`Yₜ` should be treated as write-only inside tendency functions.** Reading `Yₜ` back couples stages of the time integrator and can break reproducibility. The one accepted exception is *post-hoc limiters* (e.g., in ClimaLand) that clip the assembled `Yₜ` after all contributions have been written. They are a non-linear projection applied at the end of the right-hand side and do not couple stages.
3. **`p` must be treated as immutable from the integrator's point of view.** You can write to `p.precomputed` and `p.scratch` *as part of refreshing the cache for the current stage*, but you must not mutate `p` in ways that would result in different behavior on a subsequent call with the same values of `Y` and `t`.

## 3. Cell-center vs cell-face notation (`ᶜ` / `ᶠ`)

ClimaCore-based repos use the Unicode prefixes `ᶜ` (cell-center, U+1D9C, typed `\^c<TAB>`) and `ᶠ` (cell-face, U+1DA0, typed `\^f<TAB>`) to mark the staggered-grid location of a field. The prefix is part of the name, whether the name is a variable holding a field or a function returning one: `ᶜρ` and `ᶠρ` are different fields living on different `ClimaCore.Spaces.AbstractSpace`s. Operators follow the same convention: the prefix on `ᶜgradᵥ`, `ᶠinterp`, etc. names the space of the *result*. Scalars and pointwise values have no prefix.

When introducing a new field, copy the prefix from an analogous existing one. See also [variable_list.md "Field Prefixes"](../code-quality/variable_list.md).

## 4. Parameter wiring (`ClimaParams` → physics libraries → model)

Every CliMA package follows the same pattern for physical constants:

```text
TOML files in ClimaParams/src/parameters.toml
        │
        ▼ CP.create_toml_dict(FT)
toml_dict :: CP.ParamDict
        │
        │ Constructor: ThermodynamicsParameters(toml_dict)
        │              SurfaceFluxesParameters(toml_dict, UF.GryanikParams)
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
- **`FT` is fixed by `CP.float_type(toml_dict)`.** Always derive `FT` from the parameter container or the live input.

## 5. CI and Buildkite

Most CliMA model repos run both GitHub Actions (formatter, unit tests, docs) and Buildkite (GPU-backed integration and reproducibility runs).

- **GitHub Actions** lives in `.github/workflows/`. The common workflows are `JuliaFormatter.yml` (or `julia_formatter.yml`), a unit-test workflow (`ci.yml` or `UnitTests.yml`), `Documentation.yml`, and `Downstream.yml`.
- **Buildkite** lives in `.buildkite/` with a top-level `pipeline.yml` and a runner script (e.g. `ci_driver.jl` in ClimaAtmos). Jobs are keyed by `job_id` strings that also drive output paths and reproducibility tests.
- **Allocation benchmarks** (typically in `perf/`) run by hand or in a dedicated Buildkite job, outside the unit-test CI. Allocation regressions must be caught at review time via the `@allocated == 0` pattern in [allocation_debugging.md §1](../performance/allocation_debugging.md).

When a CI job is mentioned by name in a guide or review, it is almost always a Buildkite job name; resolve it by searching `.buildkite/pipeline.yml`.

## 6. Reproducibility tests (model repos)

Some model repos (e.g. ClimaAtmos) maintain a `reproducibility_tests/` directory that pins the simulation output of canonical jobs to within bit tolerance, keyed off `reproducibility_tests/ref_counter.jl`. A code change that flips a reproducibility test is itself a finding: report it in the PR description with the job name and the MSE diff.

Changing reference counters or MSE tolerance files requires explicit user instruction. See [agent_autonomy.md](../workflow/agent_autonomy.md) for the full rule.

## 7. Diagnostics

`Diagnostics` is the user-facing name for any quantity written to output (NetCDF, HDF5) at a configurable cadence. Diagnostic *names* and *units* are public API.

- In ClimaAtmos and ClimaLand, diagnostics are defined in `src/diagnostics/` (aliased as `CAD` for ClimaAtmos). Each diagnostic has a short name, long name, units, comments, and a compute function.
- Renaming a diagnostic, changing its units, or removing a default diagnostic is a breaking change; adding a diagnostic is a feature. Both need a `NEWS.md` entry; see [changelogs_and_versions.md §1](../code-quality/changelogs_and_versions.md) for the entry format and badges.

## 8. Self-correction

If this guide is discovered to be stale or missing a pattern, update it.
