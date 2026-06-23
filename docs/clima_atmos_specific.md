# ClimaAtmos-Specific Guide

This file contains everything specific to the ClimaAtmos.jl repository: directory layout, configuration system, test groups, Buildkite jobs, and reproducibility tooling. All other files in `docs/agents/` are intended to be portable across CliMA repositories.

## Codebase map

  - `src/ClimaAtmos.jl`: package entry point. It `include`s the main subsystems; start here when you need the owning source area.
  - `src/config/`: YAML→typed-object translation layer (formerly `src/solver/`). `cli_options.jl` defines `--config_file` and `--job_id`; `yaml_helper.jl` loads `config/default_configs/default_config.yml` and merges overlay YAML; `atmos_config.jl` defines `AtmosConfig`; `model_getters.jl` and `type_getters.jl` translate config into a runnable model; `solve.jl` runs it. `parsed_args` reads are confined to this folder.
  - `src/simulation/AtmosSimulations.jl`: high-level `AtmosSimulation(config)` construction.
  - `src/cache/`: precomputed quantities allocated once per stage. Naming convention: `set_*_precomputed_quantities!(Y, p, t)` — never allocate inside these functions.
  - `src/prognostic_equations/`: tendency accumulation and implicit/explicit splitting.
  - `src/parameterized_tendencies/`: parameterization implementations.
      + `microphysics/`: microphysics tendency orchestration, SGS quadrature, limiters, Jacobian.
      + `radiation/`: RRTMGP wrappers and idealized radiation (`held_suarez.jl`).
      + `gravity_wave_drag/`: non-orographic and orographic GWD.
      + `les_sgs_models/`: Smagorinsky–Lilly, anisotropic minimum dissipation, constant horizontal diffusion.
      + `sponge/`: Rayleigh and viscous sponge tendencies.
  - EDMF code lives in `src/cache/{prognostic,diagnostic}_edmf_precomputed_quantities.jl` and `src/prognostic_equations/edmfx_*.jl`, not under `parameterized_tendencies/`.
  - `src/callbacks/`, `src/diagnostics/`, `src/setups/`, `src/surface_conditions/`, `src/topography/`, `src/parameters/`, `src/utils/`: remaining domain subtrees. Search by physics/runtime concept first.
  - `config/`: YAML/TOML config library. `default_configs/default_config.yml` is the schema baseline; `common_configs/` holds reusable numerics; `model_configs/`, `gpu_configs/`, `mpi_configs/`, `perf_configs/`, and `longrun_configs/` are scenario overlays.
  - `.buildkite/ci_driver.jl`: canonical run entry for CI-style simulations. It parses config, builds the simulation, runs `solve_atmos!`, and performs validation/output checks.
  - `.buildkite/pipeline.yml`: authoritative list of Buildkite jobs and their config combinations. Use it to see which config families are combined in automation.
  - `docs/make.jl` and `docs/src/`: Documenter entry point plus user/contributor docs. Good references for API usage and config recipes.
  - `perf/`: allocation and performance benchmarks run separately from unit tests. Not run in CI; regressions must be caught in review.
  - `reproducibility_tests/`: reproducibility test infrastructure. `ref_counter.jl` holds a single integer counter that partitions commit history into reference bins — increment it when simulation output intentionally changes.
  - `post_processing/`, `calibration/`, `runscripts/`, `examples/`: analysis scripts, calibration workflows, launch scripts, and smaller usage examples.

## Mapping the architectural layers to ClimaAtmos directories

The universal layer model in [architectural_boundaries.md](architectural_boundaries.md) maps to ClimaAtmos as follows:

  - **Parameterizations layer**: `src/parameterized_tendencies/`. Defines *how* a physical tendency is computed.
  - **Infrastructure layer**: `src/cache/`, `src/prognostic_equations/`, `src/config/`, `src/simulation/`. Defines *where* results are stored and *how* the model time-steps them.

A file under `src/parameterized_tendencies/` should not contain orchestration logic; orchestration belongs in `src/prognostic_equations/` or `src/cache/`.

## Configuration

  - Config files use `.yml` and usually encode scenario/family in the filename, e.g. `single_column_*`, `prognostic_edmfx_*`.
  - Config wiring is layered: default config loads first, then each repeated `--config_file` overlay is merged in order, with later files winning on conflicts.
  - Common pattern: one numerics file from `config/common_configs/` plus one scenario file from `config/model_configs/` or another config family.
  - YAML can point at parameter TOML via `toml: [toml/... ]`.
  - `job_id` must be unique across `config/`; this is enforced in `test/config.jl`.
  - If you add keys to `config/default_configs/default_config.yml`, keep the `help` and `value` wrapper.

## Test groups

`test/runtests.jl` groups tests by `TEST_GROUP`: `infrastructure`, `diagnostics`, `dynamics`, `parameterizations`, `restarts`, `era5`. Map your changes to the relevant group.

| Change area          | Test group          | Example Buildkite job         |
|:-------------------- |:------------------- |:----------------------------- |
| Prognostic equations | `dynamics`          | `sphere_baroclinic_wave_rhoe` |
| Microphysics / EDMF  | `parameterizations` | `prognostic_edmfx_*`          |
| Restarts             | `restarts`          | `restart_*`                   |
| Diagnostics          | `diagnostics`       | any `--diagnostics` job       |
| Config semantics     | `infrastructure`    | `config.jl`                   |

### Running a single test group

```bash
julia +1.11 --project=test -e '
  import Pkg; Pkg.test("ClimaAtmos"; test_args=["parameterizations"])
'
```

### Test layout

  - `test/config/`, `test/diagnostics/`, `test/prognostic_equations/`, `test/parameterized_tendencies/`, `test/conservation/` mostly mirror the source layout.
  - `test/test_helpers.jl`: shared testing utilities.
  - `test/config.jl`: config invariants and uniqueness checks; inspect this before changing config semantics.

## Validation surfaces specific to ClimaAtmos

When reviewing or writing changes, name the validation surface explicitly:

  - **`test/runtests.jl` test groups** for unit-level coverage.
  - **`.buildkite/ci_driver.jl` jobs** for config or runtime-workflow changes. Check `.buildkite/pipeline.yml` to identify the affected jobs.
  - **`reproducibility_tests/`** for changes that may shift simulation output. The reference counter in `reproducibility_tests/ref_counter.jl` must be incremented when output intentionally changes; do not edit it without explicit direction from the user.
  - **`perf/` allocation benchmarks** are not run in CI. Allocation regressions must be caught during review using the `@allocated` pattern.

## MSE / reproducibility

  - The reference counter lives in `reproducibility_tests/ref_counter.jl`. Increment it when intentionally changing simulation output. Never edit it without explicit user direction (see [agent_autonomy.md](agent_autonomy.md)).
  - Float32 simulations may have looser MSE thresholds. Document this explicitly when setting a new threshold.

## Local commands

  - Prefer Julia 1.11.x for local work. CI also runs 1.10 and 1.11.
  - For runtime validation, prefer `julia +1.11 --project=.buildkite .buildkite/ci_driver.jl ...`.
  - For package tests, prefer `Pkg.test()` over manually `include`ing `test/runtests.jl` because test-only deps are loaded through the package test path.

## Self-correction

If this guide is discovered to be stale or missing a pattern, update it.
