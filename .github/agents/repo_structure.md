# Codebase map

- `src/ClimaAtmos.jl`: package entry point. It `include`s the main subsystems; start here when you need the owning source area.
- `src/solver/`: configuration and CLI wiring. `cli_options.jl` defines `--config_file` and `--job_id`; `yaml_helper.jl` loads `config/default_configs/default_config.yml` and merges overlay YAML; `types.jl`, `model_getters.jl`, `type_getters.jl`, and `solve.jl` turn config into a runnable model.
- `src/simulation/AtmosSimulations.jl`: high-level `AtmosSimulation(config)` construction.
- `src/prognostic_equations/`, `src/parameterized_tendencies/`, `src/callbacks/`, `src/diagnostics/`, `src/setups/`, `src/surface_conditions/`, `src/topography/`, `src/cache/`, `src/parameters/`, `src/utils/`: domain subtrees. Search by physics/runtime concept first.
- `config/`: YAML/TOML config library. `default_configs/default_config.yml` is the schema baseline; `common_configs/` holds reusable numerics; `model_configs/`, `gpu_configs/`, `mpi_configs/`, `perf_configs/`, and `longrun_configs/` are scenario overlays.
- `.buildkite/ci_driver.jl`: canonical run entry for CI-style simulations. It parses config, builds the simulation, runs `solve_atmos!`, and performs validation/output checks.
- `.buildkite/pipeline.yml`: canonical config composition examples. Use it to see which config families are combined in automation.
- `docs/make.jl` and `docs/src/`: Documenter entry point plus user/contributor docs. Good references for API usage and config recipes.
- `perf/`, `post_processing/`, `calibration/`, `reproducibility_tests/`, `runscripts/`, `examples/`: performance tools, analysis scripts, calibration workflows, reproducibility helpers, launch scripts, and smaller usage examples.

## Configuration

- Config files use `.yml` and usually encode scenario/family in the filename, e.g. `single_column_*`, `diagnostic_edmfx_*`, `prognostic_edmfx_*`.
- Config wiring is layered: default config loads first, then each repeated `--config_file` overlay is merged in order, with later files winning on conflicts.
- Common pattern: one numerics file from `config/common_configs/` plus one scenario file from `config/model_configs/` or another config family.
- YAML can point at parameter TOML via `toml: [toml/... ]`.
- `job_id` must be unique across `config/`; this is enforced in `test/config.jl`.
- If you add keys to `config/default_configs/default_config.yml`, keep the `help` and `value` wrapper.

## Test locations

- `test/runtests.jl`: top-level test driver. Tests are grouped by `TEST_GROUP`: `infrastructure`, `diagnostics`, `dynamics`, `parameterizations`, `restarts`, `era5`.
- `test/solver/`, `test/diagnostics/`, `test/prognostic_equations/`, `test/parameterized_tendencies/`, `test/conservation/`: tests mostly mirror the source layout.
- `test/test_helpers.jl`: shared testing utilities.
- `test/config.jl`: config invariants and uniqueness checks; inspect this before changing config semantics.

## Self-correction

- If this code map is discovered to be stale, update it.
