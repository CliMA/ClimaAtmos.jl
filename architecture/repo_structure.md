# Navigating a CliMA repository

This file describes how to orient yourself in any CliMA Julia package. For the concrete directory tree of the repository you are in, see the repo-specific guide (linked from [AGENTS.md](../AGENTS.md)).

## Where to start

1. **`Project.toml`**: package name, version, deps, compat. Read first to confirm which package you are in and what it depends on.
2. **`src/<PackageName>.jl`**: the package entry point. It `include`s the major subsystems; follow the includes to find the owning source area for a feature.
3. **`README.md`** and **`docs/src/`**: the user-facing description of the package and its public API. Useful when the source layout is unfamiliar.
4. **`NEWS.md`**: recent user-visible changes. Always check this when working with a package whose public API you are unsure about.
5. **`test/runtests.jl`**: the test driver. The grouping it uses (often a `TEST_GROUP` environment variable) tells you which test buckets exist.

## Common top-level layout

Most CliMA Julia packages share these directories:

- `src/`: package source code, organized by physics or runtime concept.
- `test/`: unit and integration tests. `test/` typically mirrors `src/`.
- `docs/`: Documenter sources (`docs/make.jl`, `docs/src/`).
- `ext/`: Julia package extensions (loaded conditionally on dependencies).
- `.buildkite/`: Buildkite pipeline definitions and run drivers (where used).
- `.github/workflows/`: GitHub Actions CI.
- `perf/` or `benchmarks/`: performance/allocation benchmarks, run by hand or in a dedicated CI job (see [ecosystem_conventions.md §5](ecosystem_conventions.md)).
- `examples/`, `runscripts/`, `post_processing/`, `calibration/`: repo-dependent.

Alongside these, every repo root has `README.md`, `LICENSE`, and — in Apache-licensed repos — `NOTICE`. Apache 2.0 is the CliMA default; Oceananigans.jl and ClimaOcean.jl are MIT. See [documentation_policy.md §2.4](../code-quality/documentation_policy.md).

## Conventions to expect

- **Cache / precomputed quantities**: ClimaAtmos has a `src/cache/` area where per-stage scratch state update functions are defined. Scratch fields are allocated once at construction and never inside per-step setters (see [gpu_performance.md §3](../performance/gpu_performance.md)).
- **Parameter containers**: every package exposes a parameter container (often `*Parameters`) constructed from `ClimaParams.jl` TOML files.
- **Public API**: the symbols re-exported from `src/<PackageName>.jl` are the supported API. Internal functions and data structures may change without notice.
- **Test-only deps**: development tooling goes in `test/Project.toml` (or, in some smaller packages, the root `[extras]`/`[targets]` block), never in `[deps]`. See [dependency_management.md §1](dependency_management.md).

## How to locate a feature

1. Search `src/<PackageName>.jl` for the include that brings in the relevant subsystem.
2. Within that subsystem's directory, look for a file whose name matches the physics/runtime concept (`microphysics.jl`, `radiation.jl`, etc.).
3. Cross-reference with `test/`. The test path almost always mirrors the source path and offers a smaller entry point.
4. If still unsure, `git log --oneline -- <directory>` reveals recent activity and PR numbers; the PR descriptions are often the clearest design notes.

## When a repo deviates from these conventions

Repos like ClimaCore, Thermodynamics, and CloudMicrophysics have different source layouts than model repos. Read `src/<PackageName>.jl` and `docs/src/` first.

## Self-correction

If this guide is discovered to be stale or missing a pattern, update it.
