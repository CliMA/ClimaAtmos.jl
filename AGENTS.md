# ClimaAtmos Agent Guide

## Ecosystem Guidelines

Please refer to the shared CliMA agent index for ecosystem-wide rules regarding architecture, performance, code quality, infrastructure, and workflows:

- [docs/dev-guides/AGENTS.md](docs/dev-guides/AGENTS.md) — Shared CliMA agent guidelines.

> Shared guides live at `docs/dev-guides/` and are vendored from the canonical source:
> https://github.com/CliMA/DeveloperGuides. Edit shared guides there, not here.

## Repo-Specific Guidelines

Always read the ClimaAtmos-specific guide before working in this repository:

- [docs/clima_atmos_specific.md](docs/clima_atmos_specific.md) — directory tree, test groups, and reproducibility specifics for *this* repo.

## Local norms

- Prefer Julia 1.11.x for local work. CI also runs 1.10 and 1.11.
- For runtime validation, prefer `julia +1.11 --project=.buildkite .buildkite/ci_driver.jl ...`.
- For package tests, prefer `Pkg.test()` over manually `include`ing `test/runtests.jl` because test-only deps are loaded through the package test path.
- Keep edits inside the owning subtree when possible; use [src/ClimaAtmos.jl](src/ClimaAtmos.jl) to trace where a feature is wired.
- Match existing style: explicit names, narrow imports, comments that explain why.
- Follow the software design patterns in [docs/dev-guides/architecture/software_design_patterns.md](docs/dev-guides/architecture/software_design_patterns.md) for new code and refactor toward them when touching existing code.
- Run `julia -e 'using JuliaFormatter; format(\".\")'` before committing code. The CI formatter check pins JuliaFormatter to the major version in [.github/workflows/julia_formatter.yml](.github/workflows/julia_formatter.yml) (`version:`); the matching pin lives in [.dev/format/Project.toml](.dev/format/Project.toml). `Pkg.add("JuliaFormatter")` now installs v2 by default, which produces a different diff — use the pinned env (or the `prek` hooks below) to match CI.
- Optional but recommended: install the pre-commit hooks in [.pre-commit-config.yaml](.pre-commit-config.yaml) (`uv tool install prek && prek install`) to auto-format and trim trailing whitespace on commit. See [docs/src/contributor_guide.md](docs/src/contributor_guide.md) ("Pre-commit hooks").

## Self-correction

- If the code map in [docs/clima_atmos_specific.md](docs/clima_atmos_specific.md) is discovered to be stale, update it.
- If the user gives a correction about how work should be done in this repo, add it to `Local norms` or another clearly labeled persistent section in this file or in [docs/clima_atmos_specific.md](docs/clima_atmos_specific.md) so future sessions inherit it.
