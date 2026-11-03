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

- Prefer Julia 1.11.x for local work. GitHub CI runs every test group on 1.11 and only `infrastructure` on 1.10.
- For runtime validation, prefer `julia +1.11 --project=.buildkite .buildkite/ci_driver.jl ...`.
- For package tests, prefer `Pkg.test()` over manually `include`ing `test/runtests.jl` because test-only deps are loaded through the package test path.
- Keep edits inside the owning subtree when possible; use [src/ClimaAtmos.jl](src/ClimaAtmos.jl) to trace where a feature is wired.
- Match existing style: explicit names, narrow imports, comments that explain why.
- Comments describe the code as it is. Never document previously existing code or what
  a change fixed ("used to", "previously", "no longer"); that history belongs in the
  commit message, and in `NEWS.md` when behavior changes.
- Follow the software design patterns in [docs/dev-guides/architecture/software_design_patterns.md](docs/dev-guides/architecture/software_design_patterns.md) for new code and refactor toward them when touching existing code.
- Format code before committing. CI checks formatting via the `prek` hook in [.github/workflows/run-prek.yml](.github/workflows/run-prek.yml), which runs JuliaFormatter from the version-pinned [.dev/format/Project.toml](.dev/format/Project.toml) environment (currently `=2.10.1`). Match CI with either `prek run julia-formatter --all-files` or the pinned env directly: `julia --startup-file=no --project=.dev/format -e 'using Pkg; Pkg.instantiate(io=devnull); using JuliaFormatter; format(ARGS)' .`. Avoid `julia -e 'using JuliaFormatter; format(\".\")'` from your global environment — `Pkg.add("JuliaFormatter")` installs v2 by default and a mismatched version produces a different diff.
- Build the docs with `julia +1.11 --project=docs docs/make.jl`. `docs/Manifest.toml`
  is gitignored, so a local copy goes stale as dependencies move; refresh it the
  way [docs.yml](.github/workflows/docs.yml) does — `Pkg.develop(PackageSpec(path="."))`
  then `Pkg.instantiate()` in the `docs` environment — before concluding that a
  build failure is real. A stale manifest shows up as an `UndefVarError` for a
  ClimaCore symbol during precompilation, not as a documentation error.
- `@.` dots every call in its expression, including accessors such as
  `Fields.level` and `MatrixFields.get_field`, and some of them return their
  argument unchanged when applied to a number (`Fields.level(x, 1)` on a scalar
  is `x`), so a broadcast over them silently reads the wrong values (see
  #4952). Indexing is not dotted: `x[1]` inside `@.` indexes `x` itself, even
  when `x` is a lazy broadcast. Bind accessed fields to variables before the
  `@.`, and check fused or lazy rewrites against states that vary along every
  dimension, since a vertically uniform initial state hides this error.
- Do not end closures or `do` blocks with `return nothing` when their return
  value is ignored (as in `foreach_column` loop bodies and `unrolled_foreach`):
  the extra line affects neither latency nor runtime.
- Performance work is measured against the AMIP target configuration
  (`numerics_sphere_he16ze63.yml` + `amip_target.yml`), but it is not limited
  to code that configuration runs: measure such code with the flags that
  enable it (defaults are in `config/default_configs/default_config.yml`).
- Optional but recommended: install the pre-commit hooks in [.pre-commit-config.yaml](.pre-commit-config.yaml) (`uv tool install prek && prek install`) to auto-format and trim trailing whitespace on commit. See [docs/src/contributor_guide.md](docs/src/contributor_guide.md) ("Pre-commit hooks").

## Self-correction

- If the code map in [docs/clima_atmos_specific.md](docs/clima_atmos_specific.md) is discovered to be stale, update it.
- If the user gives a correction about how work should be done in this repo, add it to `Local norms` or another clearly labeled persistent section in this file or in [docs/clima_atmos_specific.md](docs/clima_atmos_specific.md) so future sessions inherit it.
