# ClimaAtmos Agent Guide

## Quick start by task type

Always read first:

1. [docs/agents/repo_structure.md](docs/agents/repo_structure.md) — how to navigate any CliMA repo.
2. [docs/agents/clima_atmos_specific.md](docs/agents/clima_atmos_specific.md) — directory tree, test groups, and reproducibility specifics for *this* repo.
3. [docs/agents/agent_autonomy.md](docs/agents/agent_autonomy.md) — actions that require explicit user approval.

When modifying source code in `src/`, also read:

- [docs/agents/software_design_patterns.md](docs/agents/software_design_patterns.md) — the rule set for new and changed code.
- [docs/agents/gpu_performance.md](docs/agents/gpu_performance.md) — required for any hot-path code.
- [docs/agents/type_stability.md](docs/agents/type_stability.md) — required for any new function.
- [docs/agents/ad_compatibility.md](docs/agents/ad_compatibility.md) — required when touching physics or tendency functions.

## Specialized guides (load on demand)

- **PR reviews** → [review.md](docs/agents/review.md)
- **Numerical robustness** → [numerical_robustness.md](docs/agents/numerical_robustness.md)
- **Architecture / layers** → [architectural_boundaries.md](docs/agents/architectural_boundaries.md)
- **Cross-repo calls** → [cross_repo_contracts.md](docs/agents/cross_repo_contracts.md)
- **Device-agnostic & MPI** → [clima_comms.md](docs/agents/clima_comms.md)
- **Testing** → [testing_and_validation.md](docs/agents/testing_and_validation.md)
- **NEWS / changelog** → [changelog_hygiene.md](docs/agents/changelog_hygiene.md)
- **Dependencies** → [dependency_management.md](docs/agents/dependency_management.md)
- **Code style** → [code_style.md](docs/agents/code_style.md)
- **Docstrings** → [docstring_standard.md](docs/agents/docstring_standard.md)
- **Repo-specific** → [clima_atmos_specific.md](docs/agents/clima_atmos_specific.md)

## Local norms

- Prefer Julia 1.11.x for local work. CI also runs 1.10 and 1.11.
- For runtime validation, prefer `julia +1.11 --project=.buildkite .buildkite/ci_driver.jl ...`.
- For package tests, prefer `Pkg.test()` over manually `include`ing `test/runtests.jl` because test-only deps are loaded through the package test path.
- Keep edits inside the owning subtree when possible; use [src/ClimaAtmos.jl](src/ClimaAtmos.jl) to trace where a feature is wired.
- Match existing style: explicit names, narrow imports, comments that explain why.
- Follow the software design patterns in [docs/agents/software_design_patterns.md](docs/agents/software_design_patterns.md) for new code and refactor toward them when touching existing code.
- Run `julia -e 'using JuliaFormatter; format(".")'` before committing code.

## Self-correction

- If the code map in [docs/agents/clima_atmos_specific.md](docs/agents/clima_atmos_specific.md) is discovered to be stale, update it.
- If the user gives a correction about how work should be done in this repo, add it to `Local norms` or another clearly labeled persistent section in this file or in the linked files in `docs/agents/` so future sessions inherit it.
