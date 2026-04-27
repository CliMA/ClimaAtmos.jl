# ClimaAtmos Agent Guide

## Repo exploration

- Before broad repo search or codebase exploration, read [docs/agents/repo_structure.md](docs/agents/repo_structure.md).

## Pull request reviews

- At the start of each review, tell the user that you have read [docs/agents/review.md](docs/agents/review.md) and will apply it to the review.
- When asked to review a pull request, follow [docs/agents/review.md](docs/agents/review.md).
- Apply the full checklist and output schema in [docs/agents/review.md](docs/agents/review.md); keep reviews concise, evidence-based, and findings-first.
- During review, also enforce [docs/agents/software_design_patterns.md](docs/agents/software_design_patterns.md) for changed code and adjacent affected code, as required by [docs/agents/review.md](docs/agents/review.md).

## Local norms

- Prefer Julia 1.11.x for local work. CI also runs 1.10 and 1.11.
- For runtime validation, prefer `julia +1.11 --project=.buildkite .buildkite/ci_driver.jl ...`.
- For package tests, prefer `Pkg.test()` over manually `include`ing `test/runtests.jl` because test-only deps are loaded through the package test path.
- Keep edits inside the owning subtree when possible; use [src/ClimaAtmos.jl](src/ClimaAtmos.jl) to trace where a feature is wired.
- Match existing style: explicit names, narrow imports, comments that explain why.
- Follow the software design patterns in [docs/agents/software_design_patterns.md](docs/agents/software_design_patterns.md) for new code and refactor toward them when touching existing code.

## Self-correction

- If the code map in [docs/agents/repo_structure.md](docs/agents/repo_structure.md) is discovered to be stale, update it.
- If the user gives a correction about how work should be done in this repo, add it to `Local norms` or another clearly labeled persistent section in this file so future sessions inherit it.
