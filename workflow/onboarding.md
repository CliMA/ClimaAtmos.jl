# Onboarding to a CliMA Repository

A short walk from a fresh machine to a productive REPL session inside any CliMA Julia package. This guide is generic; repo-specific details are in the package's own `*_specific.md` guide.

## 1. Install the core tools

1. **Julia.** Install via [`juliaup`](https://julialang.org/downloads/). `juliaup add release` installs the current stable channel; CliMA repos test on the current LTS and one or two newer point releases. Check the unit-test workflow under `.github/workflows/` (`ci.yml` in most repos, `UnitTests.yml` in ClimaCore) for the exact matrix.
2. **Git** and a GitHub account with an SSH key (see GitHub's [SSH setup guide](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)).
3. *(Optional but recommended)* a Julia-aware editor: VS Code with the Julia extension, Helix, or Emacs/Vim with `julia-mode`/`julia-vim`.

## 2. Clone and instantiate

```bash
git clone git@github.com:CliMA/<repo-name>.jl.git
cd <repo-name>.jl
julia --project
```

Inside the REPL:

```julia
using Pkg
Pkg.instantiate()    # resolve and download every dependency
Pkg.status()         # check the resolved versions
import <RepoName>    # confirm the package loads
```

If `Pkg.instantiate()` fails, see §8 for the recovery sequence.

## 3. Keep a long-running REPL

Julia's first-call latency comes from compilation, and restarting the REPL throws that work away. Two packages let you iterate without restarting:

- **[Revise.jl](https://github.com/timholy/Revise.jl)**: watches package source files and patches updated method definitions into the running session.
- **[Infiltrator.jl](https://github.com/JuliaDebug/Infiltrator.jl)**: drops you into an interactive REPL at a `@infiltrate` breakpoint without instrumenting the function, so it does not invalidate compiled code.

Install both into your *base* (`v1.x`) environment so every REPL gets them automatically:

```julia
julia -e 'using Pkg; Pkg.add(["Revise", "Infiltrator"])'
```

Then add a startup file at `~/.julia/config/startup.jl`:

```julia
using Revise
using Infiltrator
```

The working loop is then: start the REPL once, `using <RepoName>`, edit code, re-run.

## 4. Formatting

CliMA repos use [JuliaFormatter](https://github.com/domluna/JuliaFormatter.jl), invoked from the repo root:

```julia
using JuliaFormatter
format(".")
```

CI pins a JuliaFormatter major version that varies between repos. See [code_style.md §1](../code-quality/code_style.md) for the version-matching procedure and the recommended pre-commit hook.

## 5. Git workflow

Prefer **rebasing** over merging to keep history linear:

```bash
git fetch origin main
git rebase origin/main
```

When starting a new task, base your branch on the latest remote `main`:

```bash
git stash
git checkout main
git pull origin main
git checkout -b <initials>/<short-description>   # e.g. ts/fix-precip-bug
git stash pop
```

Each commit should be a logical unit of work and keep the model compilable.

## 6. The first PR loop

A typical first PR:

1. Branch from the latest `main` (§5).
2. Make changes; iterate in the REPL with Revise.
3. Run the package's tests with `Pkg.test()`. It activates the test environment with the test-only deps, which a bare `include("test/runtests.jl")` does not.
4. Format (§4).
5. Add a `NEWS.md` entry if the change is user-visible. See [changelogs_and_versions.md](../code-quality/changelogs_and_versions.md).
6. Commit, push, and open the PR. The repo-specific guide names the CI driver and the test groups that should be green before review.

For PR-review conventions, see [review.md](review.md). For the actions an AI agent takes only with explicit user approval, see [agent_autonomy.md](agent_autonomy.md).

## 7. Removing a feature

Removing a feature means removing it everywhere:

1. **Source**: delete implementation code, structs, and methods.
2. **Configuration**: remove options from config files and parsers; choosing a removed option must raise an `error` that lists the valid alternatives.
3. **Tests**: delete targeted tests; update integration tests to use supported alternatives. Mirror changes between `src/` and `test/`.
4. **Dependencies**: if a package was used only by the removed feature, drop it from both `[deps]` and `[compat]` ([dependency_management.md §5](../architecture/dependency_management.md)).
5. **Documentation**: update docstrings and docs pages.
6. **`NEWS.md` entry**: marked breaking (`![][badge-💥breaking]` in repos that use badges) if the feature was public API ([changelogs_and_versions.md](../code-quality/changelogs_and_versions.md)).

## 8. Resolving a stuck environment

`Pkg` occasionally fails to find a satisfiable version set, typically after a `[compat]` change. The recovery sequence, cheapest first:

```julia
import Pkg
Pkg.instantiate()   # 1. make the manifest match Project.toml
Pkg.resolve()       # 2. re-run the resolver against current compat bounds
Pkg.update()        # 3. move every direct dep to its newest compat-allowed version
```

Step 3 changes the resolved version of every dependency; an agent needs user approval for it ([agent_autonomy.md](agent_autonomy.md)).

If those do not converge, one package is usually pinned at a version outside the current compat bounds. Remove and re-add it so the resolver picks a fresh version:

```julia
Pkg.rm("OffendingPackage")
Pkg.add("OffendingPackage")
```

Two mutually-constraining packages should be removed and re-added together. `Pkg.status()` shows current pins; `Pkg.resolve()` prints the resolver's diagnostic, which names the conflicting bounds.

## 9. Useful Julia tooling beyond the basics

Install these in your *base* environment, not the project's:

- **[TestEnv.jl](https://github.com/JuliaTesting/TestEnv.jl)**: `using TestEnv; TestEnv.activate()` makes the test-only deps available in an interactive REPL, so you can debug a failing test without `Pkg.test`'s startup cost.
- **[BenchmarkTools.jl](https://github.com/JuliaCI/BenchmarkTools.jl)**: `@benchmark` for measuring time and allocations of hot-path code. See [allocation_debugging.md §5](../performance/allocation_debugging.md).
- **[LiveServer.jl](https://github.com/JuliaDocs/LiveServer.jl)**: `servedocs()` builds the docs site locally and auto-reloads on file changes.
- **[About.jl](https://github.com/tecosaur/About.jl)**: `about(x)` summarizes any value's type, memory layout, and methods.
- **[OhMyREPL.jl](https://github.com/KristofferC/OhMyREPL.jl)**: syntax highlighting in the REPL. Add `using OhMyREPL` to `~/.julia/config/startup.jl` so it loads with every session.
- **[Kaimon.jl](https://github.com/kahliburke/Kaimon.jl)**: gives AI agents direct access to the Julia REPL, alongside `Revise` and `Infiltrator`, for iteration and debugging.

## 10. Knowing where to look

- The Julia manual: [docs.julialang.org](https://docs.julialang.org/en/v1/manual/getting-started/). The [Variables](https://docs.julialang.org/en/v1/manual/variables/) through [Documentation](https://docs.julialang.org/en/v1/manual/documentation/) sections cover day-to-day work.
- [Modern Julia Workflows](https://modernjuliaworkflows.org/writing/): an opinionated guide to REPL-driven Julia development.
- The [Julia performance tips](https://docs.julialang.org/en/v1/manual/performance-tips/): read once when you start writing hot-path code; [type_stability.md](../performance/type_stability.md) and [gpu_performance.md](../performance/gpu_performance.md) are the CliMA-specific extension.
- For ecosystem-wide conventions (`Y`/`Yₜ`/`p` state, `ᶜ`/`ᶠ` notation, module aliases), see [ecosystem_conventions.md](../architecture/ecosystem_conventions.md).

## Self-correction

If this guide is discovered to be stale or missing a pattern, update it.
