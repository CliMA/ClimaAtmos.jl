# Onboarding to a CliMA Repository

A short walk from a fresh machine to a productive REPL session inside any CliMA Julia package. This guide is intentionally generic — repo-specific quirks live in the package's own `*_specific.md` guide.

## 1. Install the core tools

1. **Julia.** Install via [`juliaup`](https://julialang.org/downloads/). `juliaup add release` installs the current stable channel; CliMA repos test on the current LTS and one or two newer point releases — check `.github/workflows/ci.yml` of the repo you're working in for the exact matrix.
2. **Git** and a GitHub account with an SSH key — see GitHub's [SSH setup guide](https://docs.github.com/en/authentication/connecting-to-github-with-ssh).
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
Pkg.instantiate()    # download every dep at the manifest-pinned version
Pkg.status()         # sanity-check the resolved versions
import <RepoName>    # confirm the package loads
```

If `Pkg.instantiate()` fails, see [dependency_management.md §6](../architecture/dependency_management.md) for the standard recovery sequence.

## 3. Keep a long-running REPL — and avoid restarting it

Julia's first-call latency comes from compilation. Restarting the REPL throws that work away. The two pieces of standard tooling that let you iterate without restarting:

- **[Revise.jl](https://github.com/timholy/Revise.jl)** — watches package source files and patches updated method definitions into the running session.
- **[Infiltrator.jl](https://github.com/JuliaDebug/Infiltrator.jl)** — drops you into an interactive REPL at a `@infiltrate` breakpoint without instrumenting the function, so it does not invalidate compiled code.

Install both into your *base* (`v1.x`) environment so every REPL gets them automatically:

```julia
julia -e 'using Pkg; Pkg.add(["Revise", "Infiltrator"])'
```

Then add a startup file at `~/.julia/config/startup.jl`:

```julia
using Revise
using Infiltrator
```

Now your normal loop is: start the REPL once, `using <RepoName>`, edit code, re-run — no restart needed.

## 4. Formatting

CliMA repos use [JuliaFormatter](https://github.com/domluna/JuliaFormatter.jl), invoked from the repo root:

```julia
using JuliaFormatter
format(".")
```

CI pins a specific JuliaFormatter major version; check the JuliaFormatter workflow file under `.github/workflows/` (named `JuliaFormatter.yml` in some repos, `julia_formatter.yml` in others) and install the matching version into your base environment to avoid CI-only formatting diffs. See [code_style.md](../code-quality/code_style.md) and [ci_triage.md §10](ci_triage.md).

A pre-commit hook that runs `format(".")` is a cheap way to never see a formatter-only CI failure — JuliaFormatter ships [pre-commit instructions](https://domluna.github.io/JuliaFormatter.jl/stable/integrations/#pre-commit).

## 5. The first PR loop

A typical first PR follows this rhythm:

1. Branch: `git checkout -b <initials>/<short-description>` (e.g. `ts/fix-precip-bug`).
2. Make changes; iterate in the REPL with Revise.
3. Run the package's tests: `Pkg.test()` (prefer this over manually `include`ing `test/runtests.jl` — `Pkg.test` activates the test environment with the test-only deps).
4. Format: `using JuliaFormatter; format(".")`.
5. Add a `NEWS.md` entry if the change is user-visible — see [changelogs_and_versions.md](../code-quality/changelogs_and_versions.md).
6. Commit, push, and open the PR. The repo-specific guide names the canonical CI driver and the test groups that should be green before review.

For PR-review conventions, see [review.md](review.md). For what AI agents may and may not do without explicit approval, see [agent_autonomy.md](agent_autonomy.md).

## 6. Useful Julia tooling beyond the basics

These all live in your *base* environment, not the project's:

- **[TestEnv.jl](https://github.com/JuliaTesting/TestEnv.jl)** — `using TestEnv; TestEnv.activate()` makes the test-only deps available in an interactive REPL, so you can debug a failing test without `Pkg.test`'s startup cost.
- **[BenchmarkTools.jl](https://github.com/JuliaCI/BenchmarkTools.jl)** — `@benchmark` for measuring time and allocations of hot-path code. See [allocation_debugging.md §5](../performance/allocation_debugging.md).
- **[LiveServer.jl](https://github.com/JuliaDocs/LiveServer.jl)** — `servedocs()` builds the docs site locally and auto-reloads on file changes.
- **[About.jl](https://github.com/tecosaur/About.jl)** — `about(x)` summarizes any value's type, memory layout, and methods.
- **[OhMyREPL.jl](https://github.com/KristofferC/OhMyREPL.jl)** - provides syntax highlighting within the REPL
  - Add `using OhMyREPL` to `~/.julia/config/startup.jl` to ensure it loads with every REPL session.
- **[Kaimon.jl](https://github.com/kahliburke/Kaimon.jl)** - provides AI agents with direct access to the Julia REPL, with your standard tools like `Revise` and `Infiltrator` for quick iteration and debugging

## 7. Knowing where to look

- The Julia manual: [docs.julialang.org](https://docs.julialang.org/en/v1/manual/getting-started/). The [Variables](https://docs.julialang.org/en/v1/manual/variables/) through [Documentation](https://docs.julialang.org/en/v1/manual/documentation/) sections cover what you need for day-to-day work.
- [Modern Julia Workflows](https://modernjuliaworkflows.org/writing/) — a current, opinionated guide to REPL-driven Julia development.
- The [Julia performance tips](https://docs.julialang.org/en/v1/manual/performance-tips/) — read once when you start writing hot-path code; [type_stability.md](../performance/type_stability.md) and [gpu_performance.md](../performance/gpu_performance.md) are the CliMA-specific extension.
- For ecosystem-wide conventions (`Y`/`Yₜ`/`p` state, `ᶜ`/`ᶠ` notation, module aliases), see [ecosystem_conventions.md](../architecture/ecosystem_conventions.md).

## Self-correction

If this guide is discovered to be stale or missing a pattern, update it.
