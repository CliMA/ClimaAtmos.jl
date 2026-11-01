# Dependency Management Guide

This guide covers rules for managing Julia package dependencies in CliMA repositories.

## 1. Multiple `Project.toml`s in one repo

CliMA repos typically have several Julia environments side-by-side, each with its own `Project.toml` (and, if instantiated, `Manifest.toml`). They serve different purposes and must be kept distinct.

| Path                          | What it is                                                                                                | Activated by                                |
|:------------------------------|:----------------------------------------------------------------------------------------------------------|:--------------------------------------------|
| `Project.toml` (repo root)    | The package itself: runtime `[deps]`, `[compat]`, `[weakdeps]`, `[extensions]`.                           | `using Pkg; Pkg.activate(".")`              |
| `test/Project.toml`           | Test-only dependencies (Aqua, JET, BenchmarkTools, CUDA, Documenter, etc.).                               | `Pkg.test()`                                |
| `docs/Project.toml`           | Documentation build dependencies (Documenter, Literate, plotting).                                        | `julia --project=docs docs/make.jl`         |
| `perf/Project.toml`           | Allocation / flame / JET scripts (BenchmarkTools, Profile tools).                                         | `julia --project=perf perf/<script>.jl`     |
| `.buildkite/Project.toml`     | Pipeline driver environment (present in repos that run Buildkite).                                        | invoked by the pipeline runner              |
| `<demo-dir>/Project.toml`     | Self-contained demos (e.g. `box/`, `parcel/`, `papers/` in CloudMicrophysics; `examples/` in some repos). | `julia --project=<demo-dir>`                |

Rules of thumb:

- **A dependency used only in tests goes in `test/Project.toml`**, never in the root `[deps]` (see §2).
- **The root `Project.toml` is the package's published contract.** Adding to its `[deps]` is an API change; it forces every downstream consumer to install that dep too.
- **`docs/Project.toml` may `dev` the repo itself.** Documenter needs the in-tree source; do not list the package as a regular dep there.
- **`perf/Project.toml` is self-contained.** Allocation and flame scripts are run by hand or in a dedicated CI job; treat its `Manifest.toml` (if committed) as advisory.
- **The runtime `[compat]` block applies only to root `[deps]`.** Test/docs/perf compats are independent; keep them looser than the root bounds so the environments resolve together.

When adding a new dep, ask: *who needs this at runtime when a downstream user does `using MyPackage`?* If the answer is "no one", it belongs in a non-root environment.

### Test-target alternative

Library repos (Thermodynamics, CloudMicrophysics, SurfaceFluxes, ClimaTimeSteppers) use a separate `test/Project.toml`. Some smaller packages use the older `[extras]` + `[targets]` pattern in the root `Project.toml`:

```toml
[extras]
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
JuliaFormatter = "98e50ef6-434e-11e9-1051-2b60c6c9e899"

[targets]
test = ["Test", "JuliaFormatter"]
```

Both are accepted by `Pkg.test()`; match whichever convention the repo already uses. Do not mix them.

## 2. Runtime vs development dependencies

Never place development tools (`JuliaFormatter.jl`, `BenchmarkTools.jl`, `Aqua.jl`, `JET.jl`, ...) in the root `[deps]`; put them in `test/Project.toml` or the `[extras]`/`[targets]` block. Dev tools are the most common offender: in root `[deps]` they fail `Aqua.test_stale_deps` and inflate the dependency footprint of every downstream consumer.

## 3. Package extensions

Julia 1.9+ supports *package extensions*: optional code paths that load only when a specific dependency is also loaded by the user. Use them to offer integrations (CUDA, plotting backends, file-format readers) without forcing every consumer to install the heavy dep. A weak dependency goes in `[weakdeps]`, paired with an `[extensions]` entry that names the trigger:

```toml
[weakdeps]
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"

[extensions]
MyPackageCUDAExt = ["CUDA"]

[compat]
CUDA = "5"
```

The extension module lives at `ext/MyPackageCUDAExt.jl` and can `using` both the parent package and the weak dep. Choose `ext/` over `src/` when the integration is optional or only meaningful with the weak dep loaded; keep small, core-path deps in `src/`. `[weakdeps]` entries still need `[compat]` entries; Aqua flags missing bounds. For the Julia-level semantics, see the [Julia manual](https://pkgdocs.julialang.org/v1/creating-packages/#Conditional-loading-of-code-in-packages-(Extensions)).

## 4. Cross-package local development

When developing across multiple local packages where the local branch version is higher than the current compat allows:

1. Update `Project.toml` compat to include the new version (agents: with user approval, see [agent_autonomy.md](../workflow/agent_autonomy.md)):

   ```toml
   # Before
   CloudMicrophysics = "0.30"

   # After
   CloudMicrophysics = "0.30, 0.31"
   ```

2. Develop the local path:

   ```julia
   using Pkg
   Pkg.develop(path="/path/to/local/CloudMicrophysics.jl")
   ```

3. Verify with `Pkg.status("CloudMicrophysics")` that the local path is active.

### Nested environment conflicts

When a nested environment (`test/`, `docs/`, `perf/`, `.buildkite/`) fails to resolve against a local checkout of the parent package, run `Pkg.develop(path="..")` inside that environment so it picks up the unreleased parent. As a fallback, run the tests from the root environment, which skips the test environment:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate(); include("test/runtests.jl")'
```

## 5. Pruning obsolete packages

When a feature is removed, check whether it was the only user of any external package:

```bash
grep -r "PackageName" src/
```

If nothing references it, remove the package from both `[deps]` and `[compat]` sections of `Project.toml`. This reduces precompilation times and simplifies the dependency graph.

### Verification

After removing a dependency, verify the package still precompiles:

```bash
julia --project -e 'using PackageName'
```

## 6. Avoiding internal modules from dependencies

Prefer standard Julia operations over internal modules from dependencies. Anything a package neither `export`s nor documents is internal and may change without a breaking release; see [architectural_boundaries.md §3](architectural_boundaries.md).

## 7. Writing `[compat]` bounds

Keep `[compat]` entries as broad as the package's API supports. Overly narrow upper bounds in upstream packages are the most common source of unresolvable graphs across the CliMA ecosystem. Tighten a bound only when you can name the specific incompatibility (a removed symbol, a changed signature, a regression).

For the recovery procedure when an environment fails to resolve, see [onboarding.md §8](../workflow/onboarding.md).

## Self-correction

If this guide is discovered to be stale or missing a pattern, update it.
