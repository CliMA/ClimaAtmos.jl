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

- **A dependency used only in tests goes in `test/Project.toml`**, never in the root `[deps]`. If it leaks into root deps, `Aqua.test_stale_deps` fails.
- **The root `Project.toml` is the package's published contract.** Adding to its `[deps]` is a deliberate API change; it forces every downstream consumer to install that dep too.
- **`docs/Project.toml` may `dev` the repo itself.** Documenter needs the in-tree source; do not list the package as a regular dep there.
- **`perf/Project.toml` is local-only.** Allocation and flame scripts are run by hand or in a dedicated CI job; treat its `Manifest.toml` (if committed) as advisory, not authoritative.
- **The runtime `[compat]` block applies only to root `[deps]`.** Test/docs/perf compats are independent and should be kept lighter so they don't fight the root bounds.

When adding a new dep, ask: *who needs this at runtime when a downstream user does `using MyPackage`?* If the answer is "no one", it belongs in a non-root environment.

### Test-target alternative

Library repos (Thermodynamics, CloudMicrophysics, SurfaceFluxes, ClimaTimeSteppers) use a separate `test/Project.toml`. Some smaller packages use the older `[extras]` + `[targets]` pattern in the root `Project.toml` instead:

```toml
[extras]
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
JuliaFormatter = "98e50ef6-434e-11e9-1051-2b60c6c9e899"

[targets]
test = ["Test", "JuliaFormatter"]
```

Both are accepted by `Pkg.test()`; match whichever convention the repo already uses. Do not mix them.

## 2. Runtime vs development dependencies

**Rule**: never place development tools (for example, `JuliaFormatter.jl`, `BenchmarkTools.jl`, `Aqua.jl`, `JET.jl`) in the root `[deps]` section of `Project.toml`. Put them in `test/Project.toml` (or in the `[extras]`/`[targets]` block: see §1).

**Why**: dev tools in root `[deps]` cause `Aqua.test_stale_deps` to fail (if unused by `src/`) and inflate the dependency footprint for every downstream consumer.

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

The extension module lives at `ext/MyPackageCUDAExt.jl` and can `using` both the parent package and the weak dep. Choose `ext/` over `src/` when the integration is optional or only meaningful with the weak dep loaded; keep small, core-path deps in `src/`. `[weakdeps]` entries still need `[compat]` entries; Aqua flags missing bounds. For the full Julia-level semantics, see the [Julia manual](https://pkgdocs.julialang.org/v1/creating-packages/#Conditional-loading-of-code-in-packages-(Extensions)).

## 4. Cross-package local development

When developing across multiple local packages where the local branch version is higher than the current compat allows:

1. Update `Project.toml` compat to include the new version:

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

In nested environments (like `test/Project.toml`), running tests from the parent directory often resolves conflicts:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate(); include("test/runtests.jl")'
```

## 5. Pruning obsolete packages

When a feature is removed, check whether it was the exclusive consumer of any external package:

```bash
grep -r "PackageName" src/
```

If no remaining references exist, remove the package from both `[deps]` and `[compat]` sections of `Project.toml`. This reduces precompilation times and simplifies the dependency graph.

### Verification

After removing a dependency, verify the package still precompiles:

```bash
julia --project -e 'using PackageName'
```

## 6. Avoiding internal modules from dependencies

Prefer standard Julia operations over internal modules from dependencies. Internal modules may be refactored or removed in future versions, without being marked as breaking changes. Quantities that are not explicitly `export`-ed or documented as part of a package's public API should be considered internals, and other packages should refrain from accessing them if possible.

## 7. Writing `[compat]` bounds

Keep `[compat]` entries as broad as the package's API actually supports. Overly narrow upper bounds in upstream packages are the most common source of unresolvable graphs across the CliMA ecosystem. Tighten a bound only when you can name the specific incompatibility (a removed symbol, a changed signature, a regression). When working across nested environments (`docs/`, `test/`, `perf/`, `.buildkite/`) against a local checkout of the parent package, use `Pkg.develop(path="..")` so the subdirectory picks up unreleased changes.

For the recovery procedure when an environment fails to resolve, see [onboarding.md §8](../workflow/onboarding.md).

## Self-correction

If this guide is discovered to be stale or missing a pattern, update it.
