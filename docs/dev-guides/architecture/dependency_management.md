# Dependency Management Guide

This guide covers rules for managing Julia package dependencies in CliMA repositories.

## 1. Multiple `Project.toml`s in one repo

CliMA repos typically have several Julia environments side-by-side, each with its own `Project.toml` (and, if instantiated, `Manifest.toml`). They serve different purposes and must be kept distinct.

| Path                          | What it is                                          | Activated by                                |
|:------------------------------|:----------------------------------------------------|:--------------------------------------------|
| `Project.toml` (repo root)    | The package itself: runtime `[deps]`, `[compat]`, `[weakdeps]`, `[extensions]`. | `using Pkg; Pkg.activate(".")` |
| `test/Project.toml`           | Test-only dependencies (Aqua, JET, BenchmarkTools, CUDA, Documenter, etc.). | `Pkg.test()` |
| `docs/Project.toml`           | Documentation build dependencies (Documenter, Literate, plotting). | `julia --project=docs docs/make.jl` |
| `perf/Project.toml`           | Allocation / flame / JET scripts (BenchmarkTools, Profile tools). | `julia --project=perf perf/<script>.jl` |
| `.buildkite/Project.toml`     | Pipeline driver environment (present in repos that run Buildkite). | invoked by the pipeline runner |
| `<demo-dir>/Project.toml`     | Self-contained demos (e.g. `box/`, `parcel/`, `papers/` in CloudMicrophysics; `examples/` in some repos). | `julia --project=<demo-dir>` |

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

**Rule**: never place development tools (for example, `JuliaFormatter.jl`, `BenchmarkTools.jl`, `Aqua.jl`, `JET.jl`) in the root `[deps]` section of `Project.toml`. Put them in `test/Project.toml` (or in the `[extras]`/`[targets]` block — see §1).

**Why**: dev tools in root `[deps]` cause `Aqua.test_stale_deps` to fail (if unused by `src/`) and inflate the dependency footprint for every downstream consumer.

## 3. Package extensions

Julia 1.9+ supports *package extensions*: optional code paths in a package that only load when a specific dependency is also loaded by the user. This lets a package offer integrations (plotting, GPU, file formats) without forcing every downstream consumer to install the heavy dep.

A weak dependency lives in `[weakdeps]` and is paired with an entry in `[extensions]` that names the extension module and lists which weak deps trigger it:

```toml
[deps]
SomeCoreDep = "..."

[weakdeps]
CUDA  = "052768ef-5323-5732-b1bb-66c8b64840ba"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"

[extensions]
MyPackageCUDAExt  = ["CUDA"]
MyPackagePlotsExt = ["Plots"]

[compat]
SomeCoreDep = "1"
CUDA  = "5"
Plots = "1"
julia = "1.10"
```

The extension code lives under `ext/`:

- `ext/MyPackageCUDAExt.jl` — module that uses `CUDA` and extends methods defined in the parent package.
- `ext/MyPackagePlotsExt.jl` — module that uses `Plots`.

Each extension module is a regular Julia module: it can `using` the parent package and the weak dep, define new methods on the parent's functions, and refer to types from either.

### When to use `ext/` vs `src/`

Choose `ext/` when:

- The integration would otherwise force every consumer of the parent package to install a large, optional dep (CUDA, plotting backends, file-format readers).
- The functionality only makes sense when the weak dep is already loaded — for example, GPU-specific code paths or plotting helpers.

Keep code in `src/` when:

- The dep is small and used by the package's core path.
- The dep is required for any meaningful use of the package.

### Compat for weak deps

`[weakdeps]` entries still need `[compat]` entries — without them, `Pkg` cannot resolve a working version, and Aqua's compat check will flag the missing bound. The rules from [§2 Compat bounds](#2-compat-bounds) apply unchanged.

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

Prefer standard Julia operations over internal modules from dependencies (for example, `ClimaCore.RecursiveApply`). Internal modules may be refactored or removed in future versions. This applies to downstream consumers; within ClimaCore itself, `RecursiveApply` is part of the internal API.

| ❌ Discouraged (in downstream packages) | ✅ Preferred |
|:---|:---|
| `rzero(T)` | `zero(T)` |
| `a ⊞ b` | `a + b` |
| `a ⊠ b` | `a * b` |

## 7. Resolving a stuck environment

CliMA packages depend on a large graph of internal and external packages, and `Pkg` occasionally fails to find a satisfiable version set — especially after a downstream change to `[compat]`. The cheapest-to-most-expensive recovery sequence:

```julia
import Pkg

# 1. Make sure the manifest matches Project.toml.
Pkg.instantiate()

# 2. Re-run the resolver from scratch against the current compat bounds.
Pkg.resolve()

# 3. Move every direct dependency to its newest compat-allowed version.
Pkg.update()
```

If those do not converge, one specific package is usually pinned at a version that no longer fits the rest of the graph. Remove and re-add it so the resolver picks a fresh version:

```julia
Pkg.rm("OffendingPackage")
Pkg.add("OffendingPackage")
```

Two packages can constrain each other in conflicting ways; remove them both at once before re-adding. `Pkg.status()` shows current pins and `Pkg.resolve()` prints the resolver's own diagnostic — read that error before guessing.

When working in a subdirectory environment (`docs/`, `test/`, `perf/`, `.buildkite/`) against a local checkout of the parent package, use `Pkg.develop(path="..")` so the subdirectory picks up unreleased changes. See §1 for the full set of nested environments.

Best practice when *writing* `[compat]` entries: keep them as broad as the package's API actually supports. Overly narrow upper bounds in upstream packages are the most common source of unresolvable graphs across the ecosystem. Tighten a bound only when you can name the specific incompatibility (a removed symbol, a changed signature, a regression).

## 8. Licensing

All CliMA repositories must include:
- A `LICENSE` file (Apache License 2.0) in the repository root.
- A `NOTICE` file containing the copyright statement.

## Self-correction

If this guide is discovered to be stale or missing a pattern, update it.
