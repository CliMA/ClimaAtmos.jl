# Dependency Management Guide

This guide covers rules for managing Julia package dependencies in CliMA repositories.

## 1. Runtime vs development dependencies

**Rule**: never place development tools (for example, `JuliaFormatter.jl`, `BenchmarkTools.jl`) in the main `[deps]` section of `Project.toml`.

**Why**: it causes `Aqua.test_stale_deps` to fail and bloats the dependency tree for downstream users.

**Solution**: place dev tools in the `[extras]` section and add them to the `test` target:

```toml
[extras]
JuliaFormatter = "..."
BenchmarkTools = "..."

[targets]
test = ["Test", "JuliaFormatter", "BenchmarkTools"]
```

## 2. Cross-package local development

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

## 3. Pruning obsolete packages

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

## 4. Avoiding internal modules from dependencies

Prefer standard Julia operations over internal modules from dependencies (for example, `ClimaCore.RecursiveApply`). Internal modules may be refactored or removed in future versions.

| ❌ Discouraged | ✅ Preferred |
|:---|:---|
| `rzero(T)` | `zero(T)` |
| `a ⊞ b` | `a + b` |
| `a ⊠ b` | `a * b` |

## Self-correction

If this guide is discovered to be stale or missing a pattern, update it.
