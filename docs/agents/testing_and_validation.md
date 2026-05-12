# Testing and Validation Guide

This guide covers testing patterns for CliMA code: type-stability verification, allocation regression testing, and test group organization.

## Type-stability checks

The canonical home for type-stability tooling (`@inferred`, `JET.@report_opt`, `@code_warntype`, the Float32/Float64 test template) is [type_stability.md](type_stability.md). Use `@inferred` as a CI regression gate for any new physics function:

```julia
@test @inferred(my_physics_func(FT(1.0), FT(2.0))) isa FT
```

## Allocation regression tests

After implementing or modifying hot-path code, verify zero allocations:

```julia
# Warm up (forces compilation)
remaining_tendency!(Yₜ, Y, p, t)
# Assert zero allocations on the hot path
@test (@allocated remaining_tendency!(Yₜ, Y, p, t)) == 0
```

Allocation benchmarks in `perf/` are not run automatically in CI. Allocation regressions must be caught during review.

## Aqua.jl quality checks

All CliMA packages run `Aqua.jl` tests in CI. These checks catch common package quality issues:

- `test_stale_deps`: fails if a package in `[deps]` is not used in source code. This is the most common failure — usually caused by adding a dev tool to `[deps]` instead of `[extras]` (see [Dependency Management](dependency_management.md)).
- `test_deps_compat`: fails if `[compat]` entries are missing for dependencies.
- `test_undefined_exports`: fails if an exported symbol is not defined.
- `test_unbound_args`: detects methods with unbound type parameters (can cause ambiguities).
- `test_ambiguities`: detects method ambiguities that could cause dispatch errors.
- `test_piracies`: detects type piracy (defining methods on types you don't own).

Standard pattern across CliMA repos:

```julia
# test/aqua.jl
using Aqua
using MyPackage

@testset "Aqua tests (performance)" begin
    ua = Aqua.detect_unbound_args_recursively(MyPackage)
    @test isempty(ua)
    ambs = Aqua.detect_ambiguities(MyPackage; recursive = true)
    @test isempty(ambs)
end

@testset "Aqua tests (additional)" begin
    Aqua.test_all(MyPackage; ambiguities = false, unbound_args = false)
end
```

## AD compatibility tests

Many CliMA packages include dedicated AD test files (for example, `test/ad_tests.jl` or `test/test_ad_compatibility.jl`). The standard pattern validates ForwardDiff gradients against finite differences:

```julia
using ForwardDiff

function check_derivative(f, x; rtol = 5e-2, atol = 1e-8)
    ad = ForwardDiff.derivative(f, x)
    ε = sqrt(eps(typeof(x)))
    fd = (f(x + ε) - f(x - ε)) / (2ε)
    @test isapprox(ad, fd; rtol, atol)
end
```

When adding new physics functions, add corresponding AD tests. See [AD Compatibility](ad_compatibility.md).

## GPU test files

Several CliMA packages maintain a separate `test/runtests_gpu.jl` entry point for GPU-specific tests. The standard pattern:

```julia
# Determine array type from command-line args or CUDA availability
arg = get(ARGS, 1, "")
if arg == "Array"
    ArrayType = Array
elseif arg == "CuArray"
    import CUDA
    ArrayType = CUDA.CuArray
    CUDA.allowscalar(false)
else
    # Default: use GPU if available
    try
        import CUDA
        ArrayType = CUDA.functional() ? CUDA.CuArray : Array
    catch
        ArrayType = Array
    end
end
```

GPU tests should use `CUDA.allowscalar(false)` to catch accidental scalar indexing into GPU arrays.

## Self-correction

If this guide is discovered to be stale or missing a pattern, update it.
