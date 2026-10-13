# Type Stability Guide

This guide covers patterns for ensuring type-stable Julia code across the CliMA ecosystem, with particular emphasis on Float32 compatibility for GPU simulations.

## 1. Float32 pollution checklist

When targeting GPUs (which typically use `Float32`) or working within strict inference environments (such as `ClimaCore.lazy()` broadcasts), even a single `Float64` operation can cause the compiler to infer a `Union{Float32, Float64}` return type, which blocks kernel fusion and triggers `BroadcastInferenceError`.

Common sources and fixes:

| Source             | ❌ Bad                         | ✅ Good                                         |
|:-------------------|:-------------------------------|:------------------------------------------------|
| Float literal      | `x + 1.2` (`1.2` is `Float64`) | `x + FT(1.2)`                                   |
| Infinity / NaN     | `Inf`, `NaN`                   | `FT(Inf)`, `FT(NaN)`                            |
| Random numbers     | `rand()`                       | `rand(FT)`                                      |
| Math constants     | `2 * π` converts to `Float64`  | `2 * FT(π)` (`x * π` is sufficient if `x::FT`)  |
| Rational exponent  | `x^(2//3)` (`Rational{Int64}` promotes to `Float64`) | `cbrt(x)^2` (see [gpu_performance.md §10](gpu_performance.md)) |
| Literal zero/one   | `0.0`, `1.0`                   | `zero(FT)`, `one(FT)`                           |

## 2. Detecting type instability

### `@code_warntype`

Visual inspection of inferred types. Look for red `Any` or `Union` annotations in the output.

```julia
@code_warntype my_function(args...)
```

### `JET.@report_opt`

Finds dynamic dispatch call sites in complex call graphs. Prefer this over `@code_warntype` for deeply nested code.

```julia
using JET
JET.@report_opt my_tendency(Y, p, t)
```

### `@inferred`

Fails at test time if the return type is not fully inferred by the compiler. Use as a CI regression gate.

```julia
@test @inferred(my_physics_func(FT(1.0), FT(2.0))) isa FT
```

## 3. Abstract types in struct fields

Struct fields should be concrete or parametric for type stability and performance. See [SDP 4](../code-quality/software_design_patterns.md).

### Splitting dispatch on `Union{T, Nothing}`

When a struct contains an optional field typed as `Union{T, Nothing}`, accessing that field in a hot path triggers runtime type checks. Split the method based on the type parameter to compile away the branch.

```julia
struct MyParams{ICE}
    ice::ICE  # IceParams or Nothing
end

# ❌ Runtime type check
function compute(params::MyParams)
    if !isnothing(params.ice)
        # ice logic
    end
end

# ✅ Compile-time dispatch: branch eliminated
function compute(params::MyParams{Nothing})
    # warm-only logic
end
function compute(params::MyParams{ICE}) where {ICE <: IceParams}
    # ice logic (guaranteed present)
end
```

## 4. Type-stability test template

Gate type stability with a test that iterates over both precisions:

```julia
for FT in (Float32, Float64)
    result = my_function(FT(1.0), FT(2.0))
    @test result isa FT
    @test @inferred(my_function(FT(1.0), FT(2.0))) isa FT
end
```

For functions returning a `NamedTuple`, verify every field:

```julia
for FT in (Float32, Float64)
    result = my_tendency(FT(1.0), FT(2.0))
    for field in propertynames(result)
        @test getproperty(result, field) isa FT
    end
end
```

## 5. Diagnosing `BroadcastInferenceError`

A `BroadcastInferenceError` from `ClimaCore` usually means a function inside a `@.` or `lazy()` broadcast has a `Union` return type.

### Diagnosis steps

1. **Test the scalar function directly** with `@inferred` and `Float32` inputs. Do not rely only on the broadcast-level test.
2. **Trigger all branches**: type instability often hides in conditional branches (for example, `iszero(x) ? Inf : ...`). Ensure reproduction scripts test edge cases (zeros, thresholds) that trigger every logical path.
3. **Check `@code_warntype` output**: look for `Union{Float32, Float64}` in the return type, which indicates one branch is polluting the floating-point context.

### Common "false pass" scenario

A unit test may pass with standard inputs but fail in integrated simulations because:

- The test inputs never trigger the code path containing the Float64 literal.
- Edge-case inputs (zero, negative, very large) expose a hidden `Inf` or `NaN` literal.

## 6. Prefer using base julia general number functions

Use general number functions from Base Julia that are designed to work across numeric types, as they often have optimized methods for different types and can help maintain type stability. Examples include `iszero`, `isfinite`, `isinf`, `isone`, `isodd`, and `isnan`. These functions are more likely to return consistent types across different numeric inputs, reducing the risk of type instability in your code.

## Self-correction

If this guide is discovered to be stale or missing a pattern, update it.
