# AD Compatibility Guide

This guide covers patterns for writing Julia code that is compatible with Automatic Differentiation (AD) tools such as ForwardDiff and Enzyme. These rules apply to all tendency, physics, and parameterization functions across the CliMA ecosystem.

## Core rules

| Rule | Rationale |
|:---|:---|
| Duck-type functions ([SDP 14](software_design_patterns.md)) | Dual numbers flow through without type annotation barriers |
| `FT = typeof(x)` or `eltype(x)` ([SDP 15](software_design_patterns.md)) | Lets AD supply the numeric type |
| `zero(x)` / `one(x)` ([SDP 16](software_design_patterns.md)) | Type-agnostic; correct for Dual numbers |
| `ifelse` not `if-else` on floating-point values ([SDP 17](software_design_patterns.md)) | Both branches needed for gradient computation; avoids non-differentiable kinks |
| `@inline` every kernel function | Required for kernel fusion; transparent to AD |
| No mutation of scalar locals inside kernel | Enzyme and ForwardDiff prefer pure functional code |

## Before / after example

```julia
# ❌ AD-fragile — `where {FT}` forces x and y to share a type, so a mixed call
# like compute(Dual(1.0), 2.0) fails to dispatch; the if/else also introduces a
# non-differentiable kink and is problematic for reverse-mode AD (Enzyme).
@inline function compute(x::FT, y::FT) where {FT}
    if x > FT(0)
        return FT(1) - x^2
    else
        return FT(0)
    end
end

# ✅ AD-compatible — duck typed (each arg has its own type), branchless,
# type-agnostic constants derived from the input.
@inline function compute(x, y)
    return ifelse(x > zero(x), one(x) - x^2, zero(x))
end
```

## When type constraints are OK

Type annotations are acceptable in these specific contexts:

- **Struct constructors**: `MySGS(::Type{FT}; ...)` — needed to determine the element type for statically-sized arrays (`SVector`, `SMatrix`).
- **Dispatch on non-numeric types**: `method(::GaussianSGS, ...)` — dispatching on a distribution type or model type is fine because these are not numeric values that AD would differentiate through.
- **Performance-critical inner loops**: Sometimes needed, but verify that AD still works by testing with Dual numbers.

## AD-compatible clamping

Standard `clamp(x, low, high)` is generally safe, but for zero-clamping where gradient information must be preserved through the zero boundary:

```julia
# Keeps the Dual value zero while preserving type
@inline clamp_to_nonneg(x) = ifelse(x < zero(x), zero(x) * x, x)
```

The `zero(x) * x` idiom ensures the result has the same type (including Dual partials) as `x`, while the value is zeroed.

## Testing AD compatibility

```julia
using ForwardDiff

# Verify function accepts Dual numbers and returns Dual
x_dual = ForwardDiff.Dual(1.0f0, 1.0f0)
result = my_physics_func(x_dual, params)
@test result isa ForwardDiff.Dual

# Verify gradient computes without error
grad = ForwardDiff.derivative(x -> my_physics_func(x, params), 1.0f0)
@test isfinite(grad)
```

For multi-argument functions, use `ForwardDiff.gradient` or `ForwardDiff.jacobian`.

## Common pitfalls

1. **Type annotations on arguments**: `f(x::Float64)` rejects `ForwardDiff.Dual{Float64}`. Use duck typing.
2. **Constants typed from the wrong source**: `FT(1.2)`, `Float64(1.2)`, or any constant whose type is hardcoded (or captured from an unrelated source like `eltype(params)`) does not pick up the `Dual` type of the live input. Mixed arithmetic still promotes to `Dual`, so simple expressions work, but the `Float64` value silently loses partials if it escapes into a context typed by the wrong `FT` — e.g. returned, assigned to a `Float64` field, or stored in a `Float64`-typed array. Prefer `one(x)` / `zero(x)`, or derive `FT` from the actual input via `typeof(x)` / `eltype(x)`.
3. **Mutation**: In-place modification of arrays or mutable structs can break reverse-mode AD (Enzyme). Prefer returning new values from pure functions.
4. **String interpolation**: `"value is $x"` inside a kernel triggers dynamic dispatch and is incompatible with GPU and some AD backends.

## Self-correction

If this guide is discovered to be stale or missing a pattern, update it.
