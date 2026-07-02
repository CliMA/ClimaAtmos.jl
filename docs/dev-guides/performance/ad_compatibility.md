# AD Compatibility Guide

This guide covers patterns for writing Julia code that is compatible with Automatic Differentiation (AD) tools such as ForwardDiff and Enzyme. These rules apply to all tendency, physics, and parameterization functions across the CliMA ecosystem.

## Core rules

| Rule                                                                                    | Rationale |
|:----------------------------------------------------------------------------------------|:----------|
| Duck-type functions ([SDP 14](../code-quality/software_design_patterns.md))             | Dual numbers flow through without type-annotation barriers |
| `FT = typeof(x)` or `eltype(x)` ([SDP 15](../code-quality/software_design_patterns.md)) | Lets AD supply the numeric type |
| `zero(x)` / `one(x)` ([SDP 16](../code-quality/software_design_patterns.md))            | Type-agnostic; correctly typed for `Dual` |
| Do not write `Dual` values into non-`Dual` buffers                                      | Invalid operation; `convert(<:AbstractFloat, dual)` throws a `MethodError` |

## Before / after example

```julia
# ❌ AD-fragile: `where {FT}` forces x and y to share a type, so a mixed call
# like compute(Dual(1.0), 2.0) will throw a MethodError
@inline compute(x::FT, y::FT) where {FT} = y > FT(0) ? FT(1) - x^2 : FT(0)

# ✅ AD-compatible: x and y can have different types, but the result always
# matches the type of x
@inline compute(x, y) = y > zero(y) ? one(x) - x^2 : zero(x)
```

This example can be rewritten even more efficiently using the `ifelse` function:

```julia
@inline compute(x, y) = ifelse(y > zero(y), one(x) - x^2, zero(x))
```

Using `ifelse` is not strictly necessary for ForwardDiff or Enzyme (conditional branches can be differentiated separately), but it is preferred over generic `if/else` and `?/:` constructs to avoid thread divergence on GPUs (see [SDP 17](../code-quality/software_design_patterns.md) and the [Branchless Code Guide](branchless_code.md)).

## When type constraints are OK

Type annotations are acceptable in these specific contexts:

- **Struct constructors**: `MySGS(::Type{FT}; ...)`, where `FT` determines the element type of arrays (`SVector`, `SMatrix`) or parameter sets.
- **Dispatch on non-numeric types**: `method(::GaussianSGS, ...)`; dispatching on a distribution type or model type is fine because these are not numeric values that AD would differentiate through.

## AD-compatible clamping

Standard `clamp(x, low, high)` is generally safe for AD. For zero-clamping the simplest idiom (used in `CloudMicrophysics.Utilities.clamp_to_nonneg`) is:

```julia
@inline clamp_to_nonneg(x) = max(zero(x), x)
```

`max` propagates the Dual partials through whichever argument wins. An equivalent branchless form is `ifelse(x < zero(x), zero(x), x)`; the `zero(x)` term ensures the negative branch carries the same type (including Dual partials) as `x`.

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

1. **Type annotations on arguments**: `f(x::FT) where {FT <: AbstractFloat}` rejects `ForwardDiff.Dual` inputs, and `f(x::FT, x::FT) where {FT}` rejects inputs with mixed types. Use duck typing as much as possible, only adding type constraints when they are needed for multiple dispatch.
2. **Constants typed from the wrong source**: Any value whose type is hardcoded like `Float64(x)`, or captured from an unrelated source like `eltype(params)`, will not pick up the `Dual` type of input arguments. Avoid manual type conversions if possible, and use `one(x)` / `zero(x)` instead of `FT(1)` / `FT(0)`.
3. **Mutation**: In-place modification of arrays or mutable structs can break reverse-mode AD (Enzyme). Prefer returning new values from pure functions.

## Self-correction

If this guide is discovered to be stale or missing a pattern, update it.
