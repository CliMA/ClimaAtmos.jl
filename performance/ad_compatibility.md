# AD Compatibility Guide

This guide covers patterns for writing Julia code that is compatible with Automatic Differentiation (AD) tools such as ForwardDiff and Enzyme. These rules apply to tendency, physics, and parameterization functions in model repositories and to any code AD traverses; physics libraries follow the exception in "When type constraints are OK".

## Core rules

| Rule                                                                                    | Rationale |
|:----------------------------------------------------------------------------------------|:----------|
| Duck-type functions ([SDP 14](../code-quality/software_design_patterns.md))             | Dual numbers pass through; `where {FT}` also rejects `ClimaCore.Field`s whose `eltype`s differ |
| `FT = typeof(x)` or `eltype(x)` ([SDP 15](../code-quality/software_design_patterns.md)) | Lets AD supply the numeric type |
| `zero(x)` / `one(x)` ([SDP 16](../code-quality/software_design_patterns.md))            | Takes the type of `x`, including `Dual` |
| Do not write `Dual` values into non-`Dual` buffers                                      | `convert(<:AbstractFloat, dual)` throws a `MethodError` |

## Before / after example

```julia
# ❌ AD-fragile: `where {FT}` forces x and y to share a type, so a mixed call
# like compute(Dual(1.0), 2.0) throws a MethodError
@inline compute(x::FT, y::FT) where {FT} = y > FT(0) ? FT(1) - x^2 : FT(0)

# ✅ AD-compatible: x and y may differ in type; the result has the type of x
@inline compute(x, y) = y > zero(y) ? one(x) - x^2 : zero(x)
```

The branchless form uses `ifelse`:

```julia
@inline compute(x, y) = ifelse(y > zero(y), one(x) - x^2, zero(x))
```

ForwardDiff and Enzyme differentiate through `if/else` and `?:` as well; `ifelse` is preferred because it avoids thread divergence on GPUs (see [SDP 17](../code-quality/software_design_patterns.md) and the [Branchless Code Guide](branchless_code.md)).

## When type constraints are OK

Type annotations are acceptable in these contexts:

- **Struct constructors**: `MySGS(::Type{FT}; ...)`, where `FT` determines the element type of arrays (`SVector`, `SMatrix`) or parameter sets.
- **Dispatch on non-numeric types**: `method(::GaussianSGS, ...)`; AD does not differentiate through a distribution type or model type.
- **Library internals that keep numeric types homogeneous** (for example, CloudMicrophysics.jl, Thermodynamics.jl) may use `where {FT}` broadly; follow the package's existing style.

## AD-compatible clamping

`clamp(x, low, high)` is AD-safe. The zero-clamping idiom, used in `CloudMicrophysics.Utilities.clamp_to_nonneg`, is:

```julia
@inline clamp_to_nonneg(x) = max(zero(x), x)
```

`max` propagates the Dual partials through whichever argument wins. An equivalent branchless form is `ifelse(x < zero(x), zero(x), x)`; `zero(x)` gives the negative branch the same type as `x`, Dual partials included.

## Testing AD compatibility

```julia
using ForwardDiff

# The function accepts a Dual and returns a Dual
x_dual = ForwardDiff.Dual(1.0f0, 1.0f0)
result = my_physics_func(x_dual, params)
@test result isa ForwardDiff.Dual

# The derivative is finite
grad = ForwardDiff.derivative(x -> my_physics_func(x, params), 1.0f0)
@test isfinite(grad)
```

For multi-argument functions, use `ForwardDiff.gradient` or `ForwardDiff.jacobian`.

## Common pitfalls

1. **Type annotations on arguments**: `f(x::FT) where {FT <: AbstractFloat}` rejects `ForwardDiff.Dual` inputs (`Dual` is not an `AbstractFloat`); see Core rules for the `where {FT}` case.
2. **Constants typed from the wrong source**: A value typed by a hard-coded conversion (`Float64(x)`) or by an unrelated source (`eltype(params)`) does not pick up the `Dual` type of the arguments. Write `one(x)` / `zero(x)` rather than `FT(1)` / `FT(0)`.
3. **Mutation**: In-place modification of arrays or mutable structs can break reverse-mode AD (Enzyme). Prefer pure functions that return new values.

## Self-correction

If this guide is discovered to be stale or missing a pattern, update it.
