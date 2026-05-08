# Numerical Robustness Guide

This guide covers safe numerical idioms for avoiding NaN, Inf, and DivideError in GPU kernels and AD-compatible code.

## 1. Denominator regularization

When dividing by quantities that may approach zero, add a regularization guard to prevent `DivideError` or `NaN`.

```julia
# ❌ DivideError or NaN when x → 0
ratio = a / x

# ✅ Regularised
FT = eltype(x)
ratio = a / max(x, eps(FT))
```

For quantities that may be positive or negative but should not be zero:

```julia
safe_denom = x + copysign(eps(eltype(x)), x)
ratio = a / safe_denom
```

## 2. Safe inputs to transcendental functions

Guard inputs to `log`, `sqrt`, and non-integer `^` against domains that produce NaN or errors:

```julia
# ❌ NaN when x ≤ 0
result = log(x)
result = sqrt(x)

# ✅ Guarded
safe_x = max(x, eps(eltype(x)))
result = log(safe_x)
result = sqrt(safe_x)
```

When used inside `ifelse`, the guard must be applied **before** the `ifelse` call because both branches are always evaluated. See [SDP 17](software_design_patterns.md) and [GPU Performance Guide](gpu_performance.md).

## 3. AD-compatible clamping

Standard `clamp(x, low, high)` is generally safe for most uses. For zero-clamping where gradient information must be preserved through the zero boundary, use this idiom:

```julia
@inline clamp_to_nonneg(x) = ifelse(x < zero(x), zero(x) * x, x)
```

The `zero(x) * x` idiom ensures the result has the same type (including Dual partials for AD) as `x`, while the value is zeroed.

## 4. Conservation invariants

Mass, energy, and tracer conservation are verified at integration scale, not in unit tests. When changing a tendency, source term, or limiter, name the conservation test that should catch a bug — and if no test exists, add one or flag the gap. In ClimaAtmos this typically means a test under `test/conservation/` or a Buildkite job that monitors integrated diagnostics.

## 5. Avoid `@assert` for runtime checks inside kernels

Use `error("message")` instead of `@assert`. Do not capture runtime variables in the error message. See [SDP 11](software_design_patterns.md).

```julia
# ❌ @assert allocates; string interpolation triggers dynamic dispatch
@assert x > 0 "x must be positive, got $x"

# ✅ Static error message
x > 0 || error("x must be positive")
```

## Self-correction

If this guide is discovered to be stale or missing a pattern, update it.
