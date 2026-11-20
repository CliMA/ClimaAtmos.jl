# Numerical Robustness Guide

This guide covers safe numerical idioms for avoiding NaN, Inf, and DivideError in GPU kernels and AD-compatible code.

## 1. Denominator regularization

A division whose denominator may approach zero can be handled in several ways. The choice depends on:

- whether `x → 0` is physically valid behavior or signals a bug,
- whether downstream code can absorb `Inf` (or a large finite number),
- whether you want to bound the magnitude or expose the invalid state.

Float division does not raise an error: `a / 0.0` returns `Inf` (or `NaN` if `a` is also zero), and IEEE arithmetic propagates that through subsequent operations. Allowing `Inf` through is acceptable when a downstream `ifelse` or clamp discards the singular branch, or when the later operations are well defined at `Inf`. IEEE arithmetic does not take limits (`Inf * 0 = NaN`, `Inf - Inf = NaN`), so a later multiplication by a vanishing quantity produces `NaN` where algebra would give a finite limit.

```julia
# Sometimes fine: downstream must tolerate Inf and any NaN cascade
ratio = a / x
```

When `a/x → ∞` is physically defined but you want a bounded result for stability or discretization reasons, regularize with a *physically meaningful* floor `δ` (e.g. a minimum mass mixing ratio or cloud fraction):

```julia
ratio = a / max(x, δ)
```

The `max(x, eps(FT))` idiom is a soft NaN guard for the case where `x ≥ 0` is an invariant of valid state and the guard only absorbs round-off:

```julia
FT = eltype(x)
ratio = a / max(x, eps(FT))
```

Two hazards of this pattern:

- The guard maps negative `x` and small positive `x` to the same value, `eps(FT)`. When `x ≥ 0` is an invariant, this is a benign round-off correction. When `x < 0` comes from an upstream bug, the guard hides it and returns `≈ a · 8.4e6` for Float32.
- `eps(FT)` is far below typical physical scales, so the guard fires only within round-off of zero. If it fires on values that are not round-off noise, there is an upstream bug that the guard is hiding.

For signed denominators that should not be zero, `copysign(eps(FT), x)` preserves the sign of `x` while pushing the magnitude away from zero by `eps(FT)`:

```julia
safe_denom = x + copysign(eps(FT), x)
ratio = a / safe_denom
```

The regularized result is discontinuous at `x = 0`, jumping from `≈ -a/eps(FT)` to `≈ +a/eps(FT)` as `x` crosses zero, and that magnitude may far exceed any physically meaningful value. Where a sign-preserving near-infinite result is the intended behavior, this is the right idiom; otherwise, handle the zero crossing with an explicit branch.

If `x ≤ 0` is an invalid state, a defined sentinel is more informative than the arbitrary finite value the floors above produce; branch explicitly:

```julia
ratio = ifelse(x > δ, a / x, sentinel)
```

## 2. Safe inputs to transcendental functions

`log(x)` and `sqrt(x)` are undefined for strictly negative real `x`. On CPU, Julia raises `DomainError`; inside a GPU kernel the error cannot be caught and typically aborts the launch with an opaque message. Model state variables that should be non-negative (mixing ratios, cloud fractions, mass densities) routinely pick up small negative values from round-off, advection limiters, or explicit-step overshoots. A clip guard prevents the kernel error:

```julia
# ❌ Bad: DomainError on CPU / kernel crash or silent NaN on GPU when x < 0
result = log(x)
result = sqrt(x)

# ✅ Preferred: clamp round-off-level negatives before the call
safe_x = max(x, zero(x))   # or max(x, δ) for a physics-chosen δ
result = log(safe_x)
result = sqrt(safe_x)
```

Notes:

- `log(0)` is `-Inf` and `sqrt(0)` is `0`; both are valid IEEE values and propagate through float arithmetic. If `-Inf` is unacceptable downstream, lift the floor from `zero(x)` to a physics-chosen `δ` (see §1).
- `NaN` inputs need no guard here: `log(NaN) = sqrt(NaN) = NaN`, and the bug that produced the `NaN` is upstream.
- The clip absorbs *round-off-level* negative values. Inputs clipped from magnitudes far above round-off indicate an upstream bug that the clip is hiding.

When `log`/`sqrt`/division appears inside an `ifelse`, the guard goes *before* the `ifelse` because both branches are always evaluated. See [SDP 17](../code-quality/software_design_patterns.md), [GPU Performance Guide §1](gpu_performance.md), and the [Branchless Code Guide §3](branchless_code.md).

## 3. AD-compatible clamping

For zero-clamping, the canonical CliMA idiom is `max(zero(x), x)` (exported as `CloudMicrophysics.Utilities.clamp_to_nonneg`). See [ad_compatibility.md](ad_compatibility.md) for the full pattern and Dual-number rationale.

## 4. Conservation invariants

Mass, energy, and tracer conservation are verified at integration scale, not in unit tests. When changing a tendency, source term, or limiter, name the conservation test that should catch a bug. If no test exists, add one or flag the gap. Consult the repo-specific guide for the location of conservation tests and CI jobs.

## 5. Avoid `@assert` for runtime checks inside kernels

Use `error("static message")` instead. See [SDP 11](../code-quality/software_design_patterns.md).

## Self-correction

If this guide is discovered to be stale or missing a pattern, update it.
