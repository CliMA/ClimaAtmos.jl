# Numerical Robustness Guide

This guide covers safe numerical idioms for avoiding NaN, Inf, and DivideError in GPU kernels and AD-compatible code.

## 1. Denominator regularization

Dividing by quantities that may approach zero has several reasonable handlings. The right choice depends on:

- whether `x → 0` is physically valid behavior or signals a bug,
- whether downstream code can absorb `Inf` (or a large finite number) correctly,
- whether you want to bound the magnitude or surface the invalid state.

Float division does not raise an error: `a / 0.0` returns `Inf` (or `NaN` if `a` is also zero), and IEEE arithmetic propagates that through subsequent operations. You can sometimes deliberately allow `Inf` through, but only if downstream consumers are *known* to handle it — IEEE arithmetic does not take limits (`Inf * 0 = NaN`, `Inf - Inf = NaN`), so any later multiplication by a vanishing quantity silently produces `NaN` rather than the algebraic cancellation a mathematician would expect. Cases where allowing `Inf` is fine include a downstream `ifelse` or clamp that discards the singular branch, or operations where `Inf` results in well-defined behavior.

```julia
# Sometimes fine: downstream must tolerate Inf and any NaN cascade
ratio = a / x
```

When `a/x → ∞` is physically defined but you want a bounded result for stability or discretisation reasons, regularise with a *physically meaningful* floor `δ` (e.g. a minimum mass mixing ratio or cloud fraction):

```julia
ratio = a / max(x, δ)
```

The compact `max(x, eps(FT))` idiom is appropriate as a soft NaN-guard when `x ≥ 0` is an invariant in valid state and the guard exists only to absorb round-off:

```julia
FT = eltype(x)
ratio = a / max(x, eps(FT))
```

Things to watch with this last pattern:

- The guard treats negative `x` and small positive `x` identically (both map to `eps(FT)`). If `x ≥ 0` is genuinely invariant, this is a benign round-off correction. If `x < 0` arises from an upstream bug, the guard silently emits `≈ a · 8.4e6` for Float32 rather than surfacing the bad state.
- `eps(FT)` is often much smaller than typical physical scales, so the guard only fires within round-off of zero. If it fires far from round-off, there is a separate bug that this guard is hiding.

For signed denominators that should not be zero, `copysign(eps(FT), x)` preserves the sign of `x` while pushing the magnitude away from zero by `eps(FT)`:

```julia
safe_denom = x + copysign(eps(FT), x)
ratio = a / safe_denom
```

Use with care: the regularised result is discontinuous at `x = 0`, jumping from `≈ -a/eps(FT)` to `≈ +a/eps(FT)` as `x` crosses zero, and the magnitude `≈ a/eps(FT)` may be much larger than any physically meaningful value. Sometimes that sign-preserving "noisy ±Inf" behavior is exactly what you want; sometimes the right behavior is to surface the zero crossing with an explicit branch.

If `x ≤ 0` indicates an invalid state and you want to substitute a defined sentinel rather than a finite-but-arbitrary value, branch explicitly:

```julia
ratio = ifelse(x > δ, a / x, sentinel)
```

## 2. Safe inputs to transcendental functions

`log(x)` and `sqrt(x)` are undefined for strictly negative real `x`. On CPU, Julia raises `DomainError`; inside a GPU kernel that error cannot be caught and usually crashes the launch with hard-to-interpret error messages. Model state variables that should be non-negative — mixing ratios, cloud fractions, mass densities — routinely pick up small-magnitude negative values from round-off, advection limiters, or explicit-step over-shoots. A clip guard prevents the kernel error:

```julia
# ❌ Bad: DomainError on CPU / kernel crash or silent NaN on GPU when x < 0
result = log(x)
result = sqrt(x)

# ✅ Preferred: clamp round-off-level negatives before the call
safe_x = max(x, zero(x))   # or max(x, δ) for a physics-chosen δ
result = log(safe_x)
result = sqrt(safe_x)
```

A few practical notes:

- `log(0)` is `-Inf` and `sqrt(0)` is `0` — both propagate through float arithmetic without erroring or crashing a kernel. If `-Inf` is unacceptable downstream, lift the floor from `zero(x)` to a physics-chosen `δ` (see §1).
- `NaN` inputs do not need guarding at this layer: `log(NaN) = sqrt(NaN) = NaN`, propagating silently. The bug producing the `NaN` is upstream.
- The intent of the clip is to absorb *round-off-level* negative values. If you are routinely clipping inputs whose magnitude is far above round-off, there is an upstream bug and the clip is hiding it.

When `log`/`sqrt`/division appears inside an `ifelse`, the guard goes *before* the `ifelse` because both branches are always evaluated. See [SDP 17](../code-quality/software_design_patterns.md) and [GPU Performance Guide §1](gpu_performance.md).

## 3. AD-compatible clamping

For zero-clamping, the canonical CliMA idiom is `max(zero(x), x)` (exported as `CloudMicrophysics.Utilities.clamp_to_nonneg`). See [ad_compatibility.md](ad_compatibility.md) for the full pattern and Dual-number rationale.

## 4. Conservation invariants

Mass, energy, and tracer conservation are verified at integration scale, not in unit tests. When changing a tendency, source term, or limiter, name the conservation test that should catch a bug — and if no test exists, add one or flag the gap. Consult the repo-specific guide for the location of conservation tests and CI jobs.

## 5. Avoid `@assert` for runtime checks inside kernels

Use `error("static message")` instead. See [SDP 11](../code-quality/software_design_patterns.md).

## Self-correction

If this guide is discovered to be stale or missing a pattern, update it.
