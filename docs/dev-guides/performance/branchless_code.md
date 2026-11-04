# Branchless Code Guide

This guide explains why CliMA kernels and hot paths should avoid data-dependent
branches, and how to write the common physics patterns (case splits and
iterative solves) without them. It is the canonical home for the
*branch-avoidance discipline*; the GPU hardware mechanics live in
[gpu_performance.md §1](gpu_performance.md), and the terse rule-of-thumb forms
are [SDP 17](../code-quality/software_design_patterns.md) and
[SDP 19](../code-quality/software_design_patterns.md).

The rules below apply inside a *kernel* or *hot path*; see
[gpu_performance.md](gpu_performance.md) for the definitions. Outside
those contexts (setup, configuration, IO, once-per-run code), branch freely;
readability wins there.

## 1. Why rare branches are not rare at climate-model scale

On GPUs, a data-dependent `if/else` where threads in the same warp take
different paths forces the hardware to run *both* paths sequentially
(**thread divergence**), which can halve throughput or worse. The SIMT
mechanics are in [gpu_performance.md §1](gpu_performance.md).

The scale of a climate model makes this worse. A CliMA
simulation evaluates each pointwise function at **10⁶–10⁹ grid points**, every
timestep (and every Runge–Kutta stage), for **thousands to millions** of steps.
A branch that guards a "rare" special case (a parcel that is exactly saturated,
a point that just crossed freezing, a near-zero denominator) is, in aggregate,
*almost never rare*: at millions of points per stage, nearly every warp
contains at least one point in the special case at nearly every step, so the
"rarely taken" branch pays its divergence cost on **every step**.

The practical consequence: **inside a kernel, cost a data-dependent branch as if
it is always taken**, and default to branchless.

## 2. The branch-avoidance hierarchy

Prefer the earliest technique that applies:

1. **Eliminate the choice at compile time with dispatch.** A choice between
   *models*, *schemes*, or *present/absent* components is known
   before the kernel launches. Encode it as a singleton type and dispatch on it,
   so no branch reaches the kernel at all. See [SDP 1](../code-quality/software_design_patterns.md)
   (structs over strings) and [type_stability.md §3](type_stability.md)
   (splitting `Union{T, Nothing}` dispatch). The dead path is never compiled
   into the kernel.
2. **Select a value with `ifelse`.** For a data-dependent choice
   between two computed values, use `ifelse(cond, a, b)`; `if/else` and the
   ternary `cond ? a : b` both branch. See [SDP 17](../code-quality/software_design_patterns.md)
   and [gpu_performance.md §1](gpu_performance.md).
3. **Evaluate both physical cases, then select** (§3 below).
4. **Fix the iteration count of pointwise solvers** (§4 below).

## 3. Evaluate both cases, then select

Physics is full of case splits: above vs. below freezing, saturated vs.
subsaturated, laminar vs. turbulent. The instinct is `if T > T_freeze … else …`.
In a kernel, evaluate **both** cases unconditionally and combine them with
`ifelse`:

```julia
# ❌ Divergent: warps straddling the freezing line run both arms serially
if T > T_freeze
    rate = liquid_process(T, q)
else
    rate = ice_process(T, q)
end

# ✅ Branchless: both arms always evaluated; ifelse selects
rate_liquid = liquid_process(T, q)
rate_ice    = ice_process(T, q)
rate = ifelse(T > T_freeze, rate_liquid, rate_ice)
```

Because `ifelse` evaluates both arguments on every thread, this trades a divergent
branch for the cost of the *un-taken* arm. In a warp that straddles the split the
divergent form runs both arms anyway, so the trade costs nothing there and one
arm elsewhere. If one arm is much more expensive than the other, measure
(see §6).

Three things to keep correct:

- **Guard invalid math before the `ifelse`, not inside it.** Both arms always
  run, so a `log`, `sqrt`, or division that is valid in only one case must be
  made safe for the other (e.g. `log(max(x, δ))`). See
  [numerical_robustness.md §1–2](numerical_robustness.md) and the worked example
  in [gpu_performance.md §1](gpu_performance.md).
- **Prefer a smooth transition over a sharp switch** where physically
  defensible. A short linear or power-law ramp across the split avoids a
  discontinuity that can stall a solver or break differentiability.
  `Thermodynamics.liquid_fraction` ramps the liquid fraction linearly over a
  ~0.2 K band just below the freezing point.
- **Combine pointwise conditions with bitwise `&` / `|`, not `&&` / `||`.** The
  short-circuiting operators are themselves control-flow branches; the bitwise
  operators evaluate both operands and produce the same boolean. From
  `Thermodynamics.liquid_fraction_ramp`:

  ```julia
  supercooled_liquid = (T ≤ Tᶠ) & (T > Tⁱ)   # not (T ≤ Tᶠ) && (T > Tⁱ)
  return ifelse(T > Tᶠ, one(T), ifelse(supercooled_liquid, λᵖ, zero(T)))
  ```

## 4. Fix the iteration count of pointwise solvers

A convergence-based loop is the most damaging branch in a kernel:

```julia
# ❌ Divergent: each point needs a different number of iterations; the whole
#    warp waits for the slowest point, and the `break` is a data-dependent branch
T = T_guess
while abs(residual(T)) > tol
    T -= residual(T) / dresidual_dT(T)
end
```

Replace it with a **fixed number of iterations**, identical for every point, so
every thread runs the same straight-line code:

```julia
# ✅ Branchless: same iteration count on every thread, no convergence test
T = T_guess
for _ in 1:maxiter          # maxiter is a fixed, small constant
    T -= residual(T) / dresidual_dT(T)
end
```

The canonical CliMA example is `Thermodynamics.saturation_adjustment`. Its
GPU-default path, `saturation_adjustment_fixed_iters`, is a safeguarded Newton
loop with a fixed `maxiter = 2` and **no stopping test**: every guard in it is
an `ifelse`, `clamp`, or `max`, so every lane runs the same instructions. The
`converged` flag it returns is computed after the loop from the size of the
last Newton increment; it is a diagnostic, not a loop exit, and the convenience
methods drop it:

```julia
# Thermodynamics.jl v1.3 src/saturation_adjustment.jl (abridged)
@inline function saturation_adjustment_fixed_iters(param_set, ::ρe, ρ, e_int, q_tot, maxiter)
    T_unsat   = air_temperature(param_set, e_int, q_tot)
    saturated = _is_saturated(param_set, T_unsat, ρ, q_tot)
    T  = max(T_unsat, T_positive_floor(eltype(param_set)))
    ΔT = T
    for _ in 1:maxiter
        e_val     = internal_energy_sat(param_set, T, ρ, q_tot, Val(false))
        de_int_dT = ∂e_int_∂T_sat_ρ(param_set, T, ρ, q_tot, Val(false))
        (T, ΔT)   = _newton_update(param_set, T, (e_int - e_val) / de_int_dT)  # clamp + max, no branch
    end
    (T, ΔT) = _select_solution(saturated, T, ΔT, T_unsat)   # ifelse between saturated and exact unsaturated
    (q_liq, q_ice) = condensate_partition(param_set, T, ρ, q_tot)
    return (; T, q_liq, q_ice, converged = _fixed_iters_converged(T, ΔT))
end
```

The general-purpose, convergence-checking solvers in
[RootSolvers.jl](https://github.com/CliMA/RootSolvers.jl) (with their line
searches, derivative-fallbacks, and `maxiters = 1000` ceilings) are appropriate
for offline or setup-time work, but their many branches make them a poor fit for
a per-point kernel.

## 5. Choose the count by physical adequacy, validated offline

A fixed iteration count is only safe if you *know* it is enough. Treat the count
as **data: fix it with an offline test, and document the accuracy it achieves**.

Two principles set the bar:

- **Adequate means physically adequate.** A pointwise solve in a climate
  simulation needs an answer accurate enough that the simulation is unaffected,
  which is far coarser than floating-point precision. For temperature,
  ~**0.1 K** is almost always sufficient. Targeting `eps(FT)` would cost extra
  iterations (and divergence, if convergence-gated) to compute digits no one
  uses.
- **The test must exercise the full envelope a climate run can produce.** Sweep
  the realistic (and edge-case) range of every input (temperature, pressure,
  humidity, including supersaturation and near-zero condensate), compare against
  a high-iteration or analytic reference, and pick the *smallest* count that
  meets the physical tolerance across that whole range.

`Thermodynamics.jl` is the model to copy. Its `maxiter = 2` default is justified
by two offline test files:

- `test/default_saturation_adjustment.jl` sweeps ~250 points across realistic
  atmospheric profiles and asserts that **>98 % of points** land within 0.1 K of
  the reference and the **max error stays under 0.5 K** (the test that pins the
  count at 2).
- `test/convergence_saturation_adjustment.jl` validates correctness against a
  high-iteration (`maxiter = 40`, tight-tolerance) reference solve across the
  same profiles, in both `Float32` and `Float64`.

The docstring then records the result and its range of validity: two
iterations hold the temperature error below 2e-3 K for first-guess-to-solution
gaps up to the ~4 K seen across the tested profiles, growing to a few tenths of
a K around a 10 K gap, where a larger `maxiter` is worth passing. When you
introduce or tune a fixed-iteration solver, ship the analogous test and state the
achieved tolerance and its valid range in the docstring.

Checklist for a fixed-iteration solver:

- [ ] Loop is `for _ in 1:maxiter` with a constant `maxiter`, no `break`, no
      convergence test in the hot path.
- [ ] An offline test sweeps the full physical envelope, including edge cases.
- [ ] The test compares against a high-iteration or analytic reference and
      asserts a *physically meaningful* tolerance.
- [ ] The test runs for both `Float32` and `Float64` (see
      [type_stability.md](type_stability.md)).
- [ ] The docstring states the achieved accuracy and the range over which it
      holds.

## 6. When a branch is acceptable

Branch avoidance is a strong default inside kernels, applied with judgment. A
data-dependent branch is reasonable when:

- The code runs outside a kernel or hot path (setup, IO, config, diagnostics
  run once per step on the host). Optimize those for clarity.
- The choice is resolvable at **compile time**: use dispatch (§2, item 1).
- One arm is **far more expensive** than the other and is rarely
  needed *per warp* (not merely per point). `ifelse` evaluates both arms, so when
  the expensive arm dominates, a divergent `if` that most warps skip entirely can
  win. Measure it.
- The branch implements **recursion**: `ifelse` cannot, because it evaluates
  both arms and would recurse unconditionally. Use a plain `if`.

When you keep a data-dependent branch in a kernel, leave a one-line comment
explaining why it is worth the divergence, so a future reader (or reviewer, per
[review.md](../workflow/review.md)) does not "fix" it.

## Self-correction

If this guide is discovered to be stale or missing a pattern, update it.
