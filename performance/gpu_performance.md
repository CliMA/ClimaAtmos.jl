# GPU Performance Guide

This guide covers patterns and pitfalls for writing high-performance, GPU-compatible Julia code in CliMA. All rules apply to any code that executes inside a `@.` broadcast kernel or a `ClimaCore` column operation.

## What counts as a "kernel" or "hot path"

The rules below apply whenever the surrounding code is a *kernel* or runs inside a *hot path*. The definitions:

- **Kernel**: the right-hand side of a `@.` broadcast, the body of a `ClimaCore.lazy()` expression, the closure passed to `Operators.column_integral_*`, `Operators.column_reduce!`, or `Fields.bycolumn`, and any function transitively `@inline`d into one of the above.
- **Hot path**: any function called once per timestep (or once per Runge–Kutta stage) for every column or every grid point. In model repos this includes tendency functions, precomputed-quantity setters, and Jacobian-update functions. In library repos (e.g. Thermodynamics.jl, CloudMicrophysics.jl), it includes any function designed to be called from a broadcast kernel.

If a function is called in either context, treat every rule in this file and in [software_design_patterns.md](../code-quality/software_design_patterns.md) as binding.

## 1. SIMT and thread divergence

On GPU architectures (CUDA, ROCm), threads are grouped into warps (typically 32 threads). All threads in a warp execute the same instruction in lockstep. When a data-dependent `if/else` branch causes different threads to take different paths, the hardware must execute both branches sequentially: this is **thread divergence** and it can halve (or worse) throughput.

### The remedy: `ifelse`

Use `ifelse(cond, a, b)` to remove the divergent branch predicate. See [SDP 17](../code-quality/software_design_patterns.md) for the canonical pattern, and the [Branchless Code Guide](branchless_code.md) for the broader discipline: why rare branches are not rare at 10⁶+ grid points, physical case splits, and fixed-iteration solvers.

**Underlying principle**: `ifelse` is an ordinary function call. Both `a` and `b` are evaluated *on every thread* before the call, and `ifelse` selects which value to return. It removes the warp-divergent predicate and leaves the work of both arms in place. When one arm is much more expensive than the other, a divergent `if/else` can be the better trade; measure, and see [Branchless Code Guide §6](branchless_code.md) for when a branch is acceptable.

Three consequences follow:

- **Guard mathematically invalid operations** (`log` of a non-positive, `sqrt` of a negative, division by a possibly-zero denominator) *before* the `ifelse`. [Numerical Robustness §1–2](numerical_robustness.md) covers how to choose the floor (in general it is not `eps(FT)`).
- **No `begin...end` blocks in arguments.** Statements inside `begin...end` run unconditionally.
- **No recursion through `ifelse`.** Use a plain `if` for recursion; see [Branchless Code Guide §6](branchless_code.md).

```julia
# ❌ Bad: log(x) executes even when x ≤ 0; the begin-block runs unconditionally
result = ifelse(x > 0,
    begin
        y = log(x)  # NaN when x ≤ 0
        y + 1
    end,
    zero(x)
)

# ✅ Preferred: pre-compute safely, then select. The wrong log_term for x ≤ 0
# is discarded by the ifelse, so max(x, eps) is acceptable here even though
# it would not be a safe NaN-guard on its own (see Numerical Robustness §1).
safe_x = max(x, eps(eltype(x)))
log_term = log(safe_x) + one(x)
result = ifelse(x > zero(x), log_term, zero(x))
```

## 2. Functors over closures

Closures that capture local variables produce heap allocations ("boxed variables") and may trigger `InvalidIRError: unsupported dynamic function invocation` on GPU. Replace them with callable structs (functors). See [SDP 18](../code-quality/software_design_patterns.md) for the canonical pattern and the closure-vs-functor cost comparison.

## 3. `lazy()` broadcast fusion

`ClimaCore.lazy()` creates a lazy broadcast object that represents an operation without materializing a temporary `Field`. Multiple lazy expressions fuse into a single GPU kernel when assigned to a terminal field.

### When to use `lazy()`

`lazy` is exported by the [`LazyBroadcast.jl`](https://github.com/CliMA/LazyBroadcast.jl) package (`import LazyBroadcast: lazy`) and is re-exported through `ClimaCore`. Use `@. lazy(expr)` for any intermediate quantity that a later broadcast reads; no temporary `Field` is allocated.

```julia
# ✅ Lazy: no temporary Field allocated
ᶜT = @. lazy(TD.air_temperature(thp, TD.ρe(), Y.c.ρe / Y.c.ρ,
                                Y.c.ρq_tot / Y.c.ρ, Y.c.ρq_liq / Y.c.ρ,
                                Y.c.ρq_ice / Y.c.ρ))
result = @. lazy(physics_func(ᶜT, Y.c.ρ))
@. output_field = result  # terminal: fuses everything into one kernel
```

### NamedTuple field-access pitfall

A `lazy()` wrapper returns a `Broadcasted` object, not a `NamedTuple`. Accessing `.field_name` on it outside a fused broadcast fails with `ERROR: type Broadcasted has no field X`.

```julia
# ❌ FAILS: lazy object is not a NamedTuple
limited = @. lazy(limit_tendencies(A, B, C))
@. Yₜ.c.ρq_liq += limited.Sqₗᵐ  # ERROR
```

### Materialization pattern

To extract multiple fields from a function's `NamedTuple` result, remove `lazy()` and materialize into a scratch field from the cache, allocated once during model construction.

```julia
# ❌ Allocates a new Field every timestep
limited_field = @. limit_tendencies(A, B, C)

# ✅ Write into a pre-allocated cache field: zero allocations per timestep
@. p.scratch.ᶜlimited = limit_tendencies(A, B, C)
@. Yₜ.c.ρq_liq += p.scratch.ᶜlimited.Sqₗᵐ
@. Yₜ.c.ρq_ice += p.scratch.ᶜlimited.Sqᵢᵐ
```

**General rule**: every `Field` computed during timestepping (in a tendency function, a callback, or any other per-timestep code) is pre-allocated in the cache, typically in `src/cache/`, which is built once during model construction.

### Multi-field updates

ClimaCore's broadcast machinery does not support a tuple of `Field`s on the LHS: `@. (Yₜ.c.ρq_liq, Yₜ.c.ρq_ice) += f(...)` fails in `check_broadcast_shape` / `copyto!`. Use the scratch-field pattern from the previous section: materialize the multi-field result into a pre-allocated `NamedTuple`-of-`Field`s in the cache, then issue one `@.` per target. The function runs once; each extra `@.` reads the stored field.

## 4. `@.` broadcast rules

### Dollar interpolation for non-field arguments

Use `$expr` to prevent the `@.` macro from broadcasting over a subexpression, in particular singleton dispatch types and computed scalars.

```julia
# ✅ Singleton escaped from broadcast
@. result = physics_func(Field_A, $(GridMeanSGS()))
```

### Parameter extraction

Extract non-`Field` arguments to local variables before the `@.` block, and do not use `Ref()` as a scalar escape; see [SDP 20](../code-quality/software_design_patterns.md) for the rule and rationale.

## 5. Register pressure and function size

Large functions (roughly > 200–300 lines) may exceed the Julia compiler's inlining budget. When this happens, the broadcast kernels inside the function stay un-inlined and each broadcast heap-allocates.

**Solution**: extract logical blocks into smaller `@inline` helper functions. A small parent function stays within the compiler's inlining heuristics, so all broadcasts fuse.

## 6. Fixed iteration solvers

Convergence-based loops (`while err > tol`) cause thread divergence when different threads converge at different rates. Where the physics allows it, prefer a fixed number of iterations, chosen by physical adequacy and validated with an offline test. See the [Branchless Code Guide §4–5](branchless_code.md) and [SDP 19](../code-quality/software_design_patterns.md).

## 7. GPU-safe error handling

Use `error("static message")` instead of `@assert`, and do not interpolate runtime variables into error strings inside kernels. See [SDP 11](../code-quality/software_design_patterns.md).

## 8. `isbits` requirement

Anything passed into a GPU kernel must be `isbits` *after device adaptation*; the host-side object need not be. Many ClimaCore objects (`Field`, `Space`, and anything wrapping a `CuArray`) hold `Array`/`CuArray` fields and so are not `isbits` on the host; they become `isbits` after `Adapt.adapt(CUDA.KernelAdaptor(), x)` (equivalently `CUDA.cudaconvert(x)`) rewrites those fields into device-side `CuDeviceArray`s and similar.

```julia
julia> a = fill(0.0f0, axes(Y.c));   # ClimaCore Field
julia> isbits(a)
false
julia> isbits(CUDA.cudaconvert(a))
true
```

Verify with the post-adapt check:

```julia
@assert isbits(CUDA.cudaconvert(MyStruct(...)))
```

If the post-adapt check returns `false`, check for:

- `Vector` or `Array` fields that are neither backed by a `CuArray` nor covered by an `Adapt.adapt_structure` method (use `SVector`/`Tuple`, or add an `Adapt` rule that rewrites the field to a device-side equivalent)
- `String` fields
- `Function` fields without a type parameter (use `struct A{F <: Function}; f::F; end`)
- `mutable struct` (use an immutable struct for data passed into kernels; [SDP 8](../code-quality/software_design_patterns.md) covers where `mutable struct` is acceptable)

When defining a new struct that wraps device-resident arrays, add an `Adapt.adapt_structure(to, x::MyStruct) = MyStruct(adapt(to, x.field1), ...)` method so the post-adapt object becomes `isbits`.

Avoid passing a `DataType` (e.g. `Float64`, or its alias `FT`) directly into a broadcast kernel; on some Julia versions this fails the `isbits` check.

## 9. Allocation verification

After implementing or modifying hot-path code, verify zero allocations with the warm-up + `@allocated == 0` regression-test pattern in [allocation_debugging.md §1](allocation_debugging.md). The `perf/` benchmarks run outside the unit-test CI, so this check happens at review time (see [ecosystem_conventions.md §5](../architecture/ecosystem_conventions.md)).

## 10. Elementary functions: exponents and roots

Transcendental functions dominate the cost of many physics kernels (size distributions, terminal velocities), and the general power operator `x^y` is one of the most expensive. Choosing the right form for a fractional power often speeds a kernel up several-fold: removing rational exponents alone gave ~7x in a CloudMicrophysics 2-moment kernel.

**Underlying principle**: the cost of `x^y` depends on the *type of the exponent*.

- **Floating-point exponent** (`x^y`, `x^2.0f0`, `x^0.5f0`): computed as roughly `exp(y * log(x))`, two transcendentals plus special-case handling. Expensive.
- **Integer-literal exponent** (`x^2`, `x^3`): non-negative integer literals dispatch to `Base.literal_pow`, which is plain multiplies (`x*x`, `x*x*x`). Negative integer literals (`x^-2`) take the general integer-exponent path (`inv` + multiplies), which is equally cheap. No transcendental either way.
- **Rational-literal exponent** (`x^(2//3)`): the worst case on GPU. The GPU compiler folds it only sometimes. When it is not folded, each thread constructs a 16-byte `Rational{Int64}` and runs a 64-bit integer `gcd` (with overflow checks) to normalize it before computing the power; 64-bit integer division is emulated on the GPU and the `gcd` loop diverges. When it is folded, it still reduces to a floating-point `pow`.

`sqrt` and `cbrt` map to dedicated, much cheaper routines than `pow`, so express fractional powers through them.

The rules:

1. **Never put a `Rational` literal in an exponent inside a kernel.** Not `x^(2//3)`, not `x^(-1//2)`, not `x^(3//2)`.
2. **Rewrite a fractional power as a root composed with an integer-literal power:** `x^(p//q) == root_q(x)^p`, using `sqrt` for `q = 2` and `cbrt` for `q = 3`, with an integer *literal* for `p`.
3. **Take the root first.** Prefer `cbrt(x)^2` over `cbrt(x^2)`: the root first keeps the intermediate in range, whereas for large or small `x` squaring first overflows or underflows to `Inf`/`0`. Square-first is marginally more accurate and is acceptable for moderate compile-time constants, where overflow is impossible.
4. **The exponent must be a literal, not a variable.** Only a written literal hits `literal_pow`; a *float* literal such as `x^2.0f0` is a `pow`, so write `x^2`. If a moment order or similar must be a variable, make it an `Int`, never a `Rational`.

| Instead of | Write |
|---|---|
| `x^(1//2)`, `x^0.5f0` | `sqrt(x)` |
| `x^(3//2)` | `x * sqrt(x)` (one multiply; `sqrt(x)^3` would be two) |
| `x^(1//3)` | `cbrt(x)` |
| `x^(2//3)` | `cbrt(x)^2` |
| `x^(-2//3)` | `cbrt(x)^(-2)` |
| `x^(1//6)` | `cbrt(sqrt(x))` |

```julia
# ❌ Bad: rational exponent -> Rational{Int64} + 64-bit gcd per thread, then a pow
v = a * D^(2//3)

# ✅ Preferred: one hardware cbrt plus an integer-literal square (literal_pow)
v = a * cbrt(D)^2
```

A constant fractional power of compile-time parameters is best precomputed once, on the host, into a parameter or constructor field so it never appears in the kernel. A Float32 hazard: `Rational{Int64}` is a 64-bit type, so `x::Float32 ^ (2//3)` promotes the result to `Float64`, breaking Float32 kernels (see [type_stability.md §1](type_stability.md) for the general Float32-pollution checklist). See [Numerical Robustness §1–2](numerical_robustness.md) for guarding `sqrt` of a possibly-negative argument (`cbrt` accepts negatives, `sqrt` does not).

## 11. Other patterns worth checking

Two further patterns to consider when looking for more performance:

- **Deduplicating a repeated pure computation across sibling functions.** If several functions each recompute the same expensive quantity from the same inputs, compute it once and pass it through as a default-valued optional argument; the compiler does not reliably eliminate the common subexpression across function-call boundaries.
- **Reducing quadrature order (or other numerical-integration resolution).** Like a fixed iteration count ([Branchless Code Guide §§4–5](branchless_code.md)), a reduced order needs its own offline convergence study, checked against integrand smoothness: a cusp or kink can cap achievable accuracy independent of node count. [review.md §1](../workflow/review.md) lists what that study must show.

## Self-correction

If this guide is discovered to be stale or missing a pattern, update it.
