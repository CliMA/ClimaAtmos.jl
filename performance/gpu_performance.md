# GPU Performance Guide

This guide covers patterns and pitfalls for writing high-performance, GPU-compatible Julia code in CliMA. All rules apply to any code that executes inside a `@.` broadcast kernel or a `ClimaCore` column operation.

## What counts as a "kernel" or "hot path"

The rules below apply whenever the surrounding code is a *kernel* or runs inside a *hot path*. Define both concretely:

- **Kernel**: the right-hand side of a `@.` broadcast, the body of a `ClimaCore.lazy()` expression, the closure passed to `Operators.column_integral_*`, `Operators.column_reduce!`, or `Fields.bycolumn`, and any function transitively `@inline`d into one of the above.
- **Hot path**: any function called once per timestep (or once per Runge–Kutta stage) for every column or every grid point. In model repos this includes tendency functions, precomputed-quantity setters, and Jacobian-update functions. In library repos (e.g. Thermodynamics.jl, CloudMicrophysics.jl), it includes any function that is designed to be called from a broadcast kernel.

If a function is called in either context, treat every rule in this file and in [software_design_patterns.md](../code-quality/software_design_patterns.md) as binding.

## 1. SIMT and thread divergence

On GPU architectures (CUDA, ROCm), threads are grouped into warps (typically 32 threads). All threads in a warp execute the same instruction in lockstep. When a data-dependent `if/else` branch causes different threads to take different paths, the hardware must execute both branches sequentially — this is **thread divergence** and it can halve (or worse) throughput.

### The remedy: `ifelse`

Use `ifelse(cond, a, b)` to remove the divergent branch predicate. See [SDP 17](../code-quality/software_design_patterns.md) for the canonical pattern.

**Underlying principle**: `ifelse` is an ordinary function call. Both `a` and `b` are evaluated to a value *on every thread* before the call; `ifelse` only selects which value to return. Using `ifelse` does **not** save the work of the un-taken branch — its purpose is to eliminate the warp-divergent predicate, not to skip computation. If one branch is significantly more expensive than the other, every thread still pays its cost; sometimes a divergent `if/else` is the better trade and you should measure.

Three consequences follow:

- **Guard mathematically invalid operations** (`log` of a non-positive, `sqrt` of a negative, division by a possibly-zero denominator) *before* the `ifelse`. See [Numerical Robustness §1–2](numerical_robustness.md) for how to choose the floor (it is not `eps(FT)` in general).
- **No `begin...end` blocks in arguments.** Statements inside `begin...end` run unconditionally — they are not "guarded" by the condition.
- **No recursion through `ifelse`.** Since both branches evaluate, using `ifelse` to choose between a base case and a recursive call produces unbounded recursion. Use a plain `if` for recursion.

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

`lazy` is exported by the [`LazyBroadcast.jl`](https://github.com/CliMA/LazyBroadcast.jl) package (`import LazyBroadcast: lazy`) and is re-exported through `ClimaCore`. Use `@. lazy(expr)` for any intermediate computed quantity that is consumed by a subsequent broadcast — it prevents heap allocation of a temporary `Field`.

```julia
# ✅ Lazy: no temporary Field allocated
ᶜT = @. lazy(TD.air_temperature(thp, TD.ρe(), Y.c.ρe / Y.c.ρ,
                                Y.c.ρq_tot / Y.c.ρ, Y.c.ρq_liq / Y.c.ρ,
                                Y.c.ρq_ice / Y.c.ρ))
result = @. lazy(physics_func(ᶜT, Y.c.ρ))
@. output_field = result  # terminal: fuses everything into one kernel
```

### NamedTuple field-access pitfall

A `lazy()` wrapper returns a `Broadcasted` object, not a real `NamedTuple`. Accessing `.field_name` on it outside a fused broadcast fails with `ERROR: type Broadcasted has no field X`.

```julia
# ❌ FAILS — lazy object is not a NamedTuple
limited = @. lazy(limit_tendencies(A, B, C))
@. Yₜ.c.ρq_liq += limited.Sqₗᵐ  # ERROR
```

### Materialization pattern

When you need to extract multiple fields from a function's `NamedTuple` result, remove `lazy()` to materialize into a pre-allocated `Field`. **Do not** create a new temporary field inline each timestep — this allocates on every call. Instead, use a scratch field from the cache that was allocated once during model construction.

```julia
# ❌ Allocates a new Field every timestep
limited_field = @. limit_tendencies(A, B, C)

# ✅ Write into a pre-allocated cache field — zero allocations per timestep
@. p.scratch.ᶜlimited = limit_tendencies(A, B, C)
@. Yₜ.c.ρq_liq += p.scratch.ᶜlimited.Sqₗᵐ
@. Yₜ.c.ρq_ice += p.scratch.ᶜlimited.Sqᵢᵐ
```

**General rule**: any `Field` that is computed inside a function called during timestepping must be pre-allocated in the cache (typically in `src/cache/`). The cache is built once during model construction. Never allocate new `Field`s inside tendency functions, callbacks, or any code that runs per-timestep.

### Multi-field updates

ClimaCore's broadcast machinery does not support a tuple of `Field`s on the LHS — `@. (Yₜ.c.ρq_liq, Yₜ.c.ρq_ice) += f(...)` fails in `check_broadcast_shape` / `copyto!`. Use the scratch-field pattern from the previous section: materialize the multi-field result into a pre-allocated `NamedTuple`-of-`Field`s in the cache, then issue one `@.` per target. The cost of repeating the broadcast is paid back because the RHS is just a field load, not a recomputation.

## 4. `@.` broadcast rules

### Dollar interpolation for non-field arguments

Use `$expr` to prevent the `@.` macro from broadcasting over a subexpression. This is essential for singleton dispatch types and computed scalars.

```julia
# ✅ Singleton escaped from broadcast
@. result = physics_func(Field_A, $(GridMeanSGS()))
```

### Do not use `Ref()` as a broadcast scalar escape

`Ref()` is not the standard broadcast-escape pattern in this codebase. Its use in `src/` is limited to mutable scalar boxes in callbacks and non-broadcast contexts. Prefer parameter extraction ([SDP 20](../code-quality/software_design_patterns.md)).

### Parameter extraction

Extract non-`Field` arguments to local variables before the `@.` block — see [SDP 20](../code-quality/software_design_patterns.md) for the rule and rationale.

## 5. Register pressure and function size

Large functions (roughly > 200–300 lines) may exceed the Julia compiler's inlining budget. When this happens, broadcast kernels inside the function are not inlined, causing heap allocations for each broadcast.

**Solution**: extract complex logical blocks into smaller `@inline` helper functions. Keeping the parent function small allows the compiler to stay within its heuristics threshold, ensuring all broadcasts are correctly fused.

## 6. Fixed iteration solvers (advisory)

Convergence-based loops (`while err > tol`) cause thread divergence when different threads converge at different rates. Where the physics allows it, prefer a fixed number of iterations. See [SDP 19](../code-quality/software_design_patterns.md).

## 7. GPU-safe error handling

Use `error("static message")` instead of `@assert`, and do not interpolate runtime variables into error strings inside kernels. See [SDP 11](../code-quality/software_design_patterns.md).

## 8. `isbits` requirement

Anything passed into a GPU kernel must be `isbits` *after device adaptation*. That is the actual contract — not that the host-side object is `isbits`. Many ClimaCore objects (notably `Field`, `Space`, and anything wrapping a `CuArray`) are deliberately not `isbits` on the host because they hold `Array`/`CuArray` payloads; they become `isbits` only after `Adapt.adapt(CUDA.KernelAdaptor(), x)` (equivalently `CUDA.cudaconvert(x)`) rewrites the array fields into device-side `CuDeviceArray`s and similar.

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

- `Vector` or `Array` fields that aren't backed by a `CuArray` and don't have an `Adapt.adapt_structure` method (use `SVector`/`Tuple`, or add an `Adapt` rule that rewrites the field to a device-side equivalent)
- `String` fields
- `Function` fields without a type parameter (use `struct A{F <: Function}; f::F; end`)
- `mutable struct` (prefer immutable structs for data passed into kernels; `mutable struct` is acceptable for infrastructure objects like grids and integrators that are never passed into kernels)

When defining a new struct that wraps device-resident arrays, add an `Adapt.adapt_structure(to, x::MyStruct) = MyStruct(adapt(to, x.field1), ...)` method so the post-adapt object becomes `isbits`.

Avoid using `DataTypes` (e.g. `Float64`) or their aliases (e.g `FT`) directly in broadcast kernels. This can cause `isbits` failures on different julia versions.

## 9. Allocation verification

After implementing or modifying hot-path code, verify zero allocations with the warm-up + `@allocated == 0` regression-test pattern documented in [allocation_debugging.md §1](allocation_debugging.md). Allocation benchmarks in `perf/` are not run automatically in CI, so allocation regressions must be caught at review time.

## Self-correction

If this guide is discovered to be stale or missing a pattern, update it.
