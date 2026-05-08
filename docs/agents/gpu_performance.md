# GPU Performance Guide

This guide covers patterns and pitfalls for writing high-performance, GPU-compatible Julia code in CliMA. All rules apply to any code that executes inside a `@.` broadcast kernel or a `ClimaCore` column operation.

## What counts as a "kernel" or "hot path"

The rules below apply whenever the surrounding code is a *kernel* or runs inside a *hot path*. Define both concretely:

- **Kernel**: the right-hand side of a `@.` broadcast, the body of a `ClimaCore.lazy()` expression, the closure passed to `Operators.column_integral_*`, `Operators.column_reduce!`, or `Fields.bycolumn`, and any function transitively `@inline`d into one of the above.
- **Hot path**: any function called once per timestep (or once per Runge–Kutta stage) for every column or every grid point. Concretely: tendency functions in `prognostic_equations/`, the entries of `parameterized_tendencies/`, the `set_*_precomputed_quantities!` family in `cache/`, the Jacobian-update functions, and any `@.` broadcast they execute.

If a function is called in either context, treat every rule in this file and in [software_design_patterns.md](software_design_patterns.md) as binding.

## 1. SIMT and thread divergence

On GPU architectures (CUDA, ROCm), threads are grouped into warps (typically 32 threads). All threads in a warp execute the same instruction in lockstep. When a data-dependent `if/else` branch causes different threads to take different paths, the hardware must execute both branches sequentially — this is **thread divergence** and it can halve (or worse) throughput.

### The remedy: `ifelse`

Use `ifelse(cond, a, b)` to compute both branches and select the result branchlessly. See [SDP 17](software_design_patterns.md) for the full pattern.

**Critical pitfalls of `ifelse`**:

- Both arguments are always evaluated. Guard mathematically invalid operations (`log`, `sqrt`, division) with `max(x, eps(x))` **before** passing them as arguments.
- Never use `begin...end` blocks inside `ifelse` arguments — side effects and complex blocks execute unconditionally.
- Do not use `ifelse` to select between a base case and a recursive call. Since both branches evaluate, this causes infinite recursion. Use a standard `if` statement for recursion.

```julia
# ❌ Pitfall: log(x) executes even when x ≤ 0; begin blocks always run
result = ifelse(x > 0,
    begin
        y = log(x)  # NaN when x ≤ 0
        y + 1
    end,
    zero(x)
)

# ✅ Pre-compute safely, select with ifelse
safe_x = max(x, eps(x))
log_term = log(safe_x) + one(x)
result = ifelse(x > zero(x), log_term, zero(x))
```

## 2. Functors over closures

Closures that capture local variables produce heap allocations ("boxed variables") and may trigger `InvalidIRError: unsupported dynamic function invocation` on GPU. Replace them with callable structs (functors). See [SDP 18](software_design_patterns.md).

Performance comparison (microphysics case study):

| Implementation | Allocations per grid point | GPU status |
|:---|:---|:---|
| Closure | ~1.1 KB | ❌ InvalidIRError |
| Functor | ~16 bytes (fixed overhead) | ✅ Optimized kernel |

Validation: after a warm-up call, `@allocated integrate(functor, data)` should return 0.

## 3. `lazy()` broadcast fusion

`ClimaCore.lazy()` creates a lazy broadcast object that represents an operation without materializing a temporary `Field`. Multiple lazy expressions fuse into a single GPU kernel when assigned to a terminal field.

### When to use `lazy()`

Use `@. lazy(expr)` for any intermediate computed quantity that is consumed by a subsequent broadcast. This prevents heap allocation of a temporary `Field`.

```julia
# ✅ Lazy: no temporary Field allocated
T = @. lazy(air_temperature(thp, ts))
result = @. lazy(physics_func(T, ρ))
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

### Tuple-Fusion pattern

For zero-allocation multi-field updates, have the physics function return a plain `Tuple` and use tuple assignment in the broadcast:

```julia
# ✅ Zero allocations — no intermediate Field
@inline function limited_tendency(ρ, args...)
    lim = compute_limits(args...)
    return (ρ * lim.Sqₗᵐ, ρ * lim.Sqᵢᵐ, ρ * lim.Sqᵣᵐ, ρ * lim.Sqₛᵐ)
end

@. (Yₜ.c.ρq_liq, Yₜ.c.ρq_ice, Yₜ.c.ρq_rai, Yₜ.c.ρq_sno) +=
    limited_tendency(Y.c.ρ, ...)
```

## 4. `@.` broadcast rules

### Dollar interpolation for non-field arguments

Use `$expr` to prevent the `@.` macro from broadcasting over a subexpression. This is essential for singleton dispatch types and computed scalars.

```julia
# ✅ Singleton escaped from broadcast
@. result = physics_func(Field_A, $(GridMeanSGS()))
```

### Do not use `Ref()` as a broadcast scalar escape

`Ref()` is not the standard broadcast-escape pattern in this codebase. Its use in `src/` is limited to mutable scalar boxes in callbacks and non-broadcast contexts. Prefer parameter extraction ([SDP 20](software_design_patterns.md)).

### Parameter extraction

Extract non-`Field` arguments to local variables before the `@.` block:

```julia
thp = p.params.thermodynamics_params
@. result = my_physics(thp, Y.c.T, Y.c.ρ)
```

## 5. Register pressure and function size

Large functions (roughly > 200–300 lines) may exceed the Julia compiler's inlining budget. When this happens, broadcast kernels inside the function are not inlined, causing heap allocations for each broadcast.

**Solution**: extract complex logical blocks into smaller `@inline` helper functions. Keeping the parent function small allows the compiler to stay within its heuristics threshold, ensuring all broadcasts are correctly fused.

## 6. Fixed iteration solvers (advisory)

Convergence-based loops (`while err > tol`) cause thread divergence when different threads converge at different rates. Where the physics allows it, prefer a fixed number of iterations. See [SDP 19](software_design_patterns.md).

## 7. GPU-safe error handling

- Use `error("message")`, not `@assert`. See [SDP 11](software_design_patterns.md).
- Do not interpolate runtime variables into error strings inside kernels. The string interpolation allocates and may trigger dynamic dispatch.

```julia
# ❌ String interpolation allocates
error("Invalid value: $x")

# ✅ Static message only
error("Invalid value encountered")
```

## 8. `isbits` requirement

Structs used inside GPU kernels must be `isbits` — meaning all fields are immutable, concrete, and contain no heap-allocated data (`Vector`, `String`, abstract `Function`).

Verify with:

```julia
@assert isbits(MyStruct(...))
```

If `isbits` returns `false`, check for:
- `Vector` or `Array` fields (use `SVector` or `Tuple` instead)
- `String` fields
- `Function` fields without a type parameter (use `struct A{F <: Function}; f::F; end`)
- `mutable struct` (use immutable structs)

## 9. Allocation verification workflow

After implementing or modifying hot-path code, verify zero allocations:

```julia
# Warm up (forces compilation)
set_tendencies!(Yₜ, Y, p, t)
# Assert zero allocations
@test (@allocated set_tendencies!(Yₜ, Y, p, t)) == 0
```

Allocation benchmarks in `perf/` are not run automatically in CI. Allocation regressions must be caught during review.

## Self-correction

If this guide is discovered to be stale or missing a pattern, update it.
