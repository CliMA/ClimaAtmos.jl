# Software Design Patterns (SDPs)

This file is an agent-facing checklist for writing robust, maintainable, and GPU-compatible Julia code.

Unless explicitly instructed otherwise, treat all rules below as defaults.

## How to use this file

- Apply these patterns in new code.
- Prefer refactoring toward these patterns when touching existing code.
- If a rule must be broken, keep the exception narrow and document why.

## 1. Use structs over strings in tendency functions

Avoid string-based model dispatch in hot paths.

Bad:

```julia
struct Foo{T}
    model::T
end
function baz(f)
    if f.model == "ModelA"
        # do something ModelA-specific
    elseif f.model == "ModelB"
        # do something ModelB-specific
    end
end
f = Foo("ModelA")
baz(f)
```

Preferred:

```julia
struct ModelA end
struct ModelB end
struct Foo{T <: Union{ModelA, ModelB}}
    model::T
end
function baz(f)
    if f.model isa ModelA
        # do something ModelA-specific
    elseif f.model isa ModelB
        # do something ModelB-specific
    end
end
f = Foo(ModelA())
baz(f)
```

Exception:

- Strings are acceptable at initialization boundaries (for example, parsing user/config input), but convert to typed structs as early as possible.

## 2. Avoid `using` / `import` inside `src/` submodules

Inside `src/`, do not add local/module-internal `using` or `import` patterns like this:

```julia
module Foo

baz() = 1

module Bar
  using Foo: baz
  bing() = baz()
end

end
```

Prefer explicit qualification or project-established module patterns.

## 3. Do not use `Symbol`s in broadcasted expressions

Avoid symbol-based broadcast patterns. Use concrete, type-stable values/structures instead.

## 4. Do not use abstract types in struct fields

Struct fields should be concrete/parametric for type stability and performance.

## 5. Avoid broadcast from within kernels

GPU compilers can fail to infer through broadcast inside kernels. Prefer explicit, inference-friendly kernel code. See [gpu_performance.md](gpu_performance.md) for the canonical definition of "kernel" and "hot path".

## 6. Do not use `Function` as a struct field type

Bad:

```julia
struct A
    f::Function
end
```

Preferred:

```julia
struct A{F <: Function}
    f::F
end
```

## 7. Define `isbits` structs when possible

If a new struct does not contain `ClimaCore` objects, verify a representative instance is `isbits`:

```julia
isbits(A(...))
```

## 8. Do not use `mutable struct`

Prefer immutable structs. Only use mutability when required and explicitly justified.

## 9. Prefer `SVector` or `Tuple` over `Vector` / `Array`

For fixed-size data, use stack-friendly/static representations.

## 10. Reduce allocations

- Avoid explicit allocators like `collect` and `reshape` in performance-sensitive paths.
- Prefer allocation-light transforms (for example, `map`) instead of manual accumulate patterns that create temporary arrays.

## 11. Do not use `@assert` within kernels

Use `error(...)` instead. Do not capture runtime variables in the error message within kernels.

## 12. Do not use `@views`

Follow project conventions that avoid `@views`.

## 13. Do not use `Dict` in kernels

`Dict` is not allowed in CPU/GPU kernels. Replace with custom structs or `NamedTuple`s.

## 14. Duck-type physics functions; avoid explicit `where {FT}` on non-constructors

Prefer `function f(x, y)` over `function f(x::FT, y::FT) where {FT}` in tendency and physics functions. Concrete type annotations on arguments break Dual-number propagation needed for Automatic Differentiation (AD).

Exception: struct constructors that statically allocate `SVector`/`SMatrix` need `::Type{FT}` to determine the element type at compile time.

Bad:

```julia
# Dual numbers don't satisfy ::FT
@inline function compute(x::FT, y::FT) where {FT}
    return x^2 + y
end
```

Preferred:

```julia
# AD-compatible — types inferred from inputs
@inline function compute(x, y)
    return x^2 + y
end
```

## 15. Infer floating-point type from values, not from `where` clauses

Prefer `FT = eltype(params)`, `FT = typeof(x)`, or `FT = eltype(x)` inside function bodies. Avoid repeating `{FT}` in `where` clauses for functions that already receive typed inputs.

Bad:

```julia
function f(x::AbstractArray{FT}) where {FT}
    ε = FT(1e-10)
    # ...
end
```

Preferred:

```julia
function f(x)
    FT = eltype(x)
    ε = FT(1e-10)
    # ...
end
```

## 16. Use `zero(x)` / `one(x)` over `FT(0)` / `FT(1)` for accumulators

Type-agnostic idioms propagate numeric type (including Dual numbers) without an explicit conversion. Use `FT(constant)` only for named constants that must be a specific type.

Bad:

```julia
acc = FT(0)
```

Preferred:

```julia
acc = zero(x)
```

## 17. Replace data-dependent `if/else` with `ifelse` inside GPU kernels

On SIMT architectures, threads in a warp execute in lockstep. A data-dependent `if/else` serializes the two branches across threads (warp divergence). Use `ifelse(cond, a, b)` to compute branchlessly.

**Critical**: both arguments to `ifelse` are always evaluated. Guard mathematically invalid operations (`log`, `sqrt`, division) with `max(x, eps(x))` *before* passing them as arguments — never inside a `begin...end` block inside `ifelse`.

Bad:

```julia
# Thread divergence; log(x) also evaluates when x ≤ 0
result = if x > 0
    log(x) + 1
else
    zero(x)
end
```

Preferred:

```julia
# Branchless; safe_x guards log
safe_x = max(x, eps(x))
log_term = log(safe_x) + one(x)
result = ifelse(x > zero(x), log_term, zero(x))
```

## 18. Prefer functors over closures in broadcast or high-loop contexts

Closures that capture multiple local variables produce heap allocations and may fail to compile on GPU (`InvalidIRError: unsupported dynamic function invocation`). Encapsulate context in a concrete callable struct (functor) instead.

Bad:

```julia
f = (x) -> physics_kernel(params, state, x)
result = integrate(f, data)
```

Preferred:

```julia
struct PhysicsEval{P, S}
    params::P
    state::S
end
(e::PhysicsEval)(x) = physics_kernel(e.params, e.state, x)

result = integrate(PhysicsEval(params, state), data)
```

Validation: `@allocated integrate(PhysicsEval(params, state), data)` should be 0 after a warm-up call.

## 19. Prefer fixed iteration counts in iterative solvers inside GPU kernels

Convergence-based loops (`while err > tol`) cause thread divergence when different threads converge at different rates. Where the physics allows it, prefer a fixed number of iterations so all threads in a warp follow the same execution path.

This is a performance guideline, not a strict rule. Use judgement: fixed iterations are appropriate when a small count (for example, 2–5 Newton steps) is known to be sufficient for the physical accuracy required.

## 20. Extract parameters and non-field scalars before `@.` blocks

Capturing complex parameter structs or thermodynamic parameter containers directly inside a `@.` broadcast expression forces the broadcast engine to determine their broadcast shape at runtime. Extracting them to named local variables before the broadcast (a) makes the broadcast shape unambiguous to the compiler, (b) prevents potential shape-mismatch errors in ClimaCore's field-space broadcast engine, and (c) keeps the broadcast expression readable.

Bad:

```julia
@. result = my_physics(p.params.thermodynamics_params, Y.c.T, Y.c.ρ)
```

Preferred:

```julia
thp = p.params.thermodynamics_params
@. result = my_physics(thp, Y.c.T, Y.c.ρ)
```

Note: `Ref()` is not the standard broadcast-escape pattern in this codebase. Its current use is limited to mutable scalar boxes in callbacks and non-broadcast contexts. Prefer parameter extraction.

## 21. No keyword arguments inside GPU kernels

Keyword arguments introduce a sorter trampoline that can prevent inlining and trigger dynamic dispatch on GPU compilers. Use positional arguments; pass parameter containers instead of individual named constants.

Bad:

```julia
@inline function transform(T, θ; L_v = 2.5e6, c_p = 1004)
    # ...
end
```

Preferred:

```julia
@inline function transform(params, T, θ)
    L_v = get_latent_heat(params)
    c_p = get_cp(params)
    # ...
end
```
