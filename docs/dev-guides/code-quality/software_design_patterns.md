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

## 2. Avoid `using` / `import` between submodules of the same package

Inside `src/`, do not introduce new `using` or `import` statements that pull names from a *sibling* or *parent* submodule of the same package. This rule does not restrict `using`/`import` of external packages; those are normal Julia idioms.

Bad:

```julia
module Foo

baz() = 1

module Bar
  using Foo: baz   # same-package cross-submodule import
  bing() = baz()
end

end
```

Prefer explicit qualification (`Foo.baz()` at the call site) or follow whatever module-wiring convention the package already uses. The goal is to keep include/initialization order auditable and prevent accidental cycles between submodules.

## 3. Do not use `Symbol`s in broadcasted expressions

Avoid symbol-based broadcast patterns. Use concrete, type-stable values/structures instead.

## 4. Do not use abstract types in struct fields

Struct fields should be concrete/parametric for type stability and performance.

## 5. Avoid broadcast from within kernels

GPU compilers can fail to infer through broadcast inside kernels. Prefer explicit, inference-friendly kernel code. See [gpu_performance.md](../performance/gpu_performance.md) for the canonical definition of "kernel" and "hot path".

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

Anything passed into a GPU kernel must be `isbits` *after device adaptation*, not necessarily on the host. ClimaCore objects (e.g. `Field`, `Space`) are deliberately not `isbits` on the host and become `isbits` only after `Adapt.adapt(CUDA.KernelAdaptor(), x)` (i.e. `CUDA.cudaconvert(x)`).

For a new struct that does not wrap device-resident arrays, the host-side check suffices:

```julia
isbits(A(...))
```

For a struct that *does* wrap a `Field`, `CuArray`, or similar, the meaningful check is the post-adapt one (see [gpu_performance.md §8](../performance/gpu_performance.md)):

```julia
isbits(CUDA.cudaconvert(A(...)))
```

In the wrapping case, also define `Adapt.adapt_structure` so the post-adapt object actually becomes `isbits`.

## 8. Prefer immutable structs

Prefer immutable structs for types that are passed into GPU kernels or broadcast expressions. `mutable struct` is acceptable for infrastructure types that are never passed into kernels, for example grid objects (`ClimaCore.Grids`), topologies, and time-stepping integrators (`ClimaTimeSteppers.TimeStepperIntegrator`).

## 9. Prefer `SVector` or `Tuple` over `Vector` / `Array`

For fixed-size data, use stack-friendly/static representations.

## 10. Reduce allocations

- Avoid explicit allocators like `collect` and `reshape` in performance-sensitive paths.
- Prefer allocation-light transforms (for example, `map`) instead of manual accumulate patterns that create temporary arrays.

## 11. Do not use `@assert` within kernels

Use `error("static message")` instead. Do not capture runtime variables in the error string within a kernel: string interpolation allocates and, on GPU, typically fails to compile because the device runtime lacks the full `print_to_string` machinery.

Bad:

```julia
@assert x > 0 "x must be positive, got $x"   # @assert may be removed; interpolation allocates
```

Preferred:

```julia
x > 0 || error("x must be positive")   # static message, no interpolation
```

## 12. Do not use `@views`

Follow project conventions that avoid `@views`.

## 13. Do not use `Dict` in kernels

`Dict` is not allowed in CPU/GPU kernels. Replace with custom structs or `NamedTuple`s.

## 14. Duck-type physics functions; avoid explicit `where {FT}` on non-constructors

This rule is strongest for model-side, tendency, and AD-traversed code: prefer `function f(x, y)` over `function f(x::FT, y::FT) where {FT}`. The `where {FT}` form binds every annotated argument to the *same* concrete element type, which rejects mixed-AD calls (e.g. `f(Dual(1.0), 2.0)`) and rejects `ClimaCore.Field`s whose `eltype`s differ from each other or from a `Float`. Duck typing lets each argument carry its own type and lets AD flow through naturally.

Exceptions:

- **Struct constructors** that statically allocate `SVector`/`SMatrix` need `::Type{FT}` to determine the element type at compile time.
- **Library internals where homogeneous numeric types are intentional** (for example, CloudMicrophysics.jl, Thermodynamics.jl) may use `where {FT}` broadly. Defer to the package's existing style.

Bad:

```julia
# Rejects f(Dual(1.0), 2.0) and any caller with mismatched eltypes
@inline function compute(x::FT, y::FT) where {FT}
    return x^2 + y
end
```

Preferred:

```julia
# AD-compatible: types inferred from inputs
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

A data-dependent `if/else` in a GPU kernel causes warp divergence; `ifelse(cond, a, b)` computes branchlessly. Both arguments are always evaluated, so guard mathematically invalid operations (`log`, `sqrt`, division) *before* the `ifelse`, not inside a `begin...end` block inside it.

For the full explanation (SIMT semantics, why `ifelse` does not skip work, and the worked `log(x)` example), see [GPU Performance Guide §1](../performance/gpu_performance.md). For the broader branch-avoidance discipline (including evaluating both arms of a physical case split and combining pointwise conditions with `&`/`|` rather than `&&`/`||`), see the [Branchless Code Guide](../performance/branchless_code.md). For choosing the right floor in the pre-guard, see [Numerical Robustness §1–2](../performance/numerical_robustness.md).

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

Convergence-based loops (`while err > tol`, or `for ...; converged && break; end`) cause thread divergence when different threads converge at different rates: the warp runs until the slowest point finishes, and the early-exit `break` is itself a data-dependent branch. Where the physics allows it, prefer a fixed number of iterations so all threads in a warp follow the same execution path.

Fix the count by *physical adequacy* (e.g. temperature to ~0.1 K, not to `eps(FT)`), and determine it with an offline test that sweeps the full range of conditions a climate run can produce. The canonical example is `Thermodynamics.saturation_adjustment` (a fixed `maxiter = 2` Newton solve, no convergence flag). For the methodology, the offline-test checklist, and the worked example, see the [Branchless Code Guide §4–5](../performance/branchless_code.md).

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

## 22. Use SafeTestsets.jl to avoid leaky unit tests

Prefer `@safetestset` over `@testset` + nested `include` so variables and imports do not leak between test files. See [testing_and_validation.md](../infrastructure/testing_and_validation.md) for the pattern and per-repo conventions.

## 23. Do not use list comprehensions

Avoid list comprehensions like `[getproperty(dist, p) for p in params]` in hot paths or GPU code, as they explicitly allocate `Array`s on the heap.

Use `map` over a `Tuple` or `SVector`, which returns a `Tuple` or `SVector` respectively without heap allocation. The rule is about the *input type*: `map` over a `Vector` / `Array` still allocates a new `Array`.

```julia
# Bad: allocates a Vector
x = [f(p) for p in params]   # params is a Vector

# Preferred: input is a Tuple → map returns a Tuple, no allocation
x = map(f, (a, b, c))

# Preferred: input is an SVector → map returns an SVector
x = map(f, SVector(a, b, c))
```

## 24. Limit use of @generated functions

Generated functions are versatile and helpful for debugging, but they have significantly higher compilation latencies than non-generated functions.

To minimize compilation time, only use generated functions when absolutely necessary. This includes the following situations:

- Guaranteeing inlining or unrolling of code, to avoid compilation errors when the compiler fails to perform these optimizations.
- Dynamically constructing a `String` inside a GPU kernel, which requires runtime allocations.
- Dynamically constructing a `Symbol` inside a GPU kernel, since `Symbol`s are implemented as interned strings.

## Self-correction

If this guide is discovered to be stale or missing a pattern, update it.
