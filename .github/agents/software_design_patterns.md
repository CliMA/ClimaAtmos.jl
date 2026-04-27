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

GPU compilers can fail to infer through broadcast inside kernels. Prefer explicit, inference-friendly kernel code. (A kernel is the RHS of any @. / Operators.column_integral_* / Fields.bycolumn expression.)

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
