# Software Design Patterns (SDPs)

This document outlines robust and maintainable software patterns to follow. Using robust and maintainable software patterns is very important because often times patterns are spread, and while deviating from maintainable patterns may be fine occasionally, it cannot be the norm.

### Avoid Abstract types in structs

### Use structs over strings in tendency functions

Avoid using

```julia
struct Foo{T}
    model::T
end
function baz(f)
    if f.model == "ModelA"
        # do something ModelA-specific
    elseif f.model == "ModelB"
        # do something ModelA-specific
    end
end
f = Foo("ModelA")
baz(f)
```

Instead use

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
        # do something ModelA-specific
    end
end
f = Foo(ModelA())
baz(f)
```

There are exceptions, however. Strings used during initialization may be the only option, for example, when parsing string inputs into data structures.

### Avoid `using SomePkg / import SomePkg` in `src/`

Avoid using:
```julia
module Foo

baz() = 1

module Bar
  using Foo: baz
  bing() = baz()
end

end
```

### Do *not* use `Symbol`s in broadcasted expressions

### Do *not* use abstract types in structs

### Avoid broadcast from within kernels

Occasionally, the gpu compiler is not able to infer through broadcast expressions from _within_ a gpu kernel.

### Do *not* use `Function` in struct definition
Use `F <: Function` instead.

### Define `isbits` structs
When you define a new struct that doesn't contain `ClimaCore` objects, check that `isbits(A)` is `true`, where `A` is a typical instance of your struct.

### Do *not* use `mutable struct`s

### Use `SVector`s or `Tuple`s instead of `Vector`s or `Array`s

### Reduce allocations
- `collect` and `reshape` both explicitly allocate - avoid these
- Use `map` over initializing/accumulating loops when applicable, as in this case:


### Do *not* use `@assert`
Use `error` instead, and don't capture variables in the error message

### Do *not* use `@views`

### Do *not* use `Dict`s in kernels
`Dict`s are not allowed in CPU or GPU kernels. They can be replaced by custom structs or by `NamedTuple`s.
