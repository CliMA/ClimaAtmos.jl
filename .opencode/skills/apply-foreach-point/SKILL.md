---
name: apply-foreach-point
description: Instructions on how to collectly apply `foreach_point` construct in ClimaAtmos. Use when asked to apply it bu the user.
---

## Description of the `foreach_point` construct

The primery goal of the `foreach_point` is to fuse multiple kernels into a single kernel and thus decrease memory transfers from the global memory.

In ClimaAtmos each bradcasted expressions (in normal julia sense, often identifiable by `@.` macro) evaluate as a single kernel on a GPU. What `foreach_point` allows us to do, is to combine multiple PUERLY POINTWISE expressions together. Consider e.g.: 

```julia
  @. A = B + C * D
  @. A1 = 3 * A
```

This is two kernels and is undesiderd. With `foreach_point` 
we can combine the into a single kernel:

```julia
foreach_point(A1, A, B, C, D) do A1, A, B, C, D
  @. A = B + C * D
  @. A1 = 3 * A
end
```

## Where `foreach_point` is defined

`foreach_point` is defined in the **`ClimaCore.DataLayouts`** module. Import it
explicitly at the top of the file where you use it:

```julia
import ClimaCore.DataLayouts: foreach_point
# or, if the file already does `import ClimaCore.DataLayouts`:
#   DataLayouts.foreach_point(...)
```

Do not look for it under `ClimaCore.foreach_point` or
`ClimaCore.Fields.foreach_point` — it is *not* re-exported there. When calling
inside a module that already aliases the submodule (e.g.
`import ClimaCore.DataLayouts as DL`), use `DL.foreach_point`.

This construct is suported if values in the argument list are of following type: 
 - ClimaCore.Fields.Field
 - ClimaCore.DataLayouts.DataLayout 

In the argument list we can also put lazy expressions involvinf DataLayouts. Lazy expression can be identified by the use of `lazy` function (e.g. `d = @. lazy(A + B)`).

Note that thelazy expressions involving Fields DO NOT WORK YET. If they are present, you need to switch from using Fields to DataLayouts. You can extract if using the `field_values` funcion e.g.: 

```julia
A_data_layout = ClimaCore.Fields.field_values(A_field)
```
## Passing non DataLayout or Field arguments

Julia sometimes wraps variables captured from the parent scope into a lambda into a `Ref` (this is called boxing). We need to avoid it as it will lead to faliures on a GPU.
For that reason if you need to use some other variable inside the clousure, you need to wrap it into a `let block`. For example: 

```julia
scalar = 3.0
let scalar = scalar
    foreach_point(A1, A, B, C, D) do A1, A, B, C, D
        @. A = B + C * D
        @. A1 = scalar * A
  end
end
```

### Only `isbits` captures are allowed on a GPU

Every value captured into a `foreach_point` closure is eventually passed *into the
GPU kernel*, so it must be a bitstype (`Base.isbits`). The two captures that most
often sneak in as **non-bitstype** are:

  - **A `DataType`, e.g. `FT = eltype(p.params)` then `FT(x)` inside the
    closure.** `DataType` is *not* `isbits`, and CUDA.jl refuses to compile the
    kernel with `KernelError: passing non-bitstype argument … .#FT is of type
    Type{Float32} which is not isbits.` **Fix:** evaluate the type on the host
    *and* store the typed value in a `let` binding, capturing the value rather
    than the type:

    ```julia
    FT = eltype(p.params)
    α_ft = FT(α)            # typed scalar, captured instead of `FT`
    let α_ft = α_ft, ...
        foreach_point(...) do ...
            ... α_ft ...    # never write `FT(...)` inside the closure
        end
    end
    ```

    The same applies to any `Val{...}`, `typeof(...)`, or module/type object
    you might be tempted to interpolate.

  - **A `NamedTuple`.** NamedTuples *are* isbits only when every field is
    isbits; a NamedTuple holding a struct with a `DataType` field (e.g. a
    `CloudFractionFloorParams`-like bundle that accidentally stores a type)
    silently becomes non-isbits. Prefer a `struct` built from plain scalars,
    or unpack the tuple into separate scalar `let` bindings.

If you hit `KernelError: passing non-bitstype argument`, the offending captured
value is named in the `.field is of type T which is not isbits` line of the
error — fix it by precomputing a typed scalar on the host and binding *that*
in the `let` block.

### Every function called from the closure must be `@inline`

Functions called inside a `foreach_point` closure run on the GPU. If a
callee is **not** `@inline`, the GPU compiler cannot specialize it and falls
back to **dynamic function invocation**, which triggers
`InvalidIRError: … unsupported call to an unknown function (call to
gpu_gc_pool_alloc)` or `unsupported dynamic function invocation (call to
foo(...))`.

The fix is to add `@inline` to every helper you call from the closure — and
to *its* callees, transitively. In practice this means the pointwise helpers
(`specific`, `env_value`, `draft_sum`, `ρa⁰`, `draft_area`,
`sgs_weight_function`, …) all need `@inline`.

Equally important: **avoid Base reduction functions** like `mapreduce`,
`reduce`, `first(iter...)` inside the closure — they use dynamic dispatch.
Use the `UnrolledUtilities` family instead (`unrolled_sum`,
`unrolled_mapreduce`, `unrolled_reduce`, …), which are `@inline` and
manually unrolled over tuples, so the GPU compiler can fully specialize them.
`unrolled_mapreduce(f, op, itrs...)` even accepts multiple iterators, so you
can reduce over zipped tuples without `zip`.

## Using Local Cache

The kernels in ClimaAtoms are in general memory bound. Hence we wish to load them with additional computation. To accomplish them we use a technique we will call 'Local Caching' this means instead of using precomputed values, we will recompute them inside the kernel. 

Consider a minimal example:

```julia

function calculate_D(Y, p) 
    f1 = Y.fy1 
    f2 = p.fp1

    return @. lazy(f1 + f2)
end

function calculate_C(Y, p) 
    f1 = Y.fy1 
    f2 = p.fp2

    return @. (f1 * f2)
end


function computation!(Y, p) 
    res = p.result_field

    C = calculate_C(Y, p)
    D = calculate_D(Y, p)

    @. res = C + D
end
```

We should refactor this like the following
```julia
# Pass argument field explicitly and remove the lazy expression.
function calculate_D(f1, f2) 
    return @. f1 + f2
end

function calculate_C(f1, f2) 
    return @. (f1 * f2)
end

function computation!(Y, p) 
    res = p.result_field

    foreach_point(res, Y.fy1, p.fp1, p.fp2) do res, f1, fp1, fp2
        C = calculate_C(f1, fp2)
        D = calculate_D(f1, fp1)

        @. res = C + D
    end
end
```

In short you need to:
 - Pass the fields required to evaluate the expression as function parameters
   explicitly
 - Remove any lazy annotations (it is optional but may be good practice)
 - Call the functions inside the `foreach_point` clousure and pass the required fields as arguments in the argument list.


## General advise

- When implementing the diff do not get concerend with missing imports too much. Rather try to prepere somthing and iterate (by running) and observing the errors until you get it right.
- Do not try to understand the details of each construct (e.g. `unrolled_sum`) just 
  use patterns in the codebase and apply it in a similar way.
