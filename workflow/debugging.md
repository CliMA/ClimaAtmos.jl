# Interactive Debugging Recipes

Practical recipes for chasing wrong values, NaNs, and dispatch surprises in CliMA Julia code. For *allocation* debugging see [allocation_debugging.md](../performance/allocation_debugging.md); for type-inference issues see [type_stability.md](../performance/type_stability.md). This guide covers everything else.

## 1. Numerical instability: NaNs, blowups, and "it crashed at step 437"

A simulation that starts cleanly and crashes a few hundred steps in is usually a numerical-stability issue, not a logic bug. The standard workflow:

1. **Find the failing step.** Run the simulation once with the failing config and note the step number where the NaN or domain error appears.
2. **Reproduce up to just before the crash.** Start an interactive simulation and advance it one step at a time until one step before the failure. CliMA model repos drive [`ClimaTimeSteppers`](https://github.com/CliMA/ClimaTimeSteppers.jl), so the call is `CTS.step!(integrator)` (with `import ClimaTimeSteppers as CTS`).
3. **Inspect the state.** Look at `Y`, `Yₜ`, and `p.precomputed` fields for: NaNs, negative values where positivity is required (e.g. specific humidity, density, water content), zeros where a quantity must be nonzero, and extreme magnitudes that suggest a runaway feedback.

```julia
import ClimaCore as CC
# Quick check across a field
extrema(parent(Y.c.ρ))             # finite range?
any(isnan, parent(Y.c.ρ))          # any NaN?
```

For atmosphere-specific stability heuristics (CFL, hyperviscosity, sponge layers), see the [ClimaAtmos stability wiki](https://github.com/CliMA/ClimaAtmos.jl/wiki/Stability-of-simulations).

4. **Enable `DebugOnly.call_post_op_callback`** if the source of the instability isn't obvious. When enabled, this callback runs after every `ClimaCore` operation, allowing you to determine the exact operation that produces the first NaN. The callback is expensive, so only enable it for debugging. For setup, see the [ClimaCore debugging guide](https://clima.github.io/ClimaCore.jl/dev/debugging/#DebugOnly.call_post_op_callback).

## 2. Inspecting state with `@show`, `@info`, and `Infiltrator`

In order of intrusiveness:

```julia
@show x                  # one-line printout: "x = 3.14"
@info "before update" x y # structured log line, easier to grep
```

`@show` and `@info` are fine for narrowing down where a value goes wrong. When you want to *stop* and explore the surrounding scope, use [Infiltrator.jl](https://github.com/JuliaDebug/Infiltrator.jl):

```julia
using Infiltrator
function suspicious_fn(a, b)
    c = a + b
    Main.@infiltrate     # REPL opens here with a, b, c visible
    return c / a
end
```

At the `infil>` prompt you can read locals, call functions, and continue with `@continue`. Infiltrator must live in your base environment (`~/.julia/config/startup.jl` is the usual place) so Julia finds it regardless of the active project.

For step-through execution, [Debugger.jl](https://github.com/JuliaDebug/Debugger.jl)'s `@enter f(args...)` works for small functions but is slow on the interpreter; Infiltrator is faster for most CliMA hot-path debugging.

## 3. Dispatch debugging: why is *that* method being called?

Subtle bugs often come from a function being called with a slightly wrong argument type, dispatching to a more generic method that returns the wrong thing. Three quick introspection moves:

```julia
@which my_func(arg1, arg2)         # the exact method that will run

methods(my_func)                   # every method of my_func

import InteractiveUtils
InteractiveUtils.methodswith(typeof(arg1), my_func)  # all methods of my_func taking arg1's type
```

If `@which` points at a more abstract method than you expected, the argument's static type is the culprit; track back to where it lost its concrete type. For type-instability tooling (`@code_warntype`, `JET.@report_opt`) see [allocation_debugging.md §§3–4](../performance/allocation_debugging.md).

## 4. Plotting `ClimaCore.Field`s

When numerical inspection isn't enough, a heatmap usually reveals the structure (a single column? a hemisphere? the poles?) of a bug. `ClimaCoreMakie` plots `Field`s directly:

```julia
using ClimaCoreMakie, CairoMakie, Makie

field = ...                                  # a 2D Field
fig = Figure()
ax = Axis(fig[1, 1], title = "T_sfc $(extrema(parent(field)))")
hm = fieldheatmap!(ax, field)
Colorbar(fig[:, end+1], hm)
Makie.save("T_sfc.png", fig)
```

For a 3D `Field`, extract a level first:

```julia
import ClimaCore as CC
field_sfc = CC.Fields.level(field_3d, 1)     # bottom level
fieldheatmap!(ax, field_sfc)
```

For Oceananigans state inspection, the [ClimaCoupler debugging guide](https://clima.github.io/ClimaCoupler.jl/dev/debugging/) has a "Plotting Oceananigans fields" section that covers 2D fields (`OC.interior()`), 3D fields (surface-level indexing), and `AbstractOperation` fields.

## 5. Common patterns that produce silently-wrong values

| Symptom                                          | Likely cause                                                         |
|:-------------------------------------------------|:---------------------------------------------------------------------|
| A field is zero where it should be updated      | The writer was never wired into the integrator (the tendency function exists but no caller assigns into `Yₜ.<field>`) |
| A field carries stale values across stages       | A tendency function reads from `Yₜ` instead of writing to it; `Yₜ` must be write-only per [ecosystem_conventions.md §2](../architecture/ecosystem_conventions.md) |
| Result differs by `~eps` from a reference        | Reordered floating-point arithmetic from a refactor, usually harmless, but flag with `🤖precisionΔ` ([changelogs_and_versions.md §1.4](../code-quality/changelogs_and_versions.md)) |
| Float32 simulation diverges where Float64 is fine | A `1.0`/`Inf`/`6^x` literal promoted to Float64, see [type_stability.md §1](../performance/type_stability.md) |
| NaN appears only on GPU                          | A scalar-indexing fallback that returns garbage (see [clima_comms.md §5](../infrastructure/clima_comms.md)), or a non-`isbits` arg in a kernel (see [gpu_performance.md §8](../performance/gpu_performance.md)) |
| Result depends on MPI rank count                 | A non-associative reduction or per-rank random state: see [clima_comms.md §2](../infrastructure/clima_comms.md) |

## 6. Other common pitfalls

- If an operation on ClimaCore `Field`s shows unexpected values in the REPL, check whether the `Field` has a mask. Masked values can appear as NaN or garbage when printed, even though the non-masked data is correct.

## Self-correction

If this guide is discovered to be stale or missing a pattern, update it.
