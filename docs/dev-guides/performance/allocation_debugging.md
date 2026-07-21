# Allocation Debugging Workflow

Practical recipes for finding and fixing heap allocations in CliMA Julia code. The companion rules (what counts as a hot path and which patterns to use) live in [gpu_performance.md](gpu_performance.md). This guide is the *debug-loop* counterpart: what to do when `@allocated` is not zero.

## 1. The standard regression test

Every CliMA repo with a numerical hot path pins zero allocations with a unit test. The canonical pattern (warm up, then assert) appears in `Thermodynamics.test/type_stability.jl`, `ClimaTimeSteppers.test/...`, and the precomputed-quantity setters of every model repo:

```julia
# Warm up: forces compilation, fills in any first-call caches
my_hot_function(args...)

# Assert
@test (@allocated my_hot_function(args...)) == 0
```

Two things to know:

- `@allocated` returns *bytes* allocated since the last call. The first invocation is dominated by compilation and is never zero, so always warm up first.
- `@allocated` ignores *stack* allocations and small inlined boxes; it counts genuine heap allocations. So `0` from `@allocated` is a strong signal.

When the test fails, the next sections tell you how to localize the cause.

## 2. Localizing an allocation: `Profile.Allocs`

`@allocated` tells you *that* there is an allocation but not *where*. `Profile.Allocs` (a stdlib module on currently supported Julia versions) does. The pattern used in `CloudMicrophysics.test/performance_tests.jl`:

```julia
import Profile

Profile.clear()
Profile.Allocs.@profile sample_rate = 1 my_hot_function(args...)
results = Profile.Allocs.fetch()
sorted = sort(results.allocs, by = x -> x.size)
for a in sorted
    println(a)            # type and size
    println(a.stacktrace) # where it came from
end
```

Tips:

- `sample_rate = 1` records *every* allocation; lower rates (e.g. `0.01`) are appropriate for longer runs.
- The largest allocation is usually the most informative: sort by `.size` and start from the bottom.
- The stack trace points at the *allocating line*, not the *root cause*. A boxed closure variable will show as an allocation in `Core.Box` deep inside the kernel; the fix is upstream, where the closure was constructed.

## 3. Localizing a *dispatch* allocation: `JET.@report_opt`

A common source of allocations is a single dynamic dispatch buried in an otherwise typed function. `Profile.Allocs` will show *that* `jl_apply_generic` allocated; `JET.@report_opt` will show *which call site* the compiler failed to resolve. Used in `CloudMicrophysics.test/performance_tests.jl` (`JET.@test_opt`) and `ClimaTimeSteppers.perf/jet.jl`:

```julia
using JET
JET.@report_opt my_hot_function(args...)
```

JET prints a list of `runtime dispatch detected` entries with the unresolved call and the failing argument type. Reading order:

1. Find the deepest call that is yours (not a stdlib or third-party method).
2. Look at the *argument types*: a `Union{...}`, `Any`, or `Function` is the giveaway. Track that argument back to where it lost type information (a struct field typed `::Function` instead of `::F`, a `Vector{Any}`, an `Union{Nothing, T}` field accessed without dispatch; see [SDP 6](../code-quality/software_design_patterns.md) and [type_stability.md §3](type_stability.md)).
3. Fix the upstream type instability; the JET entry disappears and the allocation goes with it.

Use `JET.@test_opt my_hot_function(args...)` in a `@testset` for a CI-style regression gate.

## 4. Localizing an inference allocation: `@code_warntype`

When the suspect is a single function and you want to see what the compiler inferred, `@code_warntype` highlights any non-concrete inferred types in red. Useful for the smallest reproducer; less useful for deep call graphs (use JET there).

```julia
@code_warntype my_hot_function(args...)
```

Read the `Body::T` line first: if `T` is `Any` or a `Union`, the return type is unstable. Then scan for red `Union{...}` annotations in the body.

## 5. Benchmarking: `BenchmarkTools.@benchmark`

`@allocated` is binary (zero or not zero). For *characterizing* allocations and their cost over many calls, use BenchmarkTools, as in `CloudMicrophysics.test/performance_tests.jl` and `ClimaTimeSteppers.perf/jet.jl`:

```julia
using BenchmarkTools
trial = @benchmark $(splat(my_hot_function))($args) samples = 100 evals = 100
# trial.memory:   total bytes allocated per call
# trial.allocs:   number of distinct allocations per call
# minimum(trial): most useful time estimate (resilient to GC noise)
```

Two pitfalls:

- Always interpolate (`$x`) into the macro so the benchmarked function sees the value, not a global ref.
- `$foo($args...)` can itself allocate from splatting; prefer `$(splat(foo))($args)` or pass arguments individually.

## 6. Flame graphs: `Profile` + `ProfileCanvas` / `PProf`

For longer hot paths (e.g. one full RK step), a flame graph shows where time *and* allocations cluster. The pattern in `ClimaTimeSteppers.perf/flame.jl`:

```julia
using Profile, ProfileCanvas
Profile.clear()
@profile for _ in 1:100_000; my_step!(args...); end
ProfileCanvas.html_file("flame.html")  # save artifact; or .view() locally
```

A model-repo equivalent (e.g. `perf/flame.jl` in ClimaAtmos) profiles a single timestep; the same idea applies to library code.

## 7. Common allocation sources and their fixes

| Symptom (in `Profile.Allocs` stacktrace)         | Likely cause                                          | Fix |
|:-------------------------------------------------|:------------------------------------------------------|:----|
| `Core.Box` deep inside a kernel                  | Captured local variable in a closure                  | Convert closure to a functor ([SDP 18](../code-quality/software_design_patterns.md)) |
| `jl_apply_generic` with no obvious source        | Dynamic dispatch on a `Function`/`Any` field           | Parameterize the field ([SDP 6](../code-quality/software_design_patterns.md)) |
| Allocation inside a `@.` broadcast               | Non-`isbits` argument captured in the broadcast        | Extract parameters to locals ([SDP 20](../code-quality/software_design_patterns.md)); or check `isbits(cudaconvert(...))` ([gpu_performance.md §8](gpu_performance.md)) |
| `Array{Float64}` allocation in a Float32 path    | A `Float64` literal promoting the result                | Track down the literal ([type_stability.md §1](type_stability.md)) |
| `Union{T, Nothing}` access in a hot path         | Optional field forces a runtime check                  | Split the method by the field's type parameter ([type_stability.md §3](type_stability.md)) |
| Allocation inside `ifelse`                       | A `begin...end` block or non-`isbits` branch result    | Pre-compute both branches outside `ifelse` ([SDP 17](../code-quality/software_design_patterns.md)) |
| Allocation only on GPU, not CPU                  | A kernel-incompatible operation (string interp, Dict)  | See [gpu_performance.md §7](gpu_performance.md) and [SDP 13](../code-quality/software_design_patterns.md) |

## 8. When `@allocated` says zero but you still suspect a leak

Three failure modes where `@allocated == 0` is misleading:

1. **The allocation happens during compilation, not execution.** First-call allocations are real cost, just amortized. If startup time matters (e.g. precompile-sensitive jobs), profile with `--trace-compile=stderr`.
2. **The allocation happens on the GPU side.** `@allocated` only measures host allocations. Use `CUDA.@time` or `CUDA.memory_status()` to see device allocations.
3. **The allocation happens elsewhere in the call graph.** A function can be allocation-free locally while its caller (a wrapping `@.` broadcast, or a `bycolumn` iterator) allocates. Pin the test at the outermost layer that runs per timestep, not at a leaf function.

## Self-correction

If this guide is discovered to be stale or missing a pattern, update it.
