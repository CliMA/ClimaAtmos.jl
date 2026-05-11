# Device-Agnostic and Distributed Code (ClimaComms)

This guide covers patterns for writing code that runs on CPU and GPU, and on single-process and MPI-distributed configurations. All CliMA model packages use `ClimaComms.jl` to abstract over device and parallelism.

## 1. Device-agnostic code

### Get the array type from the context, not from `Array(...)` or `CUDA.CuArray`

Direct references to a concrete array type tie code to a single backend.

```julia
# ❌ Hard-codes CPU
buffer = Array{FT}(undef, n)

# ❌ Hard-codes CUDA
using CUDA
buffer = CUDA.CuArray{FT}(undef, n)

# ✅ Device-agnostic
ArrayType = ClimaComms.array_type(device)
buffer = ArrayType{FT}(undef, n)
```

### Move data to host only behind a clear guard

Do not silently call `Array(field)` on a GPU field to peek at values; this triggers a host transfer that is invisible to readers and breaks performance. Use it only inside diagnostics or test code where the cost is acceptable, and document why. The same applies to scalar indexing (`field[i]`), which requires `CUDA.@allowscalar` on a GPU array — wrap the call so the cost is visible.

### `CUDA.allowscalar(false)` in tests

GPU tests should disable scalar indexing so accidental `field[i]` accesses surface immediately rather than running 1000× slower:

```julia
import CUDA
CUDA.allowscalar(false)
```

This is the standard pattern in `test/runtests_gpu.jl` across CliMA packages. See [testing_and_validation.md](testing_and_validation.md).

## 2. MPI / distributed code

### Root-only IO

Files, prints, logs, and progress reporting must be guarded with a root-rank check; otherwise every rank writes the same file or line and you get garbled output, file corruption, or test flakiness.

```julia
# ❌ Every rank prints
@info "Iteration $i complete"

# ✅ Root only
ClimaComms.iamroot(context) && @info "Iteration $i complete"
```

The same rule applies to opening output files, writing checkpoints, and updating progress bars.

### Collective operations are collective

Functions like `ClimaComms.barrier`, `ClimaComms.allreduce`, and `ClimaComms.bcast` must be called by **every** rank or the program deadlocks. Do not place them inside an `iamroot` branch, and do not place them inside a conditional that may be true on some ranks and false on others.

```julia
# ❌ Deadlock: only root reaches the barrier
if ClimaComms.iamroot(context)
    ClimaComms.barrier(context)
end

# ✅ All ranks call the barrier
ClimaComms.barrier(context)
```

### Determinism across rank counts

Tests that depend on global reductions can be sensitive to floating-point reordering across rank counts. When asserting reproducibility under MPI, prefer integer counts, exact equality on small arrays, or tolerances that exceed `eps(FT) * N` for sums of length `N`.

## 3. Selecting CPU vs GPU at run time

The standard pattern across CliMA test entry points:

```julia
arg = get(ARGS, 1, "")
if arg == "Array"
    ArrayType = Array
elseif arg == "CuArray"
    import CUDA
    ArrayType = CUDA.CuArray
    CUDA.allowscalar(false)
else
    try
        import CUDA
        ArrayType = CUDA.functional() ? CUDA.CuArray : Array
    catch
        ArrayType = Array
    end
end
```

Use this pattern when adding a new GPU test entry point so all packages stay consistent.

## 4. Common pitfalls

- **Implicit host transfer**: `sum(field)` on a GPU field is fine; `field[1]` is not. The first goes through a GPU reduction; the second triggers `allowscalar` and either errors (recommended) or silently transfers.
- **Random number streams under MPI**: `rand()` is not synchronized across ranks. Seed explicitly per rank, or generate randomness on root and broadcast.
- **`println` inside kernels**: not GPU-compatible and not MPI-safe. See [gpu_performance.md](gpu_performance.md) §7 for the static-error rule.
- **Saving to a shared filesystem from every rank**: causes file lock contention. Use root-only IO or rank-suffixed filenames.

## Self-correction

If this guide is discovered to be stale or missing a pattern, update it.
