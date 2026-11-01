# Device-Agnostic and Distributed Code (ClimaComms)

This guide covers patterns for writing code that runs on CPU and GPU, and on single-process and MPI-distributed configurations. All CliMA model packages use `ClimaComms.jl` to abstract over device and parallelism.

> To set up a machine and **run** a model on GPU (install Julia, add `CUDA.jl`, runtime compatibility, `CLIMACOMMS_DEVICE`), see [running_on_gpu.md](../workflow/running_on_gpu.md).

## 1. Acquiring the device and context

Most patterns in this guide need a `device` and a `context` in scope:

```julia
import ClimaComms
ClimaComms.@import_required_backends   # loads the CUDA and MPI extensions if installed; see running_on_gpu.md §4

device  = ClimaComms.device()          # CPU or CUDA, from the CLIMACOMMS_DEVICE environment variable (default CPU)
context = ClimaComms.context(device)   # SingletonCommsContext, or MPICommsContext if an MPI backend is loaded
ClimaComms.init(context)               # required before any collective; no-op for singleton contexts
```

The MPI backend is a package extension loaded by `using MPI`; single-process runs do not need it. Obtain `device` and `context` once at the top of the entry point and pass them down; library functions take them as arguments.

## 2. Device-agnostic code

### Get the array type from the device, not from `Array(...)` or `CUDA.CuArray`

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

`Array(field)` on a GPU field copies it to the host. Call it only in diagnostics or test code where the transfer is acceptable, and say why in a comment. Scalar indexing (`field[i]`) on a GPU array requires `CUDA.@allowscalar` and transfers one element per access; wrap the call so the cost is visible.

### `CUDA.allowscalar(false)` in tests

GPU tests disable scalar indexing so that an accidental `field[i]` errors rather than running ~1000× slower:

```julia
import CUDA
CUDA.allowscalar(false)
```

This is the standard setup in GPU test entry points such as `test/runtests_gpu.jl`. See [testing_and_validation.md](testing_and_validation.md).

## 3. MPI / distributed code

In a single-process (non-MPI) context, `ClimaComms.iamroot(context)` returns `true` and the collective operations (`barrier`, `allreduce`, `bcast`) are no-ops, so the patterns below are safe to write unconditionally.

### Root-only IO

Files, prints, logs, checkpoints, and progress reporting must be guarded with a root-rank check; otherwise every rank writes the same file or line, which garbles output, corrupts files, and makes tests flaky.

```julia
# ❌ Every rank prints
@info "Iteration $i complete"

# ✅ Root only
ClimaComms.iamroot(context) && @info "Iteration $i complete"
```

### Collective operations are collective

Functions like `ClimaComms.barrier`, `ClimaComms.allreduce`, and `ClimaComms.bcast` must be called by **every** rank or the program deadlocks. Keep them out of `iamroot` branches and of any conditional that can differ between ranks.

```julia
# ❌ Deadlock: only root reaches the barrier
if ClimaComms.iamroot(context)
    ClimaComms.barrier(context)
end

# ✅ All ranks call the barrier
ClimaComms.barrier(context)
```

### Determinism across rank counts

A global reduction sums in a different order at each rank count, so its floating-point result changes with the number of ranks. When asserting reproducibility under MPI, prefer integer counts, exact equality on small arrays, or tolerances that exceed `eps(FT) * N` for sums of length `N`.

## 4. Selecting CPU vs GPU at run time

Some test entry points select the array type from `ARGS`:

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

Use this pattern when adding a GPU test entry point to a package that already uses it; otherwise, select the device with `CLIMACOMMS_DEVICE` ([running_on_gpu.md §5](../workflow/running_on_gpu.md)).

## 5. Common pitfalls

- **Implicit host transfer**: `sum(field)` on a GPU field runs as a GPU reduction; `field[1]` is scalar indexing and errors under `CUDA.allowscalar(false)` (§2).
- **Random number streams under MPI**: `rand()` is not synchronized across ranks. For an independent per-rank stream, seed an explicit RNG with a rank-dependent seed (`rng = Random.Xoshiro(base_seed + ClimaComms.mypid(context))`); for cross-rank-synchronized randomness, generate on root and broadcast (`ClimaComms.bcast`).
- **`println` inside kernels**: not GPU-compatible and not MPI-safe. See [gpu_performance.md §7](../performance/gpu_performance.md) for the static-error rule.
- **Saving to a shared filesystem from every rank**: causes file lock contention. Use root-only IO or rank-suffixed filenames.

## Self-correction

If this guide is discovered to be stale or missing a pattern, update it.
