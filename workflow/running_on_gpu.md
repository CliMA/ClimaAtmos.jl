# Running a CliMA Model on GPU

This guide takes you from a fresh machine to a CliMA model that executes its kernels on an NVIDIA GPU.

This is a *run-the-model* guide. For writing GPU-compatible code (kernels, broadcasts, `isbits`), see [gpu_performance.md](../performance/gpu_performance.md); for the device-agnostic and MPI patterns used inside library code, see [clima_comms.md](../infrastructure/clima_comms.md).

## 1. Install Julia

Install via [`juliaup`](https://julialang.org/downloads/) as described in [onboarding.md §1](onboarding.md). The machine needs a working NVIDIA **driver**; `CUDA.jl` downloads its own CUDA runtime (step 3), so no system CUDA toolkit is needed.

## 2. Add `CUDA.jl`

GPU support is provided through `CUDA.jl`, loaded as a package extension. Add it to the project you run the model from (or to your base environment for interactive use):

```julia
using Pkg
Pkg.add("CUDA")
```

The device is selected by the `CLIMACOMMS_DEVICE` environment variable (step 5); installing `CUDA.jl` alone leaves the model on the CPU.

## 3. CUDA runtime compatibility

The CUDA *runtime* used by `CUDA.jl` must be compatible with the CUDA *driver* installed on the machine. There are three ways to satisfy this:

- **Automatic (default).** On instantiation, `CUDA.jl` downloads a CUDA runtime that is compatible with the local driver. This is the right choice on a personal workstation or a fresh cloud node.
- **Pin a specific runtime version**, e.g. when the auto-selected version misbehaves:

  ```julia
  CUDA.set_runtime_version!(v"11.8")
  ```

- **Use the local system toolkit.** Use this on HPC systems where CUDA is provided through environment modules:

  ```julia
  CUDA.set_runtime_version!(local_toolkit = true)
  ```

`set_runtime_version!` records the choice in `LocalPreferences.toml` and prompts a restart. Verify the resolved runtime, driver, and detected device with `CUDA.versioninfo()`.

## 4. Load the backend with `@import_required_backends`

`ClimaComms` discovers device and MPI backends through package extensions that load only when the underlying package (`CUDA`, `MPI`) is present. The macro loads every backend available in the current environment:

```julia
using ClimaComms
ClimaComms.@import_required_backends   # loads the CUDA (and MPI) extension if installed
```

Place this near the top of your entry point or test file, before calling `ClimaComms.device()`. It is the standard preamble in CliMA GPU test files (e.g. `test/gpu_tests.jl` in CloudMicrophysics, `test/runtests_gpu.jl` in SurfaceFluxes). With `CLIMACOMMS_DEVICE="CUDA"` (step 5) set and `CUDA.jl` not imported, `ClimaComms` errors; the macro loads the backend so that the device resolves to `CUDADevice`.

## 5. Select the device at run time

`ClimaComms.device()` reads the `CLIMACOMMS_DEVICE` environment variable and **defaults to `CPU`**; it does not probe the hardware. To run on the GPU, set it to `CUDA`:

```bash
export CLIMACOMMS_DEVICE="CUDA"   # in the shell, no spaces around =
```

or, equivalently, at the top of your Julia script before any CliMA package loads:

```julia
ENV["CLIMACOMMS_DEVICE"] = "CUDA"
```

Allowed values are `CPU`, `CPUSingleThreaded`, `CPUMultiThreaded`, and `CUDA`. Some test entry points instead select the array type from `ARGS` ([clima_comms.md §4](../infrastructure/clima_comms.md)).

A related variable controls the communication context. On a cluster, `ClimaComms` may try to initialize MPI and crash (e.g. `PMI2_Init failed`) even for a single-process run. Force a singleton context with:

```bash
export CLIMACOMMS_CONTEXT="SINGLETON"
```

## 6. Verify the device

Confirm that the device resolved to the GPU:

```julia
using ClimaComms
ClimaComms.@import_required_backends
device  = ClimaComms.device()          # expect CUDADevice
context = ClimaComms.context(device)
print(summary(context))                # prints the device type, and the GPU UUID on CUDADevice
```

If `device` is a CPU type when you expected CUDA, the `CLIMACOMMS_DEVICE` variable was not set in the process that started Julia.

## Self-correction

If this guide is discovered to be stale or missing a pattern, update it.
