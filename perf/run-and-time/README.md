# Compenent profiling

This directory contains the  GPU test and profiling harnesses are run from this directory. Instantiate
the local environment once, then invoke a harness with the model file to run:

```sh
export CLIMACOMMS_DEVICE=CUDA
julia --project -e 'using Pkg; Pkg.instantiate()'
julia --project HARNESS_SCRIPT.jl models/MODEL.jl
```

To perform fast iteration make sure you have `Revise.jl` installed in the base
environment and use `-i` when running the script to enter the REPL after. Then
the same model can be re-run using:
```julia
include("HARNESS_SCRIPT.jl")
```

## Harness scripts

- `timer_harness.jl` uses BenchmarkTools to measure average GPU execution time
  (as well as setup time and warmup/compilation time).

- `quickprof_harness.jl` uses CUDA.jl profiling and prints a summary of the
  measurements for each kernel:

  ```sh
  julia --project quickprof_harness.jl models/hyperdiff_tendency.jl
  ```

- `prof_harness.jl` should be used to run the profiling with Nsight Compute or
  Nsight Systems.

  ```sh
  ncu -o output.ncu-rep --import-source 1 --profile-from-start=off --set=full \
    julia --project prof_harness.jl models/hyperdiff_tendency.jl
  ```

  We also need to make sure that nsys/ncu is useing the same CUDA version as
  Julia. To change the version used by julia run, e.g.:
  ```
  julia --project -e 'using CUDA; CUDA.set_runtime_version!(v"12.2")'
  ```

## Models

The files containing the specific components to be tested are in `models/`:
- `hyperdiff_tendency.jl`
- `implicit_tendency.jl`
- `set_cloud_frac.jl`
- `set_CM_cache.jl`
- `sgs.jl`

These define two functions that are used by the harness scripts:
- `case_setup()` to setup the model state needed for the component.
- `case_run(state)` to execute the component that is to be profiled.
