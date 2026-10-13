# Running on GPUs and MPI

The same model code runs on a single CPU, on many CPU cores or nodes through
MPI, and on GPUs. Device and process selection belong to
[ClimaComms.jl](https://clima.github.io/ClimaComms.jl/stable/) and to the
spaces ClimaCore builds on it; ClimaCore's
[Run on a GPU](@extref ClimaCore Run-on-a-GPU) and
[Run distributed with MPI](@extref ClimaCore Run-distributed-with-MPI) cover
the backends, the environment variables, and what makes code GPU-compatible.
This page covers what ClimaAtmos adds.

Any script that builds and solves a simulation works on any backend, provided
it loads the one the environment requests through
[`ClimaComms.@import_required_backends`](@extref). This one runs the aquaplanet
configuration from [Global Simulations](global_simulations.md):

```julia
# run_aquaplanet.jl
import ClimaComms
ClimaComms.@import_required_backends   # loads CUDA.jl and/or MPI.jl on demand
import ClimaAtmos as CA

config = CA.AtmosConfig(
    "config/model_configs/prognostic_edmfx_aquaplanet.yml";
    job_id = "my_aquaplanet",
)
CA.solve_atmos!(CA.AtmosSimulation(config))
```

Launch it with the ClimaComms variables set, and the MPI launcher when
distributing:

```bash
CLIMACOMMS_DEVICE="CUDA" julia --project run_aquaplanet.jl
```

```bash
CLIMACOMMS_CONTEXT="MPI" CLIMACOMMS_DEVICE="CUDA" srun --ntasks=4 julia \
    --project run_aquaplanet.jl
```

The second form, one MPI rank per GPU, is the configuration used for
high-resolution global simulations. CUDA.jl and MPI.jl are not dependencies of
ClimaAtmos; install them once into your default environment, as described in
[Installation](installation.md).

Three behaviors are specific to ClimaAtmos:

  - In a YAML configuration, the `device` key selects the device: `auto`, the
    default, defers to the environment, and an explicit value
    (`CPUSingleThreaded`, `CPUMultiThreaded`, `CUDADevice`) overrides it.
  - In a distributed run, the root process (rank 0) writes the diagnostic
    NetCDF files; the HDF5 checkpoints are written collectively by all ranks
    through parallel HDF5.
  - ClimaAtmos triggers garbage collection on all processes together, every
    1000 steps by default, so that collections do not run at different times
    on different ranks. The `CLIMAATMOS_GC_NSTEPS` environment variable sets
    the interval.

GPU memory is often the limiting factor. If a simulation runs out of memory,
reduce the number of horizontal elements or vertical levels, or distribute the
run over more GPUs with MPI.

For first-time machine setup, from a fresh node to a working GPU run, see the
shared guide
[running\_on\_gpu.md](https://github.com/CliMA/ClimaAtmos.jl/blob/main/docs/dev-guides/workflow/running_on_gpu.md);
for writing kernel-compatible code, [gpu\_performance.md](https://github.com/CliMA/ClimaAtmos.jl/blob/main/docs/dev-guides/performance/gpu_performance.md).
