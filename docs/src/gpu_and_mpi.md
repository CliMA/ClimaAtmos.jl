# Running on GPUs and MPI

The same model code runs on a single CPU, on many CPU cores or nodes through
MPI, and on GPUs. [ClimaComms.jl](https://clima.github.io/ClimaComms.jl/stable/)
selects the compute device and the communication context from environment
variables, so switching hardware requires no change to Julia scripts or YAML
configuration files: set the environment variables and launch the process
accordingly. (In a YAML configuration, the device can also be set with the
`device` key, e.g. `device: CUDADevice`.)

The examples below launch a run script from the repository root. Any script
that builds and solves a simulation works, provided it loads the backend that
the environment variables request, through
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

## Running on a GPU

Launch Julia on a machine with a compatible NVIDIA GPU and set
`CLIMACOMMS_DEVICE` to `"CUDA"`:

```bash
CLIMACOMMS_DEVICE="CUDA" julia --project run_aquaplanet.jl
```

GPU support is loaded through
[CUDA.jl](https://cuda.julialang.org/stable/), which must be available in the
load path. The package's own `Project.toml` does not include it; install it
once into your default environment from a plain `julia` session
(`import Pkg; Pkg.add("CUDA")`), where the load path picks it up, as described
in [Installation](installation.md). For first-time
machine setup, from a fresh node to a working GPU run, including CUDA runtime
and driver compatibility, see the shared guide
[running\_on\_gpu.md](https://github.com/CliMA/ClimaAtmos.jl/blob/main/docs/dev-guides/workflow/running_on_gpu.md).

!!! note

    GPU memory is often the limiting factor. If a simulation runs out of
    memory, reduce the number of horizontal elements or vertical levels, or
    distribute the run over more GPUs with MPI (below).

## Running with MPI

To distribute a simulation across CPU cores or compute nodes:

 1. Add [MPI.jl](https://juliaparallel.org/MPI.jl/stable/) to your default
    environment (`import Pkg; Pkg.add("MPI")`), as with CUDA.jl above.
 2. Make an MPI implementation available. On clusters, this is usually a
    system module; see the
    [MPI.jl configuration documentation](https://juliaparallel.org/MPI.jl/stable/configuration/)
    for pointing Julia at a system MPI.
 3. Set `CLIMACOMMS_CONTEXT` to `"MPI"`.
 4. Launch through the MPI launcher (`mpiexec`, `mpirun`, or `srun` under
    Slurm).

```bash
CLIMACOMMS_CONTEXT="MPI" srun --ntasks=4 julia --project run_aquaplanet.jl
```

Two behaviors specific to distributed runs:

  - The root process (rank 0) writes the diagnostic NetCDF files; the HDF5
    checkpoints are written collectively by all ranks through parallel HDF5.
  - ClimaAtmos triggers garbage collection on all processes together, every
    1000 steps by default, to keep collections from running at different
    times on different ranks. The `CLIMAATMOS_GC_NSTEPS` environment variable
    sets the interval.

## Combining MPI and GPUs

The two settings compose: each MPI rank drives one GPU.

```bash
CLIMACOMMS_CONTEXT="MPI" CLIMACOMMS_DEVICE="CUDA" srun --ntasks=4 julia \
    --project run_aquaplanet.jl
```

This is the configuration used for high-resolution global simulations.

For writing code that runs on GPUs (kernel compatibility, broadcasting,
allocation constraints), see the shared developer guide
[gpu\_performance.md](https://github.com/CliMA/ClimaAtmos.jl/blob/main/docs/dev-guides/performance/gpu_performance.md);
for the device-agnostic patterns used inside library code, see
[clima\_comms.md](https://github.com/CliMA/ClimaAtmos.jl/blob/main/docs/dev-guides/infrastructure/clima_comms.md).
