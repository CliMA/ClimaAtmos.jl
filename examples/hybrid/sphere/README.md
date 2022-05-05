# A pipeline to run the sphere simulations on Caltech's central cluster

## Running the simulation

Here is a sbatch script template for setting up simulations on caltech central hpc.
```
#!/bin/bash

#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=$YOUR_JOB_NAME
#SBATCH --time=15:00:00
#SBATCH --output=$YOUR_SIMULATION_LOG_DIR/simulation.log

module purge
module load julia/1.7.2 openmpi/4.0.0

export JULIA_MPI_BINARY=system
export JULIA_CUDA_USE_BINARYBUILDER=false
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:=1}

#export RESTART_FILE=$YOUR_JLD2_RESTART_FILE

CA_EXAMPLE=$HOME'/ClimaAtmos.jl/examples/'
DRIVER=$CA_EXAMPLE'hybrid/driver.jl'

julia --project=$CA_EXAMPLE -e 'using Pkg; Pkg.instantiate()'
julia --project=$CA_EXAMPLE -e 'using Pkg; Pkg.API.precompile()'

julia --project=$CA_EXAMPLE --threads=8 $DRIVER --TEST_NAME held_suarez_rhoe --output_dir=$YOUR_SIMULATION_OUTPUT_DIR

```
In the runscript, one needs to specify the following environmant variable:
* `RESTART_FILE`: if run from a pre-existing jld2 data saved from a previous simulation.

Commonly used command line arguements for experiment setps:
* `--TEST_NAME`: specifies which experiment are to run (e.g., baroclinic_wave_rhoe);
* `--t_end`: specifies the simulation time in seconds;
* `--FLOAT_TYPE`: can be Float32 or Float64; is default to Float32 if not specified;
* `--regression_test`: a boolean var to specify whether the regression test is performed; is default to true if not specified;
* `--enable_threading`: a boolean var to enable multi-threading; defaults to true; Note that Julia must be launched with, for example, `--threads=8`.
* `--job_id`: a uniquely defined id for a job; is default based on the parsed args of the experiment if not specified;
* `--output_dir`: specifies the output directory that saves all the jld2 outputs; is default to `job_id` if not specified.
* `--dt_save_to_disk`: specifies the frequency in seconds to save the data into jld2 files. Defaults to `dt_save_to_disk = 0`, which means no jld2 outputs.
* `--forcing`: specifies the forcing used. Options: [`nothing` (default), `held_suarez`]
* `--microphy`: specifies the microphysics scheme used. Options: [`nothing` (default), `0M`]
* `--turbconv`: specifies the turbulence scheme used. Options: [`nothing` (default), `const_diff`]

To use `held_suarez_rhoe` as an example, one needs to modify [this driver](https://github.com/CliMA/ClimaAtmos.jl/blob/main/examples/hybrid/driver.jl) into the specific setup.


## Remapping the CG nodal outputs in `jld2` onto the regular lat/lon grids and save into `nc` files

`remap_pipeline.jl` remaps CG output onto lat/lon using the `TempestRemapping` subpackage. One needs to specify the following environment variables:
* `JLD2_DIR`: the directory of saved `jld2` files from the simulation;
* `THERMO_VAR`: either `e_tot` or `theta` based on the thermodynamic variable of the simulation;
* `NC_DIR`: the directory where remapped `nc` files will be saved in; if not specified, a subdirectory named `nc` will be created under `JLD2_DIR`;
* `NLAT` and `NLON`: the number of evenly distributed grids in latitudes and longitudes; if not specified, they are default to `90` and `180` respectively.

### Note: A computing node is needed to run the remapping on caltech central hpc. It gives the following warning messages without interrupting the process.
```
/home/****/.julia/artifacts/db8bb055d059e1c04bade7bd86a3010466d5ad4a/bin/ApplyOfflineMap: /central/software/julia/1.7.0/bin/../lib/julia/libcurl.so.4: no version information available (required by /home/jiahe/.julia/artifacts/a990d3d23ca4ca4c1fcd1e42fc198f1272f7c49b/lib/libnetcdf.so.18)
```

