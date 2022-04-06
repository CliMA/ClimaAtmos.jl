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

export TEST_NAME=sphere/held_suarez_rhoe
export OUTPUT_DIR=$YOUR_SIMULATION_OUTPUT_DIR
#export RESTART_FILE=$YOUR_JLD2_RESTART_FILE

CC_EXAMPLE=$HOME'/ClimaCore.jl/examples/'
TESTCASE=$CC_EXAMPLE'hybrid/driver.jl'

julia --project=$CC_EXAMPLE -e 'using Pkg; Pkg.instantiate()'
julia --project=$CC_EXAMPLE -e 'using Pkg; Pkg.API.precompile()'

julia --project=$CC_EXAMPLE --threads=8 $TESTCASE

```
In the runscript, one needs to specify the following environmant variable:
* `TEST_NAME`: the experiment to run;
* `OUTPUT_DIR`: the directory for jld2 data being saved at;
* `RESTART_FILE`: if run from a pre-existing jld2 data saved from a previous simulation.

Meanwhile, to enable multithreads, one needs to change [here](https://github.com/CliMA/ClimaCore.jl/blob/main/examples/hybrid/driver.jl#L51) in `driver.jl` to be `enable_threading() = true`.

To use `sphere/held_suarez_rhoe` as an example, one needs to modify [these lines](https://github.com/CliMA/ClimaCore.jl/blob/main/examples/hybrid/sphere/held_suarez_rhoe.jl#L6-L16) into the specific setup. In particular, `dt_save_to_disk=FT(0)` means no jld2 outputs. A non-zero value specifies the frequency in seconds to save the data into jld2 files. 


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

