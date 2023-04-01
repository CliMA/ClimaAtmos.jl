# A pipeline to run the sphere simulations on Caltech's central cluster

## Running the simulation

Here is a sbatch script template for setting up simulations using multithreading on caltech central hpc.
```
#!/bin/bash

#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=$YOUR_JOB_NAME
#SBATCH --time=1:00:00
#SBATCH --output=$YOUR_SIMULATION_LOG_DIR/simulation.log

module purge
module load julia/1.8.5

export JULIA_MPI_BINARY=system
export JULIA_CUDA_USE_BINARYBUILDER=false
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:=1}

#export RESTART_FILE=$YOUR_HDF5_RESTART_FILE

CA_EXAMPLE=$HOME'/ClimaAtmos.jl/examples/'
DRIVER=$CA_EXAMPLE'hybrid/driver.jl'

julia --project=$CA_EXAMPLE -e 'using Pkg; Pkg.instantiate()'
julia --project=$CA_EXAMPLE -e 'using Pkg; Pkg.build("HDF5")'
julia --project=$CA_EXAMPLE -e 'using Pkg; Pkg.API.precompile()'

julia --project=$CA_EXAMPLE --threads=8 $DRIVER --forcing held_suarez --output_dir=$YOUR_SIMULATION_OUTPUT_DIR

```

Here is a sbatch script template for setting up simulations using mpi on caltech central hpc.
```
#!/bin/bash

#SBATCH --mem=32G
#SBATCH --ntasks=2
#SBATCH --job-name=$YOUR_JOB_NAME
#SBATCH --time=1:00:00
#SBATCH --output=$YOUR_SIMULATION_LOG_DIR/simulation.log

module purge
module load julia/1.8.5 openmpi/4.1.1 hdf5/1.12.1-ompi411

export JULIA_MPI_BINARY=system
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:=1}
export CLIMACORE_DISTRIBUTED="MPI"
export JULIA_HDF5_PATH=""

#export RESTART_FILE=$YOUR_HDF5_RESTART_FILE

CA_EXAMPLE=$HOME'/ClimaAtmos.jl/examples/'
DRIVER=$CA_EXAMPLE'hybrid/driver.jl'

julia --project=$CA_EXAMPLE -e 'using Pkg; Pkg.instantiate()'
julia --project=$CA_EXAMPLE -e 'using Pkg; Pkg.build("MPI")'
julia --project=$CA_EXAMPLE -e 'using Pkg; Pkg.build("HDF5")'
julia --project=$CA_EXAMPLE -e 'using Pkg; Pkg.API.precompile()'

mpiexec julia --project=$CA_EXAMPLE $DRIVER --forcing held_suarez --output_dir=$YOUR_SIMULATION_OUTPUT_DIR

```

In the runscript, one needs to specify the following environmant variable:
* `RESTART_FILE`: if run from a pre-existing hdf5 data saved from a previous simulation.

Commonly used command line arguments for experiment setups are [here](https://clima.github.io/ClimaAtmos.jl/dev/cl_args/).



## Remapping the CG nodal outputs in `jld2` or `hdf5` onto the regular lat/lon grids and save into `nc` files

`remap_pipeline.jl` remaps CG output onto lat/lon using the `TempestRemapping` subpackage. One needs to specify the following environment variables:
* `JLD2_DIR`: the directory of saved `jld2` files from the simulation;
* `HDF5_DIR`: the directory of saved `hdf5` files from the simulation;
* `NC_DIR`: the directory where remapped `nc` files will be saved in; if not specified, a subdirectory named `nc` will be created under `JLD2_DIR`;
* `NLAT` and `NLON`: the number of evenly distributed grids in latitudes and longitudes; if not specified, they are default to `90` and `180` respectively.

### Note: A computing node is needed to run the remapping on caltech central hpc.
