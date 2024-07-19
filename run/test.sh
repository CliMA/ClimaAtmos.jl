#!/bin/bash
#SBATCH --ntasks=8
#SBATCH --job-name=test
#SBATCH --time=2:00:00

module purge
module load julia/1.9.4 cuda/12.2 ucx/1.14.1_cuda-12.2 openmpi/4.1.5_cuda-12.2 nsight-systems/2023.3.1

CA_PATH='/home/jschmitt/ClimaAtmos.jl/'
CA_EXAMPLE=$CA_PATH'examples/'
DRIVER=$CA_EXAMPLE'hybrid/driver.jl'
CONFIG_FILE=$CA_PATH'config/central_configs/sphere_baroclinic_wave_rhoe.yml'

export OPENBLAS_NUM_THREADS=1
export JULIA_NVTX_CALLBACKS=gc
export OMPI_MCA_opal_warn_on_missing_libcuda=0
export JULIA_MAX_NUM_PRECOMPILE_FILES=100
export JULIA_LOAD_PATH="${JULIA_LOAD_PATH}:${CA_PATH}.buildkite"
export CLIMACORE_DISTRIBUTED="MPI"
export SLURM_KILL_BAD_EXIT=1
export MPITRAMPOLINE_LIB="/groups/esm/software/MPIwrapper/ompi4.1.5_cuda-12.2/lib64/libmpiwrapper.so"
export MPITRAMPOLINE_MPIEXEC="/groups/esm/software/MPIwrapper/ompi4.1.5_cuda-12.2/bin/mpiwrapperexec"

julia --project=$CA_EXAMPLE -e 'using Pkg; Pkg.instantiate()'
julia --project=$CA_EXAMPLE -e 'using Pkg; Pkg.precompile()'
julia --project=$CA_EXAMPLE -e 'using Pkg; Pkg.status()'

srun julia --project=$CA_EXAMPLE $DRIVER --config_file $CONFIG_FILE