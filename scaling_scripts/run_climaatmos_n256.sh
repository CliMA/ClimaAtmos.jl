#!/bin/bash
#
#SBATCH --nodes=32
#SBATCH --partition=a3mega
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-core=1
#SBATCH --threads-per-core=1
#SBATCH --exclusive
##SBATCH --array=[120,240,480,960]
#SBATCH --array=[120]
#SBATCH --output=slurm_outputs/n256_out-%j_%a.txt
#SBATCH --error=slurm_outputs/n256_err-%j_%a.txt

export JULIA_CUDA_MEMORY_POOL=none
export CLIMACOMMS_DEVICE=CUDA
export CLIMACOMMS_CONTEXT=MPI

export OPAL_PREFIX="/sw/openmpi-5.0.5"
export PATH="/sw/openmpi-5.0.5/bin:$PATH"
export LD_LIBRARY_PATH="/sw/openmpi-5.0.5/lib:$LD_LIBRARY_PATH"

julia -e 'using Pkg; Pkg.add("CUDA"); using CUDA; CUDA.set_runtime_version!(local_toolkit=true)'
julia -e 'using Pkg; Pkg.add("MPI"); using MPI'
julia -e 'using Pkg; Pkg.add("MPIPreferences"); using MPIPreferences; use_system_binary(library_names="/sw/openmpi-5.0.5/lib/libmpi", mpiexec="/sw/openmpi-5.0.5/bin/mpiexec", force=true)'

#julia --project=.buildkite -e 'using Pkg; Pkg.develop(path=".")'
#julia --project=.buildkite -e 'using Pkg; Pkg.instantiate(;verbose=true)'
#julia --project=.buildkite -e 'using Pkg; Pkg.precompile()'
#julia --project=.buildkite -e 'using CUDA; CUDA.precompile_runtime()'
#julia --project=.buildkite -e 'using Pkg; Pkg.status()'

/sw/openmpi-5.0.5/bin/mpiexec -n 256 julia --project=.buildkite .buildkite/ci_driver.jl --config_file config/longrun_configs/longrun_moist_baroclinic_wave_scaling_${SLURM_ARRAY_TASK_ID}.yml
