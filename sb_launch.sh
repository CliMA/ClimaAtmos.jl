#!/bin/bash
#
#SBATCH --nodes=96
#SBATCH --partition=a3
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8 # was 1
#SBATCH --ntasks-per-core=1
#SBATCH --threads-per-core=1
#SBATCH --exclusive
#SBATCH --output=slurm_outputs/job-%j-h-1024-nr-768.out
#SBATCH --error=slurm_outputs/job-%j-h-1024-nr-768.err


export OPAL_PREFIX="/sw/openmpi-5.0.5"
export PATH="/sw/openmpi-5.0.5/bin:$PATH"
export LD_LIBRARY_PATH="/sw/openmpi-5.0.5/lib:$LD_LIBRARY_PATH"

export UCX_WARN_UNUSED_ENV_VARS=n # suppress harmless ucx warnings
export JULIA_CUDA_MEMORY_POOL=none
export CLIMACOMMS_DEVICE=CUDA
export CLIMACOMMS_CONTEXT=MPI
	
/sw/openmpi-5.0.5/bin/mpiexec -n 768 ~/.julia/juliaup/julia-1.10.5+0.x64.linux.gnu/bin/julia --project=examples examples/hybrid/driver.jl --config_file config/gpu_configs/gpu_aquaplanet_dyamond_helem_1024_0M.yml --job_id gpu_aquaplanet_dyamond_helem_1024_nr_768_v6_yfc

