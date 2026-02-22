#!/bin/bash

#SBATCH --time=6:00:00   # walltime
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks=11  # number of processor cores (i.e. tasks)
#SBATCH --mem=100G   # memory
#SBATCH --cpus-per-task=1
#SBATCH -J "scm_runner"   # job name
#SBATCH --output=scm_runner_%j.out

module purge
export MODULEPATH="/resnick/groups/esm/modules:$MODULEPATH"
module load climacommon/2025_07_30

julia --project scm_runner.jl "$@"

echo finished