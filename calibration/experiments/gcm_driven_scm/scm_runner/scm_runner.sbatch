#!/bin/bash

#SBATCH --time=10:00:00   # walltime
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks=8  # number of processor cores (i.e. tasks)
#SBATCH --mem=75G   # memory
#SBATCH -J "scm_runner"   # job name
#SBATCH --output=scm_runner.out

module purge
export MODULEPATH="/groups/esm/modules:$MODULEPATH"
module load climacommon/2024_05_27

julia --project scm_runner.jl

echo finished