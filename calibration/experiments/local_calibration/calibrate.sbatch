#!/bin/bash

#SBATCH --time=06:00:00   # walltime
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks=1  # number of processor cores (i.e. tasks)
#SBATCH --mem-per-cpu=3G   # memory per CPU core
#SBATCH -J "scm_calibration"   # job name
#SBATCH --output=scm_calibration.out

module purge
export MODULEPATH="/groups/esm/modules:$MODULEPATH"
module load climacommon/2024_05_27

julia --project hpc_calibration.jl

echo finished