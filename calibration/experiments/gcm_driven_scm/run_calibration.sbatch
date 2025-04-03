#!/bin/bash

#SBATCH --time=10:00:00   # walltime
#SBATCH -J "scm_calibration"   # job name
#SBATCH --ntasks=200  # Number of tasks (~ num ensemble members)
#SBATCH --mem-per-cpu=25G
#SBATCH --cpus-per-task=1  # CPUs per task
#SBATCH --output=scm_calibration.out

module purge
export MODULEPATH="/groups/esm/modules:$MODULEPATH"
module load climacommon/2025_07_30

julia --project run_calibration.jl

echo finished