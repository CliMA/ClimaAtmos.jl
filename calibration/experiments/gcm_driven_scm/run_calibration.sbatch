#!/bin/bash

#SBATCH --time=7:00:00   # walltime
#SBATCH -J "scm_calibration"   # job name
#SBATCH --ntasks=101   # Number of tasks (~ num ensemble members + 1)
#SBATCH --mem-per-cpu=18G
#SBATCH --cpus-per-task=1  # CPUs per task
#SBATCH --output=scm_calibration.out

module purge
export MODULEPATH="/groups/esm/modules:$MODULEPATH"
module load climacommon/2024_12_16

julia --project run_calibration.jl

echo finished