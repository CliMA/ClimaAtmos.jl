#!/bin/bash

#SBATCH --time=30:00:00   # walltime
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks=1  # number of processor cores (i.e. tasks)
#SBATCH --mem=60G   # memory per CPU core
#SBATCH -J "2scm_cal_restart"   # job name
#SBATCH --output=scm_cal_restart.out

module purge
export MODULEPATH="/groups/esm/modules:$MODULEPATH"
module load climacommon/2024_05_27

julia --project restart_calibration.jl "$@"

echo finished
