#!/bin/bash

#SBATCH --time=15:00:00   # walltime
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks=1  # number of processor cores (i.e. tasks)
#SBATCH --mem=5G   # memory per CPU core
#SBATCH -J "restart_scm_calibration"   # job name
#SBATCH --output=restart_scm_calibration.out

module purge
export MODULEPATH="/groups/esm/modules:$MODULEPATH"
module load climacommon/2024_05_27

julia --project restart_calibration.jl "$@"

echo finished
