#!/bin/bash

#SBATCH --time=08:00:00   # walltime
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks=1  # number of processor cores (i.e. tasks)
#SBATCH --mem-per-cpu=16G   # memory per CPU core
#SBATCH -J "scm_calibration2"   # job name
#SBATCH --output=scm_calibration2.out

module purge
export MODULEPATH="/groups/esm/modules:$MODULEPATH"
module load climacommon/2024_05_27

julia --project hpc_calibration.jl

julia --project ../post_processing/make_plots.jl 10 adaptive_t

echo finished