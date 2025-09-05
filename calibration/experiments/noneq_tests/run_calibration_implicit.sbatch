#!/bin/bash

#SBATCH --ntasks=20  # number of processor cores (i.e. tasks)
#SBATCH --time=6:00:00  # walltime
#SBATCH --mem-per-cpu=25G   # memory per CPU core
#SBATCH --cpus-per-task=1   # CPUs per task
#SBATCH --mail-user=oalcabes@caltech.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH -J "noneq_calibration_implicit"   # job name
#SBATCH --output=noneq_calibration_implicit.out

module purge
module load climacommon

julia --project=. run_calibration_implicit.jl 2>&1 | tee -a workers.txt