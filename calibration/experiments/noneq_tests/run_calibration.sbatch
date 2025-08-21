#!/bin/bash

#SBATCH --ntasks=10  # number of processor cores (i.e. tasks)
#SBATCH --time=06:00:00   # walltime
#SBATCH --mem-per-cpu=25G   # memory per CPU core
#SBATCH --cpus-per-task=1   # CPUs per task
#SBATCH --mail-user=oalcabes@caltech.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH -J "noneq_calibration"   # job name
#SBATCH --output=noneq_calibration.out

module purge
module load climacommon

julia --project=. run_calibration.jl 2>&1 | tee -a workers.txt