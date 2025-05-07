#!/bin/bash
#SBATCH --ntasks=100
#SBATCH --time=11:00:00
#SBATCH --mem-per-cpu=25G
#SBATCH --cpus-per-task=1  # CPUs per task
#SBATCH -o workers.txt
#SBATCH -e workers_error.txt
#SBATCH --output=scm_calibration.out

#mv julia*.out juliaout/
module purge
module load climacommon

julia --project=. run_calibration.jl 2>&1 | tee -a workers.txt
