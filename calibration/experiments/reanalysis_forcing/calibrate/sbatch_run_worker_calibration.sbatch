#!/bin/bash
#SBATCH --ntasks=101
#SBATCH --time=10:10:00
#SBATCH --mem-per-cpu=8000
#SBATCH -o workers.txt
#SBATCH -e workers_error.txt

mv julia*.out juliaout/

module load climacommon/2024_10_08

julia --project=. run_worker_calibration.jl 2>&1 | tee -a workers.txt
