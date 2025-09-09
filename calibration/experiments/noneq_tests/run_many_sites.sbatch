#!/bin/bash
#SBATCH --ntasks=800
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=25G
#SBATCH --cpus-per-task=1  # CPUs per task
#SBATCH -o workers.txt
#SBATCH -e workers_error.txt
#SBATCH --output=scm_ensemble1.out
#SBATCH --mail-user=jschmitt@caltech.edu
#SBATCH --mail-type=ALL

#mv julia*.out juliaout/
module purge
module load climacommon

julia --project=. run_ensemble.jl 2>&1 | tee -a workers.txt