#!/bin/bash
#SBATCH --ntasks=8
#SBATCH --time=6:00:00
#SBATCH --mem-per-cpu=25G
#SBATCH --cpus-per-task=1  # CPUs per task
#SBATCH --output=many_sites.out
#SBATCH --mail-user=oalcabes@caltech.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

#mv julia*.out juliaout/
module purge
module load climacommon

julia --project=. run_many_sites.jl 2>&1 | tee -a workers.txt