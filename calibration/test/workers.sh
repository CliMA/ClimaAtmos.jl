#!/bin/bash
#SBATCH --ntasks=12
#SBATCH --time=01:00:00
#SBATCH -o workers.txt

julia --project=calibration/test calibration/test/slurm_workers.jl
