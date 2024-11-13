#!/bin/bash
#SBATCH --time=01:35:00
#SBATCH --ntasks=52
#SBATCH --cpus-per-task=3
#SBATCH --output sb_worker_calibration.txt

julia --project=calibration/experiments/gcm_driven_scm calibration/experiments/gcm_driven_scm/run_worker_calibration.jl
