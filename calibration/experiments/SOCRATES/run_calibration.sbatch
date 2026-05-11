#!/bin/bash
#SBATCH --job-name=socrates_cal
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=socrates_calibration.%j.out

set -euo pipefail

cd /home/jbenjami/Research_Schneider/CliMA/ClimaAtmos.jl
julia --project=calibration/experiments/gcm_driven_scm calibration/experiments/SOCRATES/run_calibration.jl
