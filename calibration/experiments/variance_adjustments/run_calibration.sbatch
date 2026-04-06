#!/bin/bash
#SBATCH --job-name=variance_calib
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=variance_calibration_%j.out

# GPU: add --gres=gpu:1 and export CLIMACOMMS_DEVICE=CUDA if your site uses CUDA-aware MPI/ClimaComms.
cd "${SLURM_SUBMIT_DIR}" || exit 1
export VARIANCE_CALIB_WORKERS=${VARIANCE_CALIB_WORKERS:-14}

julia --project=. -e 'import Pkg; Pkg.instantiate()'
julia --project=. run_calibration.jl
