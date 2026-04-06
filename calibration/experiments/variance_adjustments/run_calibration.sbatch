#!/bin/bash
#SBATCH --job-name=variance_calib
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=variance_calibration_%j.out

# GPU (optional): e.g. --gres=gpu:1 and export CLIMACALIB_DEVICE=CUDADevice (see README).
cd "${SLURM_SUBMIT_DIR}" || exit 1
export VARIANCE_CALIB_WORKERS=${VARIANCE_CALIB_WORKERS:-14}

# Full pipeline (reference truth + EKI) in one Julia session; set VA_SKIP_REFERENCE=1 if y already exists.
julia --project=. run_e2e.jl
