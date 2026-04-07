#!/bin/bash
#SBATCH --job-name=va_full_study
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=va_full_study_%j.out

# Submit from calibration/experiments/variance_adjustments
cd "${SLURM_SUBMIT_DIR}" || exit 1

# `run_full_study.jl` runs EKI + naive + ~80 forwards (merged θ, forward_eki/) + figures by default.
# Exploratory SCM-only ladder: --forward-baseline-scm. Optional: --skip-forward  --forward-baseline-only  --skip-naive  --forward-skip-done  --naive-skip-done
export VARIANCE_CALIB_WORKERS="${VARIANCE_CALIB_WORKERS:-14}"

julia --project=. -e 'import Pkg; Pkg.instantiate()'
julia --project=. run_full_study.jl --forward-skip-done
