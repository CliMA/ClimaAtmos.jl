#!/bin/bash
#SBATCH --job-name=va_calib_sweep
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# One task per entry in CALIB_SWEEP_CONFIGS inside run_calibration_sweep.jl (default: 3).
#SBATCH --array=0-2

cd "${SLURM_SUBMIT_DIR}" || exit 1
export CALIB_SWEEP_TASK_ID="${SLURM_ARRAY_TASK_ID}"

julia --project=. -e 'import Pkg; Pkg.instantiate()'
julia --project=. scripts/run_calibration_sweep.jl
