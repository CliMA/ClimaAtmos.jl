#!/bin/bash
#SBATCH --job-name=va_sweep_res
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/slurm/va_sweep_res_%A_%a.out
#
# No #SBATCH --array — pass `--array=0-$((N-1))` at submit (see submit script or note below).
#
#   ./scripts/submit_forward_sweep_resolution.sh
#
# or:
#   N=$(julia --project=. scripts/sweep_forward_runs.jl --print-task-count --resolution-ladder)
#   sbatch --array=0-$((N-1)) scripts/sweep_forward_array_resolution.sbatch

cd "${SLURM_SUBMIT_DIR}" || exit 1
mkdir -p logs/slurm

julia --project=. -e 'import Pkg; Pkg.instantiate()'
julia --project=. scripts/sweep_forward_runs.jl --resolution-ladder --skip-done
