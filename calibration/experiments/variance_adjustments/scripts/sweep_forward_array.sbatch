#!/bin/bash
#SBATCH --job-name=va_sweep
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#
# No #SBATCH --array here: pass it at submit time so it always matches `flatten_sweep_tasks`.
# Slurm: options on the `sbatch` command line override directives in this file.
#
#   ./scripts/submit_forward_sweep_baseline.sh
#
# or manually (same flags as in `julia ...` below for counting):
#   N=$(julia --project=. scripts/sweep_forward_runs.jl --print-task-count --baseline-only)
#   sbatch --array=0-$((N-1)) scripts/sweep_forward_array.sbatch
#
# Submit from this experiment directory (or set SLURM_SUBMIT_DIR accordingly).

cd "${SLURM_SUBMIT_DIR}" || exit 1

julia --project=. -e 'import Pkg; Pkg.instantiate()'
julia --project=. scripts/sweep_forward_runs.jl --baseline-only --skip-done
