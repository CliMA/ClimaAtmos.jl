#!/usr/bin/env bash
# Submit `sweep_forward_array.sbatch` with a job array matching the current forward grid
# (`--baseline-only`, default registry and ladder params). Slurm CLI `--array` overrides
# any `#SBATCH --array` in the file; we do not edit tracked `.sbatch` files.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
N="$(julia --project="$ROOT" "$ROOT/scripts/sweep_forward_runs.jl" --print-task-count --baseline-only)"
[[ "$N" =~ ^[0-9]+$ ]] || {
    echo "submit_forward_sweep_baseline.sh: expected integer task count on stdout, got: $N" >&2
    exit 1
}
LAST=$((N - 1))
exec sbatch --array="0-${LAST}" "$ROOT/scripts/sweep_forward_array.sbatch"
