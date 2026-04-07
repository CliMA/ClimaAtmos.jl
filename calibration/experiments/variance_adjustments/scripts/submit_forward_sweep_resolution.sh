#!/usr/bin/env bash
# Submit `sweep_forward_array_resolution.sbatch` with `--array` from the resolution-ladder
# task count (default registry and ladder params).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
N="$(julia --project="$ROOT" "$ROOT/scripts/sweep_forward_runs.jl" --print-task-count --resolution-ladder)"
[[ "$N" =~ ^[0-9]+$ ]] || {
    echo "submit_forward_sweep_resolution.sh: expected integer task count on stdout, got: $N" >&2
    exit 1
}
LAST=$((N - 1))
exec sbatch --array="0-${LAST}" "$ROOT/scripts/sweep_forward_array_resolution.sbatch"
