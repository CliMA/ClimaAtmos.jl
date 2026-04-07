#!/usr/bin/env bash
# Single entry: reference + EKI for the active experiment YAML (see run_e2e.jl --help).
# Optional: JULIA_MAIN_THREADS (default 4) sets julia -t for the driver process only.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec julia --project="$ROOT" -t "${JULIA_MAIN_THREADS:-4}" "$ROOT/run_e2e.jl" "$@"
