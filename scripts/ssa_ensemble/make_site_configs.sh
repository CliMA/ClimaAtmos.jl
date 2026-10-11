#!/usr/bin/env bash
# Generate one YAML per cfsite from the ensemble base config.
#
# Usage: scripts/ssa_ensemble/make_site_configs.sh [BASE_YML] [OUT_DIR]
#   BASE_YML defaults to config/ensemble/ssa_history/base_gcmdriven_ssa.yml
#   OUT_DIR  defaults to config/ensemble/ssa_history/sites
#
# Sites: GPCI transect 16-33 (Shen et al. 2022) and Southern Ocean 82, 92,
# 94, 96, 99, 100 — 24 columns. The forcing artifact
# cfsite_gcm_forcing/HadGEM2-A_amip.2004-2008.07.nc carries groups site2..site115.
# Only `cfsite_number` and `job_id` differ between sites; everything else is
# inherited verbatim so a change to the base file re-generates the ensemble.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_YML="${1:-$REPO_ROOT/config/ensemble/ssa_history/base_gcmdriven_ssa.yml}"
OUT_DIR="${2:-$REPO_ROOT/config/ensemble/ssa_history/sites}"

# Keep this list in sync with SITES in submit_ensemble.sh.
SITES="$(seq 16 33) 82 92 94 96 99 100"

mkdir -p "$OUT_DIR"
for site in $SITES; do
    out="$OUT_DIR/site${site}.yml"
    sed -e "s|^cfsite_number:.*|cfsite_number: \"site${site}\"|" \
        -e "s|^job_id:.*|job_id: \"ssa_history_site${site}\"|" \
        "$BASE_YML" > "$out"
    grep -q "^cfsite_number: \"site${site}\"" "$out" || { echo "cfsite_number not set in $out" >&2; exit 1; }
    grep -q "^job_id: \"ssa_history_site${site}\"" "$out" || { echo "job_id not set in $out" >&2; exit 1; }
    echo "wrote $out"
done
