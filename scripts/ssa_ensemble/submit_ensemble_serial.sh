#!/usr/bin/env bash
#SBATCH --job-name=ssa_hist_serial
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/home/zglaser/autodebug/logs/ensemble_serial_%j.log
#
# ONE job that runs every site in a single Julia process (see
# run_ensemble_serial.jl for why). Generate the site configs first with
# make_site_configs.sh, then:
#   sbatch --export=SHA=<sha> scripts/ssa_ensemble/submit_ensemble_serial.sh
# WT / ENV default to the ~/autodebug probe layout for SHA (probe_setup.sh).
# Pass SITES="16 17 82" to run a subset; default is the full 24-site list.
set -euo pipefail
SHA="${SHA:?set SHA=<worktree sha> (or WT and ENV explicitly)}"
WT="${WT:-$HOME/autodebug/probe/wt_${SHA}}"
ENV="${ENV:-$HOME/autodebug/probe/env_${SHA}}"
SITES="${SITES:-$(seq -s ' ' 16 33) 82 92 94 96 99 100}"
CONFIGS=()
for s in $SITES; do CONFIGS+=("$WT/config/ensemble/ssa_history/sites/site${s}.yml"); done
module purge
module load climacommon/2025_12_22
unset LD_LIBRARY_PATH
export CLIMACOMMS_DEVICE=CPU
export CLIMACOMMS_CONTEXT=SINGLETON
export JULIA_NUM_THREADS=1
cd "$WT"
julia --project="$ENV" scripts/ssa_ensemble/run_ensemble_serial.jl "${CONFIGS[@]}" </dev/null
