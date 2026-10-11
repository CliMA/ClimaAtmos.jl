#!/usr/bin/env bash
#SBATCH --job-name=ssa_hist
#SBATCH --partition=batch
#SBATCH --array=0-23
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#
# Slurm ARRAY job: one CPU task per cfsite of the sea-salt history ensemble.
#
# Submit (from anywhere):
#   sbatch [--export=SHA=<sha>,WT=...,ENV=...] scripts/ssa_ensemble/submit_ensemble.sh
#
# WT / ENV default to the ~/autodebug probe layout for a given SHA:
#   WT  = $HOME/autodebug/probe/wt_<sha>   (ClimaAtmos worktree with the configs)
#   ENV = $HOME/autodebug/probe/env_<sha>  (instantiated Julia project)
# Logs: $HOME/autodebug/logs/ensemble_<site>_<jobid>_<taskid>.log
# Output: $WT/output/<job_id>/output_active (ClimaAtmos default `output_dir`).
set -euo pipefail

# Keep this list in sync with SITES in make_site_configs.sh (and --array above).
SITES=($(seq 16 33) 82 92 94 96 99 100)
SITE="${SITES[$SLURM_ARRAY_TASK_ID]}"

SHA="${SHA:?set SHA=<worktree sha> (or WT and ENV explicitly)}"
WT="${WT:-$HOME/autodebug/probe/wt_${SHA}}"
ENV="${ENV:-$HOME/autodebug/probe/env_${SHA}}"
LOG_DIR="$HOME/autodebug/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/ensemble_site${SITE}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"

CONFIG="$WT/config/ensemble/ssa_history/sites/site${SITE}.yml"
JOB_ID="ssa_history_site${SITE}"

module purge
module load climacommon/2025_12_22
unset LD_LIBRARY_PATH
export CLIMACOMMS_DEVICE=CPU
export CLIMACOMMS_CONTEXT=SINGLETON
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

{
    echo "site=$SITE config=$CONFIG job_id=$JOB_ID WT=$WT ENV=$ENV host=$(hostname) $(date)"
    [ -f "$CONFIG" ] || { echo "missing $CONFIG (run make_site_configs.sh)"; exit 1; }
    cd "$WT"
    # ci_driver.jl CLI (src/config/cli_options.jl): --config_file (repeatable), --job_id.
    julia --project="$ENV" "$WT/.buildkite/ci_driver.jl" \
        --config_file "$CONFIG" --job_id "$JOB_ID"
} > "$LOG" 2>&1
