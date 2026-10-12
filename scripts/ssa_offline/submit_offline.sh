#!/bin/bash
# submit_offline.sh <sha> <offline_config.yml> <online_output_active_dir> <tag> [output_every_s]
# One CPU job running offline_bins.jl from the probe worktree of <sha> (the
# configs reference toml/prognostic_edmfx_1M.toml relative to that directory).
set -euo pipefail
sha=$1; cfg=$2; sdir=$3; tag=$4; every=${5:-600}
AD=${AD:-$HOME/autodebug}
job=$AD/jobs/job_offline_${tag}.sh
mkdir -p "$AD/jobs" "$AD/logs" "$AD/offline"
cat > "$job" <<EOJ
#!/bin/bash
#SBATCH --job-name=off_$tag
#SBATCH --partition=batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=$AD/logs/offline_${tag}_%j.log
module purge
module load climacommon/2025_12_22
unset LD_LIBRARY_PATH
export CLIMACOMMS_DEVICE=CPU
export CLIMACOMMS_CONTEXT=SINGLETON
export JULIA_NUM_THREADS=1
cd $AD/probe/wt_$sha || exit 9
julia --project=$AD/probe/env_$sha $AD/bomex/offline_bins.jl $cfg $sdir $AD/offline/${tag}.nc $every </dev/null
EOJ
chmod +x "$job"
sbatch "$job"
