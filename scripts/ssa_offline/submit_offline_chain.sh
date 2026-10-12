#!/bin/bash
# submit_offline_chain.sh <sha> <online_output_active_dir> <tag> [output_every_s] [big=50bin|24bin]
# One CPU job that runs offline_bins.jl twice in sequence on the same saved
# online states: first the 2-bin + SSLTH model (validation against the online
# run), then the 50-bin + SSLTH model (truth spectrum). Sequential so the
# cluster never sees more than one job for this.
set -euo pipefail
sha=$1; sdir=$2; tag=$3; every=${4:-600}; big=${5:-50bin}
AD=${AD:-$HOME/autodebug}
job=$AD/jobs/job_offline_chain_${tag}.sh
mkdir -p "$AD/jobs" "$AD/logs" "$AD/offline"
cat > "$job" <<EOJ
#!/bin/bash
#SBATCH --job-name=offc_$tag
#SBATCH --partition=batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=$AD/logs/offline_chain_${tag}_%j.log
module purge
module load climacommon/2025_12_22
unset LD_LIBRARY_PATH
export CLIMACOMMS_DEVICE=CPU
export CLIMACOMMS_CONTEXT=SINGLETON
export JULIA_NUM_THREADS=1
cd $AD/probe/wt_$sha || exit 9
echo "=== OFFLINE 2bin  \$(date)"
julia --project=$AD/probe/env_$sha $AD/bomex/offline_bins.jl $AD/bomex/offline_2bin_sslth.yml $sdir $AD/offline/${tag}_2bin.nc $every </dev/null
echo "=== OFFLINE $big \$(date)"
julia --project=$AD/probe/env_$sha $AD/bomex/offline_bins.jl $AD/bomex/offline_${big}_sslth.yml $sdir $AD/offline/${tag}_${big}.nc $every </dev/null
echo "=== CHAIN DONE \$(date)"
EOJ
chmod +x "$job"
sbatch "$job"
