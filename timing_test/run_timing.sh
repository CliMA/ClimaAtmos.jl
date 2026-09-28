#!/usr/bin/env bash
# Chemistry timing test: run 6 matched simulations and collect SYPD vs #species.
#
# Two lines for the plot: bomex and aquaplanet; x-axis = number of chemistry
# species (0 = no chemistry, 3 = ABBA, 15 = JPM); y-axis = simulated years per day.
#
# CHEM mode (env var CHEM, default "on"):
#   on  -> Musica loaded; the per-cell MICM solve runs every step (full chemistry).
#   off -> Musica NOT loaded; the ρq_gas_* tracers are still created and ADVECTED,
#          but the chemistry solve is a no-op. This isolates tracer-transport cost
#          from chemistry-solve cost. (The 0-species runs are identical in both
#          modes -- they have no tracers -- and serve as a shared baseline.)
# Outputs are tagged by mode so the two sweeps don't clobber each other:
#   timing_results/sypd_vs_species_<chem|nochem>.csv  and  <job>_<chem|nochem>.log
#
# First invocation precompiles ClimaAtmos + Musica and builds the MICM artifacts,
# which can take a while. SYPD excludes JIT compilation (solve_atmos! warms up
# before timing).
#
# Run from anywhere:
#   bash timing_test/run_timing.sh              # full chemistry
#   CHEM=off bash timing_test/run_timing.sh     # tracers advected, solve off
set -uo pipefail

# Julia launcher. Override with e.g. JULIA="julia +1.11" if you have that channel.
JULIA="${JULIA:-julia +1.11.6}"

# Chemistry mode.
CHEM="${CHEM:-on}"
case "$(echo "$CHEM" | tr '[:upper:]' '[:lower:]')" in
  on)  export TIMING_CHEMISTRY=on;  TAG=chem ;;
  off) export TIMING_CHEMISTRY=off; TAG=nochem ;;
  *) echo "CHEM must be 'on' or 'off' (got '$CHEM')"; exit 1 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
DRIVER="$SCRIPT_DIR/timing_driver.jl"
OUTDIR="$SCRIPT_DIR/timing_results"
mkdir -p "$OUTDIR"
CSV="$OUTDIR/sypd_vs_species_${TAG}.csv"

cd "$REPO"

# platform  species-count  job_id
JOBS=(
  "bomex       0  timing_bomex_no_mechanism"
  "bomex       3  timing_bomex_abba"
  "bomex      15  timing_bomex_jpm"
  "aquaplanet  0  timing_aquaplanet_no_mechanism"
  "aquaplanet  3  timing_aquaplanet_abba"
  "aquaplanet 15  timing_aquaplanet_jpm"
)

echo "platform,n_species,chemistry,sypd,job_id" > "$CSV"
echo "=== CHEM=$CHEM (TIMING_CHEMISTRY=$TIMING_CHEMISTRY) -> tag '$TAG' ==="

for row in "${JOBS[@]}"; do
  read -r platform nspecies job <<< "$row"
  cfg="config/model_configs/${job}.yml"
  log="$OUTDIR/${job}_${TAG}.log"
  echo ">>> [$platform, $nspecies species, chem=$CHEM] $job"

  $JULIA --project=.buildkite "$DRIVER" \
      --config_file "$cfg" --job_id "$job" 2>&1 | tee "$log"

  # solve_atmos! logs e.g. "[ Info: sypd: 1.234" or "sypd: 0.005 (sdpd = 1.825)".
  # Grab the first numeric token after "sypd:".
  sypd=$(grep -oE 'sypd: [0-9.eE+-]+' "$log" | tail -1 | awk '{print $2}')
  if [[ -z "$sypd" ]]; then
    echo "!!! No SYPD found for $job (did it crash? see $log)"
    sypd="NA"
  fi
  echo "$platform,$nspecies,$CHEM,$sypd,$job" >> "$CSV"
  echo ">>> $job -> sypd=$sypd"
  echo
done

echo "Done. Results:"
cat "$CSV"
echo
echo "CSV: $CSV"
