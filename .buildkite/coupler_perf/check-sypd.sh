#!/usr/bin/env bash

SYPD_FILE=output/gpu_amip_progedmf_1M_land_he16/artifacts/sypd.txt

if [[ ! -f "$SYPD_FILE" ]]; then
    echo "❌ SYPD file not found: $SYPD_FILE"
    exit 1
fi

SYPD=$(cat "$SYPD_FILE" | tr -d '[:space:]')

if [[ -z "$SYPD" ]]; then
    echo "❌ SYPD file is empty"
    exit 1
fi

echo "SYPD: $SYPD"

PERCENT_CHANGE=$(echo "scale=2; (($SYPD - $BASELINE_SYPD) / $BASELINE_SYPD) * 100" | bc)

if (( $(echo "$PERCENT_CHANGE <= $MIN_PERCENT_CHANGE" | bc -l) )); then
    echo "❌ SYPD changed by $PERCENT_CHANGE% (threshold: $MIN_PERCENT_CHANGE%)"
    exit 1
else
    echo "✅ SYPD change ($PERCENT_CHANGE%) is okay (threshold: $MIN_PERCENT_CHANGE%)"
fi
