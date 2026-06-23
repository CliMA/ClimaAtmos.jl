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

# Fetch the baseline from the last successful main build, fall back to BASELINE_SYPD env var.
FETCHED_BASELINE=$(curl -sf \
    -H "Authorization: Bearer ${BUILDKITE_API_TOKEN}" \
    "https://api.buildkite.com/v2/organizations/${BUILDKITE_ORGANIZATION_SLUG}/pipelines/${BUILDKITE_PIPELINE_SLUG}/builds?branch=main&state=passed&per_page=1" \
    | jq -r '.[0].meta_data.baseline_sypd // empty')

if [[ -n "$FETCHED_BASELINE" ]]; then
    echo "📥 Using baseline_sypd=$FETCHED_BASELINE from last passing main build"
    BASELINE_SYPD="$FETCHED_BASELINE"
else
    echo "⚠️  Could not fetch baseline from main; using fallback BASELINE_SYPD=$BASELINE_SYPD"
fi

if [[ -z "$BASELINE_SYPD" ]]; then
    echo "❌ No baseline SYPD available (API fetch failed and BASELINE_SYPD env var is unset)"
    exit 1
fi

PERCENT_CHANGE=$(echo "scale=2; (($SYPD - $BASELINE_SYPD) / $BASELINE_SYPD) * 100" | bc)

if [[ "${BUILDKITE_BRANCH}" == "main" ]]; then
    # On main, always record the new baseline regardless of perf change.
    echo "✅ SYPD change ($PERCENT_CHANGE%) (threshold: $MIN_PERCENT_CHANGE%)"
    buildkite-agent meta-data set baseline_sypd "$SYPD"
    echo "📌 Stored baseline_sypd=$SYPD in build metadata"
else
    if (( $(echo "$PERCENT_CHANGE <= $MIN_PERCENT_CHANGE" | bc -l) )); then
        echo "❌ SYPD changed by $PERCENT_CHANGE% (threshold: $MIN_PERCENT_CHANGE%)"
        exit 1
    fi
    echo "✅ SYPD change ($PERCENT_CHANGE%) is okay (threshold: $MIN_PERCENT_CHANGE%)"
fi
