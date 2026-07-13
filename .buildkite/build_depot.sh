#!/bin/bash
# Builds the shared, read-only Julia depot for climaatmos-ci and atomically
# publishes it. Reads SHARED_DEPOT_ROOT and JULIA_DEPOT_PATH from the environment
# (set in depot_pipeline.yml). See that file for the overall design.
set -euo pipefail

STAGING="$JULIA_DEPOT_PATH"                 # Julia builds here
mkdir -p "$STAGING"

# Start from the last published depot so we only build the delta and stay
# self-contained. First run has no `shared` symlink -> build from scratch.
if [ -e "$SHARED_DEPOT_ROOT/shared" ]; then
  cp -a "$(readlink -f "$SHARED_DEPOT_ROOT/shared")/." "$STAGING/"
  chmod -R u+w "$STAGING"                   # published copy is read-only
fi

echo "--- Instantiate + precompile"
julia --project=.buildkite -e 'using Pkg; Pkg.instantiate(); Pkg.precompile(strict=true)'

echo "--- Lock and publish (atomic symlink swap)"
chmod -R a-w "$STAGING"
cd "$SHARED_DEPOT_ROOT"
ln -sfn "versions/$(basename "$STAGING")" shared.new
mv -Tf shared.new shared

echo "--- Prune old versions (keep 3)"
cd versions
for old in $(ls -1dt */ 2>/dev/null | tail -n +4 || true); do
  chmod -R u+w "$old"; rm -rf "$old"
done
