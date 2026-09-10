#!/bin/bash
# Builds the shared, read-only Julia depot for climaatmos-ci and atomically
# publishes it. Reads SHARED_DEPOT_ROOT and JULIA_DEPOT_PATH from the environment
# (set in depot_pipeline.yml). See that file for the overall design.
set -euo pipefail

STAGING="$JULIA_DEPOT_PATH"                 # Julia builds here
mkdir -p "$STAGING"

# Start from the last published depot so we only build the delta and stay
# self-contained. If `shared` symlink is missing while the version trees it
# pointed at are intact, fall back to the newest of those instead of
# building the whole depot.
BASE="$(readlink -f "$SHARED_DEPOT_ROOT/shared" 2>/dev/null || true)"
if [ ! -d "$BASE" ]; then
  latest=$(find "$SHARED_DEPOT_ROOT/versions" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null |
           grep -Ex '[0-9]+' | grep -vx "$(basename "$STAGING")" | sort -n | tail -1)
  BASE="${latest:+$SHARED_DEPOT_ROOT/versions/$latest}"
fi
if [ -n "$BASE" ] && [ -d "$BASE" ]; then
  echo "--- Seeding staging depot from $BASE"
  cp -a "$BASE/." "$STAGING/"
  chmod -R u+w "$STAGING"                   # published copy is read-only
  # A corrupted depot (e.g. a version that was `rm -rf`'d while consumers had
  # its files open on NFS) leaves empty artifact dirs. Pkg treats an existing
  # artifact dir as installed and never re-downloads it, so every JLL would then
  # fail at dlopen. Drop any artifact dir with no files in it.
  if [ -d "$STAGING/artifacts" ]; then
    find "$STAGING/artifacts" -mindepth 1 -maxdepth 1 -type d \
      -exec sh -c '[ -z "$(find "$1" -type f -print -quit)" ]' _ {} \; \
      -print -exec rm -rf {} + | sed 's/^/Removing hollow artifact: /'
  fi
fi

echo "--- Instantiate + precompile"
julia --project=.buildkite -e 'using Pkg; Pkg.Registry.update(); Pkg.instantiate(); Pkg.precompile(strict=true)'

# Also bake --check-bounds=yes caches. The CI "Checkbounds" step runs
# perf/benchmark.jl under --check-bounds=yes, Julia folds the check-bounds mode
# into the precompile cache slug, so the default (auto) caches above are stale
# for that step and it would otherwise recompile the whole tree into its
# per-build depot. These caches use a distinct slug and coexist with the ones
# above, letting that step read the shared depot instead of rebuilding it.
echo "--- Precompile (--check-bounds=yes)"
julia --check-bounds=yes --project=.buildkite -e 'using Pkg; Pkg.precompile(strict=true)'

echo "--- Lock and publish (atomic symlink swap)"
# Make everything readable before removing write access. Julia copies the
# permission bits of a package's source file onto its compiled cache file, so
# if a source file has no read bits the cache file has none either. Removing
# write access alone would leave that cache file with no permissions at all,
# and every CI job would fail with "Permission denied" when Julia tries to
# open it while looking for caches to load.
chmod -R a+rX "$STAGING"
chmod -R a-w "$STAGING"
cd "$SHARED_DEPOT_ROOT"
ln -sfn "versions/$(basename "$STAGING")" shared.new
mv -Tf shared.new shared

echo "--- Prune old versions (keep 3)"
# Sort by build number, because mtime is unreliable. Never prune whatever
# `shared` currently points at.
published=$(basename "$(readlink -f shared)")
cd versions
for old in $(find . -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | grep -Ex '[0-9]+' | sort -n | head -n -3); do
  [ "$old" = "$published" ] && continue
  # Deleting can fail with "Directory not empty" if a CI job on this node
  # still has files from the old version open. On the NFS filesystem those
  # files are renamed instead of removed until the job closes them, so their
  # directories cannot be deleted yet. The new depot is already published, so
  # this must not fail the build. The next run deletes what is left over.
  chmod -R u+w "$old"; rm -rf "$old" || true
done
