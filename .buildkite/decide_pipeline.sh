#!/bin/bash
# Decider step for non-`main` climaatmos-ci builds (see .buildkite/pipeline.yml).
#
# Merge-queue fast path: when the merge group's only change over `main` is a
# single PR whose reproducibility bundle was already staged by that PR's build
# (see should_skip_merge_queue.sh), re-running the full pipeline would only
# reproduce an already-tested state. In that case upload nothing and let the
# build pass green -- a build must still report success for the merge queue to
# merge, but it need not repeat the PR's work.
#
# We deliberately do NOT run reproducibility_tests/stage_output.jl here, so the
# PR's staged bundle is left in place; the post-merge `main` build then publishes
# it via reproducibility_tests/publish_reference.jl exactly as in the off-queue
# flow. The "bundle gets copied on merge" behaviour is unchanged.
#
# Otherwise (ordinary PR build, or a merge group combining multiple PRs, or one
# with no staged results) upload the full CI pipeline.
set -euo pipefail

if .buildkite/should_skip_merge_queue.sh; then
    buildkite-agent annotate --style success --context skip-ci \
        "Merge-queue CI skipped: reused this PR's staged results." || true
    echo "Skipping full pipeline: reusing already-tested PR results."
    exit 0
fi

echo "--- Uploading full pipeline"
buildkite-agent pipeline upload .buildkite/full_pipeline.yml
