#!/bin/bash
# Decide whether the merge-queue build can reuse an already-tested PR's results.
#
# Exit 0  => skip full CI (reuse). Exit non-zero => run full CI.
#
# All of the following must hold to skip. ANY uncertainty (unparseable branch,
# failed fetch, missing staging) falls through to running the full pipeline, so
# a wrong guess can never merge untested code:
#
#   1. This is a GitHub merge-queue build. Its branch has the form
#          gh-readonly-queue/main/pr-<pr_number>-<base_sha>
#      where <base_sha> is the commit the merge group is stacked on.
#
#   2. <base_sha> is a real commit on `main` (checked below with `is-ancestor`).
#      This is how we know the merge group is exactly ONE PR on top of `main`:
#
#      The merge queue tests PRs by stacking them. The group's <base_sha> is
#      whatever it was built on top of:
#        - front of the queue  -> stacked directly on `main`, so <base_sha> is
#          a commit on `main`.
#        - a PR behind another  -> stacked on the PR-ahead's temporary
#          gh-readonly-queue commit, which is synthetic and never reaches
#          `main`, so <base_sha> is NOT on `main`.
#
#      So "<base_sha> is on `main`" holds only when nothing is stacked beneath
#      this PR -- i.e. the group adds exactly this one PR. Groups that combine
#      multiple PRs fail this check and are never reused.
#
#   3. The merge group's tree equals the PR head's tree, i.e. the merged code is
#      identical to what the PR build tested and staged. Needed to reject a stale
#      PR: if `main` advanced past what the PR contains, the merged code differs
#      from the staged bundle, the trees differ, and full CI runs. Compares tree
#      content, not commit structure, so it works for the squash, rebase, and
#      merge-commit queue methods alike.
#
#   4. That PR already has a non-empty staged reproducibility bundle from its PR
#      build. stage_pr_data() overwrites this on every push, so it reflects the
#      PR head now being merged.
#
# Note there is NO stale-`main` correctness gap: condition 3 skips only when the
# merged code matches the staged code exactly, so the merge-queue reproducibility
# test is skipped only when re-running it is provably a no-op.
#
# NOTE: staging_root must match reproducibility_tests/reproducibility_utils.jl
# (STAGING_ROOT / pr_staging_dir); keep the two in sync.
set -euo pipefail

branch="${BUILDKITE_BRANCH:-}"

# (1) merge-queue branch + parse PR number and base sha.
[[ "$branch" =~ ^gh-readonly-queue/main/pr-([0-9]+)-([0-9a-fA-F]+)$ ]] || exit 1
pr_number="${BASH_REMATCH[1]}"
base_sha="${BASH_REMATCH[2]}"

# (2) <base_sha> must be a real commit on `main` (single-PR delta).
git fetch -q origin main || exit 1
git merge-base --is-ancestor "$base_sha" FETCH_HEAD || exit 1

# (3) the merge-group code must equal the PR head that was staged. Fetch the PR
#     head (GitHub exposes it at refs/pull/<n>/head) and require its tree to
#     match the checked-out merge commit (HEAD). This compares content, so it
#     holds for the squash, rebase, and merge-commit queue methods alike. Fails
#     (-> full CI) if pull refs are not fetchable.
git fetch -q origin "refs/pull/${pr_number}/head" || exit 1
pr_head="$(git rev-parse FETCH_HEAD)" || exit 1
git diff --quiet "$pr_head" HEAD || exit 1

# (4) staged reproducibility bundle for this PR exists and is non-empty.
#     SKIP_CI_STAGING_ROOT overrides the path for tests; production uses the
#     default below.
staging_root="${SKIP_CI_STAGING_ROOT:-/resnick/scratch/esm/slurm-buildkite/climaatmos-main-staging}"
bundle="${staging_root}/pr-${pr_number}/reproducibility_bundle"
{ [ -d "$bundle" ] && [ -n "$(ls -A "$bundle" 2>/dev/null)" ]; } || exit 1

exit 0
