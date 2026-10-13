#!/bin/bash
# GitHub Actions merge-queue fast path. Companion to
# .buildkite/should_skip_merge_queue.sh, applying the same reasoning to the
# GitHub-hosted checks: the unit-test matrix (.github/workflows/ci.yml) and the
# doc build (.github/workflows/docs.yml).
#
# Exit 0  => the merge group is provably identical to an already-green PR head,
#            so this workflow's real jobs can be skipped and a trivial gate job
#            reports the required check green.
# Exit non-zero => run the real jobs.
#
# Usage: should_skip_merge_queue.sh <check-name-prefix>
#   <check-name-prefix> selects which of the PR's checks must already be green
#   for this workflow, e.g. "ci " for the ci matrix or "docbuild" for the docs.
#
# All of the following must hold to skip. ANY uncertainty (not a merge-queue
# build, unparseable ref, failed fetch/API call, missing or non-green check)
# exits non-zero and runs the full jobs, so a wrong guess can never merge
# untested code:
#
#   1. This is a GitHub merge-queue build: GITHUB_EVENT_NAME == "merge_group"
#      and the head ref has the form
#          gh-readonly-queue/main/pr-<pr_number>-<base_sha>
#
#   2. <base_sha> (from the merge_group payload) is a real commit on `main`
#      (git merge-base --is-ancestor), i.e. the merge group stacks exactly one
#      PR on `main`. Groups combining multiple PRs stack on a synthetic
#      gh-readonly-queue commit that never reaches `main` and fail this check.
#      (Same argument as the Buildkite script; see its header for detail.)
#
#   3. The merge group's tree equals the PR head's tree, i.e. the merged code is
#      byte-identical to what the PR already tested. This is the "up to date
#      with origin/main" condition: it holds only when `main` has not advanced
#      past what the PR contains. Compares tree content, so it holds for the
#      squash, rebase, and merge-commit queue methods alike.
#
#   4. Every one of the PR's own checks whose name starts with <prefix> already
#      concluded successfully (queried with `gh pr checks`, which reports the
#      checks GitHub shows for the PR head and is agnostic to how Actions
#      associates runs). This is the GitHub analogue of the Buildkite
#      "staged bundle exists" condition: it confirms the PR build actually ran
#      these checks to green on the identical code.
set -euo pipefail

prefix="${1:?usage: should_skip_merge_queue.sh <check-name-prefix>}"

# (1) merge-queue build + parse the PR number from the head ref.
[[ "${GITHUB_EVENT_NAME:-}" == "merge_group" ]] || exit 1
head_ref="${MERGE_GROUP_HEAD_REF:-}"
base_sha="${MERGE_GROUP_BASE_SHA:-}"
[[ "$head_ref" =~ pr-([0-9]+)-[0-9a-fA-F]+$ ]] || exit 1
pr_number="${BASH_REMATCH[1]}"
[[ -n "$base_sha" ]] || exit 1

# (2) <base_sha> must be a real commit on `main` (single-PR delta).
git fetch -q origin main || exit 1
git merge-base --is-ancestor "$base_sha" FETCH_HEAD || exit 1

# (3) merge-group tree must equal the PR head tree (up to date, nothing new to
#     test). HEAD is the merge-group commit checked out by actions/checkout.
git fetch -q origin "refs/pull/${pr_number}/head" || exit 1
pr_head="$(git rev-parse FETCH_HEAD)" || exit 1
git diff --quiet "$pr_head" HEAD || exit 1

# (4) every matching check on the PR already concluded success. `gh pr checks`
#     exits non-zero when checks are pending/failing, so read the JSON (still
#     emitted) and evaluate it ourselves; require >=1 match, all buckets "pass".
checks="$(gh pr checks "$pr_number" --repo "$GITHUB_REPOSITORY" \
            --json name,bucket 2>/dev/null || true)"
[[ -n "$checks" ]] || exit 1
echo "$checks" | jq -e --arg p "$prefix" \
  '[.[] | select(.name | startswith($p))] as $m
   | ($m | length) > 0 and ($m | all(.bucket == "pass"))' >/dev/null || exit 1

exit 0
