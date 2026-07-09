#=
Publish step, run on `main` via the .buildkite/pipeline.yml dispatcher once a PR
merges: move the bundle `stage_output.jl` staged during the PR build into the
reference store that later PRs compare against, then prune. No simulations are
re-run. Idempotent — a no-op if a merge-queue build already published the commit.

See reproducibility_tests/README.md ("Publishing references on merge").
=#
include(joinpath(@__DIR__, "reproducibility_utils.jl"))

merge_commit = get(ENV, "BUILDKITE_COMMIT", "")
isempty(merge_commit) &&
    error("Repro: BUILDKITE_COMMIT is unset; cannot determine the reference commit.")

# Fail loudly rather than silently skipping: on `main` the merged PR must be
# identifiable, else no reference is ever published and the failure is invisible.
pr_number = discover_pr_number()
isnothing(pr_number) && error(
    "Repro: could not determine the merged PR for $merge_commit; no reference " *
    "published. Expected BUILDKITE_PULL_REQUEST, a gh-readonly-queue branch, or " *
    "a GitHub merge/squash commit message.",
)

commit = get_commit_sha(; commit = merge_commit)
result = publish_staged_reference(pr_number, commit)
if result == :already_published
    @info "Repro: reference for $commit already exists; nothing to publish."
elseif result == :no_staged_bundle
    # Legitimate when the PR ran no reproducibility jobs; a warning (not an
    # error) so such merges stay green.
    @warn "Repro: no staged bundle for PR #$pr_number; nothing to publish."
else
    @info "Repro: published PR #$pr_number bundle to the reference store." commit
end
