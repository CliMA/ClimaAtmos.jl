# End-of-build reproducibility handling for non-`main` `climaatmos-ci` builds.
# (On `main`, the .buildkite/pipeline.yml dispatcher runs publish_reference.jl.)
#
#   - PR / feature build: stage the bundle to `…/climaatmos-main-staging/pr-<n>/`
#     for the post-merge publish; nothing is published yet.
#   - Merge-queue build: its commit is the merge candidate, so publish directly
#     and drop the now-redundant PR staging.
#
# See reproducibility_tests/README.md ("Publishing references on merge").
include(joinpath(@__DIR__, "reproducibility_utils.jl"))

job_ids = discover_reproducibility_job_ids()
buildkite_ci = get(ENV, "BUILDKITE_PIPELINE_SLUG", nothing) == "climaatmos-ci"
branch = get(ENV, "BUILDKITE_BRANCH", "")
in_merge_queue = startswith(branch, "gh-readonly-queue/main/")
pr_number = buildkite_ci ? discover_pr_number() : nothing

if buildkite_ci && in_merge_queue
    move_data_to_save_dir(; buildkite_ci, in_merge_queue, dirs_src = job_ids)
    prune_reference_store()
    # The PR staging is now published directly, so the `main` build must not
    # publish it again.
    isnothing(pr_number) ||
        rm(pr_staging_dir(pr_number); recursive = true, force = true)
elseif !isnothing(pr_number) && !isempty(job_ids)
    staged = stage_pr_data(; dirs_src = job_ids, pr_number)
    @info "Repro: staged PR #$pr_number reproducibility bundle for publishing on merge." staged
elseif isempty(job_ids)
    @info "Repro: no reproducibility bundles found; nothing to stage."
elseif buildkite_ci
    @warn "Repro: could not determine the PR number (BUILDKITE_PULL_REQUEST); " *
          "bundle not staged, so no reference will be published on merge."
end
