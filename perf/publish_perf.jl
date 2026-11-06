#=
Publish step, run on `main` via the .buildkite/pipeline.yml dispatcher once a PR
merges: move the timings that the PR's `perf/benchmark.jl` jobs staged
(`perf/perf_tracking.jl`) to `PERF_STORE/<commit>/`, where later builds compare
against them.
=#
include(joinpath(@__DIR__, "..", "reproducibility_tests", "reproducibility_utils.jl"))
include(joinpath(@__DIR__, "perf_tracking.jl"))

pr_number = discover_pr_number()
commit = get_commit_sha()
if isnothing(pr_number)
    @warn "Perf: could not determine the merged PR for $commit; no timings published."
else
    result = publish_staged_perf(pr_number, commit)
    @info "Perf: $result timings of PR #$pr_number for $commit."
end
