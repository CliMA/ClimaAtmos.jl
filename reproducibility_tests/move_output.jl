include(joinpath(@__DIR__, "reproducibility_utils.jl"))
include(joinpath(@__DIR__, "reproducibility_test_job_ids.jl"))

job_ids = reproducibility_test_job_ids()

buildkite_ci = get(ENV, "BUILDKITE_PIPELINE_SLUG", nothing) == "climaatmos-ci"
in_merge_queue =
    startswith(get(ENV, "BUILDKITE_BRANCH", nothing), "gh-readonly-queue/main/")
cluster_data_prefix = "/resnick/scratch/esm/slurm-buildkite/climaatmos-main"

move_data_to_save_dir(;
    buildkite_ci,
    in_merge_queue,
    dirs_src = job_ids,
    ref_counter_file_PR = joinpath(@__DIR__, "ref_counter.jl"),
)

if buildkite_ci && in_merge_queue
    folders = get_reference_dirs_to_delete(; root_dir = cluster_data_prefix)
    bins = compute_bins(folders)
    if !isempty(folders)
        msg = prod(x -> "    $x\n", folders)
        @warn "Repro: deleting folders:\n$msg"
    end
    @warn "Deleted folder bins:\n $(string_bins(bins))"
    for f in folders
        rm(f; recursive = true, force = true)
    end
end
