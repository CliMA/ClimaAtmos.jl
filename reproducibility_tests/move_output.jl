
include(joinpath(@__DIR__, "reproducibility_utils.jl"))
paths = latest_comparable_paths()

all_lines = readlines(joinpath(@__DIR__, "mse_tables.jl"))
lines = deepcopy(all_lines)
filter!(x -> occursin("] = OrderedCollections", x), lines)
job_ids = getindex.(split.(lines, "\""), 2)
@assert count(x -> occursin("OrderedDict", x), all_lines) == length(job_ids) + 1
@assert length(job_ids) ≠ 0 # safety net

# Note: cluster_data_prefix is also defined in compute_mse.jl
cluster_data_prefix = "/central/scratch/esm/slurm-buildkite/climaatmos-main"
buildkite_ci = get(ENV, "BUILDKITE_PIPELINE_SLUG", nothing) == "climaatmos-ci"
commit = get(ENV, "BUILDKITE_COMMIT", nothing)
branch = get(ENV, "BUILDKITE_BRANCH", nothing)
in_merge_queue = startswith(branch, "gh-readonly-queue/main/")
if buildkite_ci
    @info "pwd() = $(pwd())"
    @info "branch = $(branch)"
    @info "commit = $(commit)"

    using Glob
    @show readdir(joinpath(@__DIR__, ".."))
    # if a contributor manually merged, we still want to move data
    # from scratch to `cluster_data_prefix`. So, let's also try moving
    # data if this is running on the main branch.
    if in_merge_queue || branch == "main"
        commit_sha = commit[1:7]
        mkpath(cluster_data_prefix)
        path = joinpath(cluster_data_prefix, commit_sha)
        mkpath(path)
        # Always move reproducibility data, so that we
        # can compare against multiple references
        for folder_name in job_ids
            src = folder_name
            dst = joinpath(path, folder_name)
            @info "Moving $src to $dst"
            if !isfile(dst)
                mv(src, dst; force = true)
            end
        end
        ref_counter_file_PR = joinpath(@__DIR__, "ref_counter.jl")
        ref_counter_file_main = joinpath(path, "ref_counter.jl")
        if !isfile(ref_counter_file_main)
            mv(ref_counter_file_PR, ref_counter_file_main; force = true)
        end
        perf_benchmarks_PR = joinpath(dirname(@__DIR__), "perf_benchmarks.json")
        perf_benchmarks_main = joinpath(path, "perf_benchmarks.json")
        if !isfile(perf_benchmarks_main)
            mv(perf_benchmarks_PR, perf_benchmarks_main; force = true)
        end
        println("New reference folder: $path")
        for (root, dirs, files) in walkdir(path)
            println("--Directories in $root")
            for dir in dirs
                println("    ", joinpath(root, dir)) # path to directories
            end
            println("--Files in $root")
            for file in files
                println("    ", joinpath(root, file)) # path to files
            end
        end
    end
else
    @info "ENV keys: $(keys(ENV))"
end

if buildkite_ci && in_merge_queue
    cleanup_central(cluster_data_prefix)
end
