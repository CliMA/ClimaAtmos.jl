
include(joinpath(@__DIR__, "self_reference_or_path.jl"))
self_reference = self_reference_or_path() == :self_reference

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
    if in_merge_queue
        commit_sha = commit[1:7]
        mkpath(cluster_data_prefix)
        path = joinpath(cluster_data_prefix, commit_sha)
        mkpath(path)
        # Only move reproducibility data if self reference:
        if self_reference
            for folder_name in job_ids
                src = folder_name
                dst = joinpath(path, folder_name)
                @info "Moving $src to $dst"
                mv(src, dst; force = true)
            end
            ref_counter_file_PR = joinpath(@__DIR__, "ref_counter.jl")
            ref_counter_file_main = joinpath(path, "ref_counter.jl")
            mv(ref_counter_file_PR, ref_counter_file_main; force = true)
        end
        perf_benchmarks_PR = joinpath(dirname(@__DIR__), "perf_benchmarks.json")
        perf_benchmarks_main = joinpath(path, "perf_benchmarks.json")
        mv(perf_benchmarks_PR, perf_benchmarks_main; force = true)
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

function reason(path)
    f = joinpath(path, "ref_counter.jl")
    if !isfile(f)
        return "ref_counter.jl does not exist"
    else
        ref_counter = parse(Int, first(readlines(f)))
        return "ref_counter: $ref_counter"
    end
end

function cleanup_central(cluster_data_prefix)
    @warn "Cleaning up old files on central"
    # Get (sorted) array of paths, `pop!(sorted_paths)`
    # is the most recent merged folder.
    sorted_paths = sorted_dataset_folder(; dir = cluster_data_prefix)
    keep_latest_n = 0
    keep_latest_ref_counters = 5
    if !isempty(sorted_paths)
        N = length(sorted_paths) - keep_latest_n
        paths_to_delete = []
        ref_counters_main = ref_counters_per_path(sorted_paths)
        i_largest_reference = argmax(ref_counters_main)
        path = sorted_paths[i_largest_reference]
        ref_counter_file_main = joinpath(path, "ref_counter.jl")
        @assert isfile(ref_counter_file_main)
        ref_counter_main = parse(Int, first(readlines(ref_counter_file_main)))

        for i in 1:N
            path = sorted_paths[i]
            ref_counter_file = joinpath(path, "ref_counter.jl")
            if !isfile(ref_counter_file)
                push!(paths_to_delete, path)
            else
                ref_counter = parse(Int, first(readlines(ref_counter_file)))
                # Just to be safe, let's also make sure that we don't delete
                # any paths with recent (let's say 5) ref counter increments ago.
                if ref_counter + keep_latest_ref_counters < ref_counter_main
                    push!(paths_to_delete, path)
                end
            end
        end
        @show ref_counter_main
        @show length(sorted_paths)
        @show length(paths_to_delete)
        @info "Deleting files:"
        for i in 1:length(paths_to_delete)
            f = paths_to_delete[i]
            @info "     (File, date): ($(f), $(Dates.unix2datetime(stat(f).mtime))). Reason: $(reason(f))"
        end
        for i in 1:length(paths_to_delete)
            rm(paths_to_delete[i]; recursive = true, force = true)
        end
    end
end

if buildkite_ci && in_merge_queue
    cleanup_central(cluster_data_prefix)
end
