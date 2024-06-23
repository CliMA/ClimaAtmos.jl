
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
if get(ENV, "BUILDKITE_PIPELINE_SLUG", nothing) == "climaatmos-ci"
    commit = ENV["BUILDKITE_COMMIT"]
    branch = ENV["BUILDKITE_BRANCH"]

    @info "pwd() = $(pwd())"
    @info "branch = $(branch)"
    @info "commit = $(commit)"

    using Glob
    @show readdir(joinpath(@__DIR__, ".."))
    if startswith(branch, "gh-readonly-queue/main/")
        commit_sha = commit[1:7]
        mkpath(cluster_data_prefix)
        path = joinpath(cluster_data_prefix, commit_sha)
        mkpath(path)
        # Only move regression data if self reference:
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
        @info "readdir(): $(readdir(path))"
    end
else
    @info "ENV keys: $(keys(ENV))"
end

function get_ref_counter(p)
    f = joinpath(p, "ref_counter.jl")
    if isfile(f)
        @info "Ref counter found: $f"
        return parse(Int, first(readlines(f)))
    else
        msg = "Expected ref counter file did not exist in path $p\n. Found (recursive) `ref_counter.jl` files:\n"
        for (root, _, files) in walkdir(p)
            for f in files
                endswith(f, "ref_counter.jl") || continue
                msg *= "    $(joinpath(root, f))"
            end
        end
        @warn msg
        return nothing
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
        debug = false
        ref_counter_last = get_ref_counter(sorted_paths[end])
        @show isnothing(ref_counter_last), ref_counter_last
        for i in 1:N
            ref_counter = get_ref_counter(sorted_paths[i])
            if isnothing(ref_counter_last) || isnothing(ref_counter)
                debug = true
            end
            # Just to be safe, let's also make sure that we don't delete
            # any paths with recent (let's say 5) ref counter increments ago.
            if !isnothing(ref_counter) && !isnothing(ref_counter_last)
                if ref_counter + keep_latest_ref_counters < ref_counter_last
                    push!(paths_to_delete, sorted_paths[i])
                end
            end
        end
        @show length(sorted_paths)
        @show length(paths_to_delete)
        if debug
            @show sorted_paths
            error("Reference counters did not exist")
        end
        @info "Deleting files:"
        for i in 1:N
            f = paths_to_delete[i]
            @info "     (File, date): ($(f), $(Dates.unix2datetime(stat(f).mtime)))"
        end
        # for i in 1:N
        #     rm(paths_to_delete[i])
        # end
    end
end

cleanup_central(cluster_data_prefix)
