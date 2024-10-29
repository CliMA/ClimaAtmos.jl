import Dates

function sorted_dataset_folder(; dir = pwd())
    matching_paths = String[]
    for file in readdir(dir)
        !ispath(joinpath(dir, file)) && continue
        push!(matching_paths, joinpath(dir, file))
    end
    isempty(matching_paths) && return ""
    # sort by timestamp
    sorted_paths =
        sort(matching_paths; by = f -> Dates.unix2datetime(stat(f).mtime))
    return sorted_paths
end

function ref_counters_per_path(paths)
    ref_counters_in_path = Vector{Int}(undef, length(paths))
    ref_counters_in_path .= -1
    for (i, path) in enumerate(paths)
        ref_counter_file = joinpath(path, "ref_counter.jl")
        !isfile(ref_counter_file) && continue
        ref_counters_in_path[i] = parse(Int, first(readlines(ref_counter_file)))
    end
    return ref_counters_in_path
end

function self_reference_or_path()
    if get(ENV, "BUILDKITE_PIPELINE_SLUG", nothing) != "climaatmos-ci"
        @warn "Not using climaatmos-ci pipeline slug, assuming self-reference"
        @info "Please review output results before merging."
        return :self_reference
    end

    # Note: cluster_data_prefix is also defined in move_output.jl
    cluster_data_prefix = "/central/scratch/esm/slurm-buildkite/climaatmos-main"
    # Get (sorted) array of paths, `pop!(sorted_paths)`
    # is the most recent merged folder.
    sorted_paths = sorted_dataset_folder(; dir = cluster_data_prefix)
    if isempty(sorted_paths)
        @warn "No paths on main found, assuming self-reference"
        @info "Please review output results before merging."
        return :self_reference
    end
    # Find oldest path in main with the same reference
    # counter as the one in the PR. If none exists,
    # then assume self reference.

    ref_counter_file_PR = joinpath(@__DIR__, "ref_counter.jl")
    @assert isfile(ref_counter_file_PR)
    ref_counter_PR = parse(Int, first(readlines(ref_counter_file_PR)))

    ref_counters_main = ref_counters_per_path(sorted_paths)
    i_largest_reference = findfirst(ref_counters_main) do ref_counter_main
        ref_counter_main == ref_counter_PR
    end
    if i_largest_reference == nothing
        @warn "`ref_counter.jl` not found on main, assuming self-reference"
        @info "Please review output results before merging."
        return :self_reference
    end
    # Largest ref-counter reference path:
    path = sorted_paths[i_largest_reference]
    ref_counter_file_main = joinpath(path, "ref_counter.jl")

    @info "Files in $path:" # for debugging
    for file_on_main in readdir(path)
        @info "   File:`$file_on_main`"
    end
    @assert isfile(ref_counter_file_main)
    ref_counter_main = parse(Int, first(readlines(ref_counter_file_main)))
    if ref_counter_PR == ref_counter_main + 1 # new reference
        @warn "`ref_counter.jl` incremented, assuming self-reference"
        @info "Please review output results before merging."
        return :self_reference
    elseif ref_counter_PR == ref_counter_main # unchanged reference
        @info "Comparing results against main path:$path"
    else
        error(
            "Unexpected reference. Please open an issue pointing to this build.",
        )
    end
    return path
end
