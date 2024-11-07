import Dates

read_ref_counter(filename) = parse(Int, first(readlines(filename)))

"""
    sorted_dataset_folder(; dir=pwd())

Return a the subdirectory paths within the given `dir` (defaults
to the current working directory) sorted by modification time
(oldest to newest).  Return an empty vector if no subdirectories
are found.
"""
function sorted_dataset_folder(; dir = pwd())
    matching_paths = filter(ispath, readdir(dir; join = true))
    isempty(matching_paths) && return ""
    # sort by timestamp
    sorted_paths =
        sort(matching_paths; by = f -> Dates.unix2datetime(stat(f).mtime))
    return sorted_paths
end


"""
    ref_counters_per_path(paths)

Read the `ref_counter.jl` file in each given path and parses the integer it contains.
Return a vector of integers, where each element corresponds to a path.
If a path does not contain a `ref_counter.jl` file, the corresponding element is -1.
It assumes that `ref_counter.jl` contains the value as the first line of the file.
"""
function ref_counters_per_path(paths)
    ref_counters_in_path = Vector{Int}(undef, length(paths))
    ref_counters_in_path .= -1
    for (i, path) in enumerate(paths)
        ref_counter_file = joinpath(path, "ref_counter.jl")
        !isfile(ref_counter_file) && continue
        ref_counters_in_path[i] = read_ref_counter(ref_counter_file)
    end
    return ref_counters_in_path
end

"""
    latest_comparable_paths(n::Integer)

Returns a vector of strings, containing the `n`
latest comparable paths based on
`reproducibility_tests/ref_counter.jl`.
"""
function latest_comparable_paths(n = 5)
    if get(ENV, "BUILDKITE_PIPELINE_SLUG", nothing) != "climaatmos-ci"
        @warn "Not using climaatmos-ci pipeline slug, assuming no comparable references"
        @info "Please review output results before merging."
        return String[]
    end

    # Note: cluster_data_prefix is also defined in move_output.jl
    cluster_data_prefix = "/central/scratch/esm/slurm-buildkite/climaatmos-main"
    # Get (sorted) array of paths, `pop!(sorted_paths)`
    # is the most recent merged folder.
    sorted_paths = sorted_dataset_folder(; dir = cluster_data_prefix)
    if isempty(sorted_paths)
        @warn "No paths on main found, assuming no comparable references"
        @info "Please review output results before merging."
        return String[]
    end
    # Find oldest path in main with the same reference
    # counter as the one in the PR. If none exists,
    # then assume no comparable references.

    ref_counter_file_PR = joinpath(@__DIR__, "ref_counter.jl")
    @assert isfile(ref_counter_file_PR)
    ref_counter_PR = read_ref_counter(ref_counter_file_PR)

    ref_counters_main = ref_counters_per_path(sorted_paths)
    i_comparable_references = findall(ref_counters_main) do ref_counter_main
        ref_counter_main == ref_counter_PR
    end
    if isnothing(i_comparable_references)
        @warn "`ref_counter.jl` not found on main, assuming no comparable references"
        @info "Please review output results before merging."
        return String[]
    end
    @info "Found $(length(i_comparable_references)) comparable references:$i_comparable_references"
    # Largest ref-counter reference path:
    paths = map(i -> sorted_paths[i], i_comparable_references)
    @info "$(length(paths)) paths found:"
    for p in paths
        @info "     $p, $(Dates.unix2datetime(stat(p).mtime))"
    end
    ref_counter_files_main = map(p -> joinpath(p, "ref_counter.jl"), paths)
    @info "$(length(ref_counter_files_main)) reference counter paths on central"
    filter!(isfile, ref_counter_files_main)
    @info "$(length(ref_counter_files_main)) reference counter paths on central after filtering isfile"

    # for p in paths
    #     @info "Files in $p:" # for debugging
    #     for file_on_main in readdir(p)
    #         @info "   File:`$file_on_main`"
    #     end
    # end
    @assert all(isfile, ref_counter_files_main)
    ref_counters_main = map(read_ref_counter, ref_counter_files_main)
    if all(rc -> ref_counter_PR == rc + 1, ref_counters_main) # new reference
        @warn "`ref_counter.jl` incremented, assuming no comparable references"
        @info "Ref counters main: $ref_counters_main."
        @info "Please review output results before merging."
        return String[]
    elseif all(rc -> ref_counter_PR == rc, ref_counters_main) # unchanged reference
        @info "Ref counters main: $ref_counters_main."
        @info "Comparing results against main path:$paths"
    else
        error(
            "Unexpected reference. Please open an issue pointing to this build.",
        )
    end

    paths = reverse(paths)[1:min(n, length(paths))]
    @info "Limiting comparable paths to $n:"
    for p in paths
        @info "     $p, $(Dates.unix2datetime(stat(p).mtime))"
    end
    # Get the top 10 most recent paths to compare against:
    return paths
end
