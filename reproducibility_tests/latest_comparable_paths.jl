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
    isempty(matching_paths) && return String[]
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
    paths = latest_comparable_paths(;
        n = 5,
        root_path = "/central/scratch/esm/slurm-buildkite/climaatmos-main",
        ref_counter_PR = read_ref_counter(joinpath(@__DIR__, "ref_counter.jl"))
    )

Returns a vector of strings, containing the `n`
latest comparable paths. The assumed folder structure
is:

```
root_path/some_folder_1/ref_counter.jl
root_path/some_folder_2/ref_counter.jl
root_path/some_folder_3/ref_counter.jl
```

If a subfolder does not contain a `ref_counter.jl` file
then it is filtered out as not-comparable. The `ref_counter.jl`
files are assumed to start with a single integer,
which is read. If that integer matches `ref_counter_PR`,
then that path is considered comparable.

`paths[1]` is the most recent comparable path, and
`paths[end]` is the oldest comparable path.
"""
function latest_comparable_paths(;
    n = 5,
    root_path = "/central/scratch/esm/slurm-buildkite/climaatmos-main",
    ref_counter_PR = read_ref_counter(joinpath(@__DIR__, "ref_counter.jl")),
)
    @info "---Finding the latest comparable paths"
    # Note: root_path is also defined in move_output.jl
    # Get (sorted) array of paths, `pop!(sorted_paths)`
    # is the most recent merged folder.
    sorted_paths = sorted_dataset_folder(; dir = root_path)
    if isempty(sorted_paths)
        @warn "No paths found in $root_path"
        return String[]
    end
    # Find oldest path in main with the same reference
    # counter as the one in the PR. If none exists,
    # then assume no comparable references.

    # Short circuit if we don't find anything:
    found_ref_counters =
        filter(p -> isfile(joinpath(p, "ref_counter.jl")), sorted_paths)
    if isempty(found_ref_counters)
        @warn "No reference counters found in paths: $sorted_paths"
        return String[]
    end

    # Find comparable paths
    comparable_paths = String[]
    @info "Reference counters found:"
    for (i, path) in enumerate(sorted_paths)
        ref_counter_file = joinpath(path, "ref_counter.jl")
        !isfile(ref_counter_file) && continue
        rc = read_ref_counter(ref_counter_file)
        comparable = ref_counter_PR == rc
        suffix = comparable ? ", comparable" : ""
        @info "     $path: $rc$suffix"
        comparable && push!(comparable_paths, path)
    end

    if isempty(comparable_paths)
        @warn "No comparable paths found in any of the paths:$sorted_paths"
        return String[]
    end

    comparable_paths = reverse(comparable_paths) # sort so that

    if length(comparable_paths) > n # limit to n comparable paths
        comparable_paths = comparable_paths[1:min(n, length(comparable_paths))]
    end

    return comparable_paths
end
