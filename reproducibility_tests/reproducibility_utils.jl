#=
################################################################################
Reproducibility Terminology.

Consider the following set of reproducibility
folders, prefixed by "reference counters", which
allow users to compare against other reproducible
states in that column.

Note that reference counter changes can "rewind"
(which may happen in the case of reverted commits).
In such cases, we do consider the rewound state as
an entirely new state, in order to fully preserve
the history (to some depth).

An important consequence of this requires precise
terminology to avoid ambiguous descriptions.

For example, "comparable references per reference counter"
is not well defined, because the reference counter can
be reverted. So, let's introduce the concept of a "bin",
which can be defined as a collection of folders
created in a period with the same reference counter.
Folders created before and after that bin have a different
reference counter. Also, `n_bins == n_reference_changes + 1`
(modulo the edge case of when there are no bins)
because, if the reference counter doesn't change, new results
are put into the same bin.
```
comparable states
         |                           ref counter changes ---->                         | oldest
         |                                                                             |
         |  bin 1      bin 2      bin 3      bin 4      bin 5      bin 6      bin 7    |
         |                                                                             |
         |  02_49f92   04_36ebe   05_beb8a   06_4d837   05_8c311   08_45875   10_bc1e0 |
         |             04_d6e48              06_d6d73              08_1cc58            |
         v             04_4c042                                                        v newest
```
################################################################################
=#

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

"""
    invalid_reference_folders(; root_path)

Return all subfolders in `root_path`
that meet the following criteria:

 - A `ref_counter.jl` file is missing
"""
function invalid_reference_folders(; root_path)
    paths = sorted_dataset_folder(; dir = root_path)
    invalid_folders = filter(paths) do p
        !isfile(joinpath(p, "ref_counter.jl"))
    end
    return invalid_folders
end

"""
    compute_bins(root_path::String)
    compute_bins(sorted_paths::Vector{String})

Return a vector of reproducibility bins.

Bins are sorted from newest to oldest:
 - `bins[1], bins[end]` are the newest and oldest bins
 - `bins[i][1], bins[i][end]` are the newest oldest comparable states.

```
comparable states
         |                           ref counter changes ---->                         | oldest
         |                                                                             |
         |  bin 1      bin 2      bin 3      bin 4      bin 5      bin 6      bin 7    |
         |                                                                             |
         |  02_49f92   04_36ebe   05_beb8a   06_4d837   05_8c311   08_45875   10_bc1e0 |
         |             04_d6e48              06_d6d73              08_1cc58            |
         v             04_4c042                                                        v newest
```
"""
compute_bins(root_path::String) =
    compute_bins(reverse(sorted_dataset_folder(; dir = root_path)))
function compute_bins(sorted_paths::Vector{String})
    bins = Vector{String}[]
    path_index = 1
    while path_index ≤ length(sorted_paths)
        paths_per_bin = String[]
        while path_index ≤ length(sorted_paths)
            path = sorted_paths[path_index]
            if isempty(paths_per_bin)
                push!(paths_per_bin, path)
                path_index += 1
            else
                ref_counter_bin = read_ref_counter(
                    joinpath(first(paths_per_bin), "ref_counter.jl"),
                )
                ref_counter_path =
                    read_ref_counter(joinpath(path, "ref_counter.jl"))
                if ref_counter_bin == ref_counter_path
                    push!(paths_per_bin, path)
                    path_index += 1
                else
                    break
                end
            end
        end
        push!(bins, paths_per_bin)
    end
    return bins
end

"""
    get_reference_paths_to_delete(;
        root_path,
        keep_n_comparable_states = 5,
        keep_n_bins_back = 7,
    )

Return a list of folders to delete.

Our reference folders are saved, and can
therefore build up significantly and take
a lot of storage.

Consider a collection of folders whose
names are prepended by the reference
counter:

```
keep_n_comparable_states
         |                           <---- keep_n_bins_back                            | oldest
         |                                                                             |
         |  bin 1      bin 2      bin 3      bin 4      bin 5      bin 6      bin 7    |
         |                                                                             |
         |  02_49f92   04_36ebe   05_beb8a   06_4d837   05_8c311   08_45875   10_bc1e0 |
         |             04_d6e48              06_d6d73              08_1cc58            |
         v             04_4c042                                                        v newest
```
With these folders, and given a reference
counter of 10, we'll see the following
behavior:
```
    get_reference_paths_to_delete(;
        keep_n_comparable_states = 4,
        keep_n_bins_back = 3
    ) -> [02_49f92, 04_36ebe, 04_d6e48, 04_4c042]

    get_reference_paths_to_delete(;
        keep_n_comparable_states = 1,
        keep_n_bins_back = 5
    ) -> [02_49f92, 04_d6e48, 04_4c042, 06_d6d73, 08_1cc58]
```

Note:
    `keep_n_references_back` is sorted _chronologically_,
    in order to correctly operate in the case of
    reverted pull requests. In other words, the above
    references may look like this:
```
keep_n_comparable_states
         |                           <---- keep_n_bins_back                            | oldest
         |                                                                             |
         |  bin 1      bin 2      bin 3      bin 4      bin 5      bin 6      bin 7    |
         |                                                                             |
         |  02_49f92   04_36ebe   05_beb8a   06_4d837   05_8c311   08_45875   10_bc1e0 |
         |             04_d6e48              06_d6d73              08_1cc58            |
         v             04_4c042                                                        v newest
```

"""
function get_reference_paths_to_delete(;
    root_path,
    keep_n_comparable_states = 5,
    keep_n_bins_back = 7,
)
    @assert isempty(invalid_reference_folders(; root_path))
    paths_to_delete = String[]
    sorted_paths = reverse(sorted_dataset_folder(; dir = root_path))
    if !isempty(sorted_paths)
        # Now, sorted_paths[1] is newest, sorted_paths[end] is oldest
        bins = compute_bins(sorted_paths)
        for i in 1:length(bins), j in 1:length(bins[i])
            if i ≤ keep_n_bins_back
                if !(j ≤ keep_n_comparable_states)
                    push!(paths_to_delete, bins[i][j])
                end
            else
                push!(paths_to_delete, bins[i][j])
            end
        end
    end
    return paths_to_delete
end
