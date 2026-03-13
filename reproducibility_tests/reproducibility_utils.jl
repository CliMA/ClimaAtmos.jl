#=
################################################################################
Reproducibility Terminology.

First, we try to be consistent in distinguishing:

 - `dir` to denote a directory (and not a file)
 - `path` to denote files or directories
 - `folder_name` to denote a folder name, for example, "`f`" in `a/f/c`
 - `file` to denote a file (and not a directory)

Consider the following set of reproducibility directories, prefixed
by "reference counters", which allow users to compare against other
reproducible states in that column.

Note that the reference counter must increment in the case of reverted commits,
resulting in an entirely new state, in order to fully preserve the history
(to some depth).

Here, we introduce the concept of a "bin", which can be defined as a collection
of directories created in a period with the same reference counter. Folders
created before and after that bin have a different reference counter. Also,
`n_bins == n_reference_changes + 1`(modulo the edge case of when there are no
bins) because, if the reference counter doesn't change, new results are put
into the same bin.

```
comparable states
         |                           ref counter changes ---->                         | oldest
         |                                                                             |
         |  bin 1      bin 2      bin 3      bin 4      bin 5      bin 6      bin 7    |
         |                                                                             |
         |  02_49f92   03_36ebe   04_beb8a   05_4d837   06_8c311   07_45875   08_bc1e0 |
         |             03_d6e48              05_d6d73              07_1cc58            |
         v             03_4c042                                                        v newest
```

# File states

Reproducibility tests inherently rely on comparing multiple states, which means
that our reproducibility testing infrastructure is _stateful_. During our
continuous integration testing (CI), files are generated, moved, and zipped. To help
assist our understanding and reasoning, we let's assume that there are two states:

## state 1: end of simulation, folder structure

 - `job_id/output_dir/`
 - `job_id/output_dir/reproducibility_bundle/`
 - `job_id/output_dir/reproducibility_bundle/ref_counter.jl`
 - `job_id/output_dir/reproducibility_bundle/prog_state.hdf5`

## state 2: data is saved for future reference

 - `commit_hash/job_id/reproducibility_bundle/`
 - `commit_hash/job_id/reproducibility_bundle/ref_counter.jl`
 - `commit_hash/job_id/reproducibility_bundle/prog_state.hdf5`

 - `commit_hash/reproducibility_bundle/ref_counter.jl`
 - `commit_hash/reproducibility_bundle/job_id/`
 - `commit_hash/reproducibility_bundle/job_id/prog_state.hdf5`

In other words, we strip out `output_dir/`, and swap `job_id` and
`reproducibility_bundle`. This is done for two reasons:

 - The ref_counter is job-independent, hence the swap
 - The `output_dir/` is redundant to the purpose of the commit hash folder

################################################################################
=#

# ────────────────────────────────────────────────────────────────────
# Configuration: enable debug output only on the CI pipeline
# ────────────────────────────────────────────────────────────────────

"""Return `true` when running on the ClimaAtmos CI pipeline."""
debug_reproducibility() =
    get(ENV, "BUILDKITE_PIPELINE_SLUG", nothing) == "climaatmos-ci"

import Dates
import OrderedCollections

"""Return a string listing all files (recursively) in `dir`."""
function string_all_files_in_dir(dir)
    msg = "Files in dir $dir\n"
    for file in all_files_in_dir(dir)
        msg *= "     $file\n"
    end
    return msg
end

"""Return a flat vector of all file paths (recursively) under `dir`."""
function all_files_in_dir(dir)
    all_files = String[]
    for (root, dirs, files) in walkdir(dir)
        for file in files
            f = joinpath(root, file)
            isfile(f) || continue # avoid symlinks
            push!(all_files, f)
        end
    end
    return all_files
end

# ────────────────────────────────────────────────────────────────────
# Reference counter: monotonically increasing version identifier
# ────────────────────────────────────────────────────────────────────

"""Read the reference counter (integer on the first line) from `file`."""
read_ref_counter(file) = parse(Int, first(readlines(file)))

"""
    sorted_dirs_with_matched_files(; dir = pwd())

Return an array of subdirectories of `dir` (defaults to the current working
directory) sorted by the reference counters contained in the folders. Return an
empty vector if no subdirectories are found.

This function recurses through `dir`, and finds all directories that have the
file `ref_counter.jl`.
"""
function sorted_dirs_with_matched_files(; dir = pwd())
    filename = "ref_counter.jl"
    matched_dirs = String[]
    for (root, dirs, files) in walkdir(dir)
        for dir in dirs
            push!(matched_dirs, joinpath(root, dir))
        end
    end
    isempty(matched_dirs) && return String[]
    filter!(x -> isfile(joinpath(x, filename)), matched_dirs)
    isempty(matched_dirs) && return String[]
    # sort by timestamp
    sorted_dirs =
        sort(matched_dirs; by = f -> read_ref_counter(joinpath(f, filename)))
    return sorted_dirs
end

"""
    ref_counters_per_dir(dirs)

Read the `ref_counter.jl` file in each given directory and parses the integer it
contains. Return a vector of integers, where each element corresponds to a
directory. If a directory does not contain a `ref_counter.jl` file, the
corresponding element is -1. It assumes that `ref_counter.jl` contains the
value as the first line of the file.
"""
function ref_counters_per_dir(dirs)
    ref_counters_in_dir = Vector{Int}(undef, length(dirs))
    ref_counters_in_dir .= -1
    for (i, dir) in enumerate(dirs)
        ref_counter_file = joinpath(dir, "ref_counter.jl")
        !isfile(ref_counter_file) && continue
        ref_counters_in_dir[i] = read_ref_counter(ref_counter_file)
    end
    return ref_counters_in_dir
end

"""
    dirs = latest_comparable_dirs(;
        n = 5,
        root_dir = "/resnick/scratch/esm/slurm-buildkite/climaatmos-main",
        ref_counter_PR = read_ref_counter(joinpath(@__DIR__, "ref_counter.jl"))
        skip = get(ENV, "BUILDKITE_PIPELINE_SLUG", nothing) != "climaatmos-ci"
    )

Returns a vector of strings, containing the `n` latest comparable directories in
the `root_dir` directory. Only directories that match the `ref_counter_PR` are
returned, and an empty vector is retuned if `skip = true`. By default, `skip`
is set to `get(ENV, "BUILDKITE_PIPELINE_SLUG", nothing) != "climaatmos-ci"`.

The assumed folder structure is:

```
root_dir/some_folder_1/ref_counter.jl
root_dir/some_folder_2/ref_counter.jl
root_dir/some_folder_3/ref_counter.jl
```

If a subfolder does not contain a `ref_counter.jl` file then it is filtered out
as not-comparable. The `ref_counter.jl` files are assumed to start with a
single integer, which is read. If that integer matches `ref_counter_PR`, then
that directory is considered comparable.

`dirs[1]` is the most recent comparable directory, and `dirs[end]` is the oldest
comparable directory.
"""
function latest_comparable_dirs(;
    n = 5,
    root_dir = "/resnick/scratch/esm/slurm-buildkite/climaatmos-main",
    ref_counter_PR = read_ref_counter(joinpath(@__DIR__, "ref_counter.jl")),
    skip = get(ENV, "BUILDKITE_PIPELINE_SLUG", nothing) != "climaatmos-ci",
)
    skip && return String[]
    bins = compute_bins(root_dir)
    isempty(bins) && return String[]
    ref_counter_bins = filter(bins) do bin
        f = joinpath(first(bin), "ref_counter.jl")
        isfile(f) && ref_counter_PR == read_ref_counter(f)
    end
    isempty(ref_counter_bins) && return String[]
    comparable_dirs = ref_counter_bins[1]
    return comparable_dirs[1:min(n, length(comparable_dirs))]
end

"""
    invalid_reference_folders(dirs)

Return all subfolders in vectory of directory, `dirs`, that meet the following
criteria:

 - A `ref_counter.jl` file is missing
"""
function invalid_reference_folders(dirs)
    invalid_folders = filter(dirs) do p
        !isfile(joinpath(p, "ref_counter.jl"))
    end
    return invalid_folders
end

"""
    compute_bins(root_dir::String)
    compute_bins(sorted_dirs::Vector{String})

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
function compute_bins(
    root_dir::String = "/resnick/scratch/esm/slurm-buildkite/climaatmos-main",
)
    dirs = sorted_dirs_with_matched_files(; dir = root_dir)
    return compute_bins(reverse(dirs))
end

function compute_bins(sorted_dirs::Vector{String})
    @assert isempty(invalid_reference_folders(sorted_dirs))
    bins = Vector{String}[]
    dir_index = 1
    while dir_index ≤ length(sorted_dirs)
        dirs_per_bin = String[]
        while dir_index ≤ length(sorted_dirs)
            dir = sorted_dirs[dir_index]
            if isempty(dirs_per_bin)
                push!(dirs_per_bin, dir)
                dir_index += 1
            else
                ref_counter_bin = read_ref_counter(
                    joinpath(first(dirs_per_bin), "ref_counter.jl"),
                )
                ref_counter_dir =
                    read_ref_counter(joinpath(dir, "ref_counter.jl"))
                if ref_counter_bin == ref_counter_dir
                    push!(dirs_per_bin, dir)
                    dir_index += 1
                else
                    break
                end
            end
        end
        push!(bins, dirs_per_bin)
    end
    return bins
end

print_bins(bins) = print_bins(stdout, bins)
print_bins(io::IO, bins) = println(io, string_bins(bins))

"""
    string_bins(bins)

Return a string summarizing the given bins.
"""
function string_bins(bins)
    io = IOBuffer()
    println(io, "Bins:")
    for (i, bin) in enumerate(bins)
        println(io, "  Bin $i:")
        for (j, state) in enumerate(bin)
            ref_counter = read_ref_counter(joinpath(state, "ref_counter.jl"))
            println(io, "    (State $j, ref_counter): ($state, $ref_counter)")
        end
    end
    return String(take!(io))
end

"""
    get_reference_dirs_to_delete(;
        root_dir,
        keep_n_comparable_states = 100,
        keep_n_bins_back = 100,
    )

Return a list of folders to delete.

Our reference folders are saved, and can therefore build up significantly and
take a lot of storage.

Consider a collection of folders whose names are prepended by the reference
counter:

```
keep_n_comparable_states
         |                           <---- keep_n_bins_back                            | oldest
         |                                                                             |
         |  bin 1      bin 2      bin 3      bin 4      bin 5      bin 6      bin 7    |
         |                                                                             |
         |  02_49f92   03_36ebe   04_beb8a   05_4d837   06_8c311   07_45875   08_bc1e0 |
         |             03_d6e48              05_d6d73              07_1cc58            |
         v             03_4c042                                                        v newest
```

With these folders, and given a reference counter of 10, we'll see the following
behavior:

```
    get_reference_dirs_to_delete(;
        keep_n_comparable_states = 4,
        keep_n_bins_back = 3
    ) -> [02_49f92, 03_36ebe, 03_d6e48, 03_4c042]

    get_reference_dirs_to_delete(;
        keep_n_comparable_states = 1,
        keep_n_bins_back = 5
    ) -> [02_49f92, 03_d6e48, 03_4c042, 05_d6d73, 07_1cc58]
```
"""
function get_reference_dirs_to_delete(;
    root_dir,
    keep_n_comparable_states = 100,
    keep_n_bins_back = 100,
)
    dirs = sorted_dirs_with_matched_files(; dir = root_dir)
    @assert isempty(invalid_reference_folders(dirs))
    dir_to_delete = String[]
    sorted_dirs = reverse(dirs)
    if !isempty(sorted_dirs)
        # Now, sorted_dirs[1] is newest, sorted_dirs[end] is oldest
        bins = compute_bins(sorted_dirs)
        for i in 1:length(bins), j in 1:length(bins[i])
            if i ≤ keep_n_bins_back
                if !(j ≤ keep_n_comparable_states)
                    push!(dir_to_delete, bins[i][j])
                end
            else
                push!(dir_to_delete, bins[i][j])
            end
        end
    end
    return dir_to_delete
end

"""
    strip_output_active_folder(folder)

Return `""` if `folder` matches `"output_active"` or `"output_XXXX"` (where
X are digits). Otherwise return `folder` unchanged. Used when remapping
paths during data movement to strip CI output directory prefixes.
"""
function strip_output_active_folder(folder)
    (folder == "output_active" || occursin(r"^output_\d{4}$", folder)) ?
    "" : folder
end

"""
    strip_output_active_path(path)

Applies `strip_output_active_folder` to all folders in the given path.
"""
strip_output_active_path(path) =
    joinpath(map(x -> strip_output_active_folder(x), splitpath(path))...)

print_dir_tree(dir) = print_dir_tree(stdout, dir)
print_dir_tree(io::IO, dir) = println(io, string_dir_tree(dir))

function string_dir_tree(dir)
    s = "Files in `$dir`:\n"
    for (root, _, files) in walkdir(dir)
        for file in files
            f = joinpath(root, file)
            isfile(f) || continue # rm symlink folders (included but not files)
            s *= "  $(replace(f, dir => ""))\n"
        end
    end
    return s
end

"""
    move_data_to_save_dir(;
        buildkite_ci = get(ENV, "BUILDKITE_PIPELINE_SLUG", nothing) ==
                       "climaatmos-ci",
        branch = get(ENV, "BUILDKITE_BRANCH", nothing),
        in_merge_queue = startswith(branch, "gh-readonly-queue/main/"),
        dirs_src,
        ref_counter_file_PR = joinpath(@__DIR__, "ref_counter.jl"),
        ref_counter_PR = read_ref_counter(ref_counter_file_PR),
        skip = get(ENV, "BUILDKITE_PIPEL Data movement will occur when this function is called:

 - on a job run in buildkite
 - when in the merge queue
 - when on the main branch
"""
function move_data_to_save_dir(;
    buildkite_ci = get(ENV, "BUILDKITE_PIPELINE_SLUG", nothing) ==
                   "climaatmos-ci",
    branch = get(ENV, "BUILDKITE_BRANCH", nothing),
    in_merge_queue = startswith(branch, "gh-readonly-queue/main/"),
    dirs_src,
    ref_counter_file_PR = joinpath(@__DIR__, "ref_counter.jl"),
    ref_counter_PR = read_ref_counter(ref_counter_file_PR),
    dest_root = "/resnick/scratch/esm/slurm-buildkite/climaatmos-main",
    commit = get_commit_sha(),
    repro_folder = "reproducibility_bundle",
    strip_folder = strip_output_active_path,
)
    buildkite_ci || return nothing

    # if a contributor manually merged, we still want to move data from scratch
    # to `dest_root`.

    @assert isfile(ref_counter_file_PR)
    if in_merge_queue || branch == "main"
        (; files_src, files_dest) = save_dir_in_out_list(;
            dirs_src,
            dest_root,
            commit,
            repro_folder,
            strip_folder,
        )
        if debug_reproducibility()
            println("******")
            foreach(print_dir_tree, dirs_src)
            println("******")
        end
        for (src, dest) in zip(files_src, files_dest)
            @assert isfile(src)
            mkpath(dirname(dest))
            mv(src, dest; force = true)
        end
        dest_repro = destination_directory(; dest_root, commit, repro_folder)
        ref_counter_file_main = joinpath(dest_repro, "ref_counter.jl")
        debug_reproducibility() &&
            @info "Repro: moving $ref_counter_file_PR to $ref_counter_file_main"
        mv(ref_counter_file_PR, ref_counter_file_main; force = true)
    else
        if debug_reproducibility()
            @warn "Repro: skipping data movement"
            @show in_merge_queue
            @show branch == "main"
        end
    end
end

"""
    get_commit_sha(;
        n_hash_characters = 7,
        commit = get(ENV, "BUILDKITE_COMMIT", "UNKNOWN")
    )

Returns a string of the commit hash.
"""
get_commit_sha(;
    n_hash_characters = 7,
    commit = get(ENV, "BUILDKITE_COMMIT", "UNKNOWN"),
) = return commit[1:min(n_hash_characters, length(commit))]

"""
    commit_sha_from_dir(commit_shas, dir)

Walk up the path components of `dir` until one matches an entry in
`commit_shas`. Returns the matching component, or errors if none is found.
"""
function commit_sha_from_dir(commit_shas, dir)
    while true
        if isempty(dir)
            error("Unfound commit sha.")
        else
            b = basename(dir)
            if b in commit_shas || any(x -> occursin(b, x), commit_shas)
                return b
            else
                dir = dirname(dir)
            end
        end
    end
end

"""
    save_dir_transform(
        src;
        job_id,
        dest_root = "/resnick/scratch/esm/slurm-buildkite/climaatmos-main",
        commit = get_commit_sha(),
        repro_folder = "reproducibility_bundle",
        strip_folder = strip_output_active_path,
    )

Returns the output file, to be saved, given:
 - `src` the source file
 - `job_id` the job ID
 - `dest_root` the destination root directory
 - `commit` the commit hash
 - `repro_folder` reproducibility folder
 - `strip_folder` function to strip folders in output path
"""
function save_dir_transform(
    src;
    job_id,
    dest_root = "/resnick/scratch/esm/slurm-buildkite/climaatmos-main",
    commit = get_commit_sha(),
    repro_folder = "reproducibility_bundle",
    strip_folder = strip_output_active_path,
)
    dest_repro = destination_directory(; dest_root, commit, repro_folder)
    src_filename = basename(src)
    dst = joinpath(dest_repro, job_id, src_filename)
    return strip_output_active_path(dst)
end

"""
    destination_directory(;
        dest_root = "/resnick/scratch/esm/slurm-buildkite/climaatmos-main",
        commit = get_commit_sha(),
        repro_folder = "reproducibility_bundle",
    )

Return the reproducibility destination directory:
`root/commit_sha/repro_folder`, given:
 - `dest_root` the destination root directory
 - `commit` the commit hash
 - `repro_folder` reproducibility folder
"""
function destination_directory(;
    dest_root = "/resnick/scratch/esm/slurm-buildkite/climaatmos-main",
    commit = get_commit_sha(),
    repro_folder = "reproducibility_bundle",
)
    return joinpath(dest_root, commit, repro_folder)
end

"""
    save_dir_in_out_list

Returns two vectors of strings, containing input and output files, for moving
data from the computed to saved directories.

```julia
(; files_src, files_dest) = save_dir_in_out_list(; dirs_src)
for (src, dest) in zip(files_src, files_dest)
    mv(src, dest; force = true)
end
```
"""
function save_dir_in_out_list(; dirs_src, kwargs...)
    files_dest = String[]
    files_src = String[]
    for src_dir in dirs_src
        job_id = basename(src_dir)
        for src in all_files_in_dir(src_dir)
            push!(files_src, src)
            push!(files_dest, save_dir_transform(src; job_id, kwargs...))
        end
    end
    return (; files_src, files_dest)
end

"""Parse a Julia literal expression from `file` and evaluate it."""
parse_file(file) = eval(Meta.parse(read(file, String)))

# ────────────────────────────────────────────────────────────────────
# RMS summary collection and printing
# ────────────────────────────────────────────────────────────────────

"""
    default_is_rms_file(file, expected_filename_prefix::String = "computed_rms")::Bool

Returns `true` if the given `file` is a computed RMS comparison file,
based on filename prefix and `.dat` extension.
"""
function default_is_rms_file(
    file,
    expected_filename_prefix::String = "computed_rms",
)
    bn = basename(file)
    return startswith(bn, expected_filename_prefix) && endswith(bn, ".dat")
end

"""
    get_computed_rms(;
        job_ids::Vector{String},
        subfolder::String = "output_active/reproducibility_bundle",
        is_rms_file::Function = default_is_rms_file,
    )

Returns a Dict mapping each job ID to either the parsed comparison result
dict (if a computed RMS file was found) or `false` (if not found).

It is expected that files exist in the form:
    `joinpath(job_ids[1], subfolder, rms_filename)`
where `is_rms_file(rms_filename)` is `true`.
"""
function get_computed_rms(;
    job_ids::Vector{String},
    subfolder::String = joinpath("output_active", "reproducibility_bundle"),
    is_rms_file::Function = default_is_rms_file,
    expected_filename_prefix = "computed_rms",
)
    @assert allunique(job_ids)

    computed_results = OrderedCollections.OrderedDict()

    isempty(job_ids) && return computed_results

    for job_id in job_ids
        all_files = filter(
            file -> is_rms_file(file, expected_filename_prefix),
            readdir(joinpath(job_id, subfolder); join = true),
        )
        computed_results[job_id] = false
        for file in all_files # parses and returns the last file sorted by name
            computed_results[job_id] = parse_file(file)
        end
    end
    return computed_results
end

"""
    print_rms_summary(io::IO = stdout; results::AbstractDict)

Print all computed RMS comparison results returned by `get_computed_rms`.
"""
function print_rms_summary(io::IO = stdout; results::AbstractDict)
    println(io, "################################# Computed RMS results")
    for job_id in keys(results)
        results[job_id] isa AbstractDict || continue
        for var in keys(results[job_id])
            println(io, "RMS[\"$job_id\"][$(var)] = $(results[job_id][var])")
        end
    end
    println(io, "#################################")
end

"""
    print_skipped_jobs(io::IO = stdout; results::AbstractDict)

Return `true` if any jobs were skipped (i.e., `get_computed_rms` returned
`false` for at least one job ID because no comparison file was found).
"""
function print_skipped_jobs(io::IO = stdout; results::AbstractDict)
    any_skipped = any(x -> x isa Bool, values(results))
    if any_skipped
        println(io, "Skipped files:")
        for job_id in keys(results)
            results[job_id] isa Bool && continue
            println(io, "     job_id:`$job_id`, file:`$(results[job_id])`")
        end
    end
    return any_skipped
end

import ArgParse

function reproducibility_test_params()
    s = ArgParse.ArgParseSettings()
    ArgParse.@add_arg_table! s begin
        "--job_id"
        help = "Uniquely identifying string for a particular job"
        arg_type = String
        "--out_dir"
        help = "Output data directory"
        arg_type = String
        "--test_broken_report_flakiness"
        help = "Bool indicating that the job is flaky, use `@test_broken` on flaky job and report flakiness"
        arg_type = Bool
        default = false
    end
    parsed_args = ArgParse.parse_args(ARGS, s)
    job_id = parsed_args["job_id"]
    out_dir = parsed_args["out_dir"]
    test_broken_report_flakiness = parsed_args["test_broken_report_flakiness"]
    return (; job_id, out_dir, test_broken_report_flakiness)
end
