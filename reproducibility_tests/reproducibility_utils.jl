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

 - `commit_hash/job_id/output_dir/`
 - `commit_hash/job_id/output_dir/reproducibility_bundle/`
 - `commit_hash/job_id/output_dir/reproducibility_bundle/ref_counter.jl`
 - `commit_hash/job_id/output_dir/reproducibility_bundle/prog_state.hdf5`

 - `commit_hash/reproducibility_bundle/ref_counter.jl`
 - `commit_hash/reproducibility_bundle/job_id/`
 - `commit_hash/reproducibility_bundle/job_id/prog_state.hdf5`

In other words, we strip out `output_dir/`, and swap `job_id` and
`reproducibility_bundle`. This is done for two reasons:

 - The ref_counter is job-independent, hence the swap
 - The `output_dir/` is redundant to the purpose of the commit hash folder

################################################################################
=#

# debug_reproducibility() = true
debug_reproducibility() =
    get(ENV, "BUILDKITE_PIPELINE_SLUG", nothing) == "climaatmos-ci"

import Dates
import OrderedCollections

function string_all_files_in_dir(dir)
    msg = "Files in dir $dir\n"
    for file in all_files_in_dir(dir)
        msg *= "     $file\n"
    end
    return msg
end

function all_files_in_dir(dir)
    all_files = String[]
    for (root, dirs, files) in walkdir(dir; follow_symlinks = true)
        for file in files
            push!(all_files, joinpath(root, file))
        end
    end
    return all_files
end

read_ref_counter(file) = parse(Int, first(readlines(file)))

"""
    sorted_dirs_with_matched_files(; dir = pwd(), filename)

Return an array of subdirectories of `dir` (defaults to the current working
directory) sorted by the reference counters contained in the folders. Return an
empty vector if no subdirectories are found.

This function recurses through `dir`, and finds all directories that have the
file `filename`.
"""
function sorted_dirs_with_matched_files(;
    dir = pwd(),
    filename = "ref_counter.jl",
)
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
        root_dir = "/central/scratch/esm/slurm-buildkite/climaatmos-main",
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
    root_dir = "/central/scratch/esm/slurm-buildkite/climaatmos-main",
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
    isnothing(ref_counter_bins) && return String[]
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
    root_dir::String = "/central/scratch/esm/slurm-buildkite/climaatmos-main";
    filename = "ref_counter.jl",
)
    dirs = sorted_dirs_with_matched_files(;
        dir = root_dir,
        filename = "ref_counter.jl",
    )
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
    msg = "Bins:\n"
    for (i, bin) in enumerate(bins)
        msg *= "  Bin $i:\n"
        for (j, state) in enumerate(bin)
            ref_counter = read_ref_counter(joinpath(state, "ref_counter.jl"))
            msg *= "    (State $j, ref_counter): ($state, $ref_counter)\n"
        end
    end
    return msg
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
    filename = "ref_counter.jl",
)
    dirs = sorted_dirs_with_matched_files(; dir = root_dir, filename)
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
    source_checksum(dir = pwd())

Return a hash from the contents of all Julia files found recursively in `dir`
(defaults to `pwd`).
"""
function source_checksum(dir = pwd())
    jl_files = String[]
    for (root, dirs, files) in walkdir(dir; follow_symlinks = true)
        for file in files
            endswith(file, ".jl") && push!(jl_files, joinpath(root, file))
        end
    end
    all_contents = map(jl_files) do jl_file
        readlines(jl_file)
    end
    joined_contents = join(all_contents, "\n")
    return hash(joined_contents)
end

function source_has_changed(;
    n = 5,
    root_dir = "/central/scratch/esm/slurm-buildkite/climaatmos-main",
    ref_counter_PR = read_ref_counter(joinpath(@__DIR__, "ref_counter.jl")),
    skip = get(ENV, "BUILDKITE_PIPELINE_SLUG", nothing) != "climaatmos-ci",
    src_dir = dirname(@__DIR__),
)
    dirs = latest_comparable_dirs(; n, root_dir, ref_counter_PR, skip)
    isempty(dirs) && return true
    latest_reference_checksum = joinpath(dirs[1], "source_checksum.dat")
    if isfile(latest_reference_checksum)
        src_checksum =
            parse(UInt64, first(readlines(latest_reference_checksum)))
        if source_checksum(src_dir) == src_checksum
            return false # all julia files are the same
        else
            return true
        end
    else
        return true
    end
end

rm_folder(path; strip_folder) =
    joinpath(filter(x -> !occursin(strip_folder, x), splitpath(path))...)

"""
    move_data_to_save_dir(;
        dest_root = "/central/scratch/esm/slurm-buildkite/climaatmos-main",
        buildkite_ci = get(ENV, "BUILDKITE_PIPELINE_SLUG", nothing) ==
                       "climaatmos-ci",
        commit = get(ENV, "BUILDKITE_COMMIT", nothing),
        branch = get(ENV, "BUILDKITE_BRANCH", nothing),
        in_merge_queue = startswith(branch, "gh-readonly-queue/main/"),
        dirs_src,
        strip_folder = Pair("output_active", ""),
        ref_counter_file_PR = joinpath(@__DIR__, "ref_counter.jl"),
        ref_counter_PR = read_ref_counter(ref_counter_file_PR),
        skip = get(ENV, "BUILDKITE_PIPELINE_SLUG", nothing) != "climaatmos-ci",
        n_hash_characters = 7,
        repro_folder = "reproducibility_bundle",
    )

Moves data in the following way:

for job_id in dest_src
  `job_id/out/repro/ref_counter.jl`  -> `commit_hash/repro/ref_counter.jl`
  `job_id/out/repro/`                -> `commit_hash/repro/job_id/`
  `job_id/out/repro/prog_state.hdf5` -> `commit_hash/repro/job_id/prog_state.hdf5`
end

Note that files not in the `repro` folder are not moved.

In other words, we strip out `out/`, and swap `job_id` and `repro`. This is done
for two reasons:

 - The ref_counter is job-independent, hence the swap
 - The `out/` is redundant to the purpose of the commit hash folder

Data movement will occur when this function is called:

 - on a job run in buildkite
 - when in the merge queue
 - when on the main branch if the `source_checksum` is different from the source
   code in the latest comparable reference
"""
function move_data_to_save_dir(;
    dest_root = "/central/scratch/esm/slurm-buildkite/climaatmos-main",
    buildkite_ci = get(ENV, "BUILDKITE_PIPELINE_SLUG", nothing) ==
                   "climaatmos-ci",
    commit = get(ENV, "BUILDKITE_COMMIT", nothing),
    branch = get(ENV, "BUILDKITE_BRANCH", nothing),
    in_merge_queue = startswith(branch, "gh-readonly-queue/main/"),
    dirs_src,
    strip_folder = "output_active",
    ref_counter_file_PR = joinpath(@__DIR__, "ref_counter.jl"),
    ref_counter_PR = read_ref_counter(ref_counter_file_PR),
    skip = get(ENV, "BUILDKITE_PIPELINE_SLUG", nothing) != "climaatmos-ci",
    n_hash_characters = 7,
    repro_folder = "reproducibility_bundle",
)
    buildkite_ci || return nothing

    # if a contributor manually merged, we still want to move data from scratch
    # to `dest_root`. But if moving data on both conditions means that data
    # will be moved twice if it's merged via the merge queue (and if it is) run
    # again on the main branch. One thing we can do to prevent the redundant
    # movement is to check if the source code has changed:

    @assert isfile(ref_counter_file_PR)
    if in_merge_queue || (
        branch == "main" &&
        source_has_changed(; n = 1, root_dir = dest_root, ref_counter_PR, skip)
    )
        commit_sha = commit[1:min(n_hash_characters, length(commit))]
        mkpath(dest_root)
        dest_dir = joinpath(dest_root, commit_sha)
        mkpath(dest_dir)
        dest_repro = joinpath(dest_dir, repro_folder)
        mkpath(dest_repro)
        # Always move reproducibility data, so that we
        # can compare against multiple references
        for src in dirs_src
            dst = joinpath(dest_repro, basename(src))
            debug_reproducibility() && @info "Repro: moving $src to $dst"
            mv(src, dst; force = true)
        end
        for dst in all_files_in_dir(dest_repro)
            dst_new = rm_folder(dst; strip_folder)
            if debug_reproducibility()
                @show isfile(dst)
                @show dst
                @show dst_new
            end
            if dst ≠ dst_new
                mkpath(dirname(dst_new))
                debug_reproducibility() &&
                    @info "Repro: re-moving $dst to $dst_new"
                mv(dst, dst_new; force = true)
            end
        end
        ref_counter_file_main = joinpath(dest_repro, "ref_counter.jl")
        debug_reproducibility() &&
            @info "Repro: moving $ref_counter_file_PR to $ref_counter_file_main"
        mv(ref_counter_file_PR, ref_counter_file_main; force = true)
        if debug_reproducibility()
            println("####################### SRC")
            for src in dirs_src
                @info(string_all_files_in_dir(src))
            end
            println("####################### DST")
            @info(string_all_files_in_dir(dest_repro))
            println("#######################")
        end
    else
        if debug_reproducibility()
            @warn "Repro: skipping data movement"
            @show in_merge_queue
            @show branch == "main"
            @show source_has_changed(;
                n = 1,
                root_dir = dest_root,
                ref_counter_PR,
                skip,
            )
        end
    end
end

parse_file(file) = eval(Meta.parse(join(readlines(file))))
# parse_file(file) = parse_file_json(file) # doesn't work for some reason
parse_file_json(file) =
    JSON.parsefile(file; dicttype = OrderedCollections.OrderedDict)


#####
##### MSE summary
#####

"""
    default_is_mse_file(file, expected_filename_prefix::String = "computed_mse")::Bool

Returns `true` or `false` if the given file is an Mean-Squared-Error(MSE) file,
based on the basename of the folder and the file extension, and if it has the
expected filename prefix `expected_filename_prefix`.
"""
default_is_mse_file(file, expected_filename_prefix::String = "computed_mse") =
    startswith(basename(file), expected_filename_prefix) &&
    endswith(file, ".json")

"""
    get_computed_mses(;
        job_ids::Vector{String},
        subfolder::String = "output_active",
        is_mse_file::Function = default_is_mse_file
    )

Returns Dict containing either a `Dict` of Dicts containing Mean-Squared-Errors
(MSEs) (if file was found) per variable in a given job ID (the key) or `Bool`
(if file was not found).

given:

 - `job_ids` vector of job IDs
 - `subfolder` sub-folder to find json files
 - `is_mse_file` a function to determine if a given file is an MSE file. See
   `default_is_mse_file` for the used criteria.

It is expected that files exist in the form: `joinpath(job_ids[1], subfolder,
json_filename)`

where `is_mse_file` is `true`.

"""
function get_computed_mses(;
    job_ids::Vector{String},
    subfolder::String = "output_active",
    is_mse_file::Function = default_is_mse_file,
    expected_filename_prefix = "computed_mse",
)
    @assert allunique(job_ids)

    computed_mses = OrderedCollections.OrderedDict()

    isempty(job_ids) && return computed_mses

    for job_id in job_ids
        all_files = readdir(joinpath(job_id, subfolder); join = true)
        for file in all_files
            computed_mses[job_id] =
                if is_mse_file(file, expected_filename_prefix)
                    parse_file(file)
                else
                    false
                end
        end
    end
    return computed_mses
end

"""
    print_mse_summary(io::IO = stdout; mses::AbstractDict)

Prints the Mean-Squared-Errors (MSEs) returned by `get_computed_mses`.
"""
function print_mse_summary(io::IO = stdout; mses::AbstractDict)
    println(io, "################################# Computed MSEs")
    for job_id in keys(mses)
        mses[job_id] isa AbstractDict || continue
        for var in keys(mses[job_id])
            println(io, "MSEs[\"$job_id\"][$(var)] = $(mses[job_id][var])")
        end
    end
    println(io, "#################################")
end

"""
    print_skipped_jobs(io::IO = stdout; mses::AbstractDict)

Returns a Bool indicating if any jobs were skipped. This occurs when
`get_computed_mses` returns a dict with at least one value whose type is a
Bool.
"""
function print_skipped_jobs(io::IO = stdout; mses::AbstractDict)
    any_skipped = any(x -> x isa Bool, values(mses))
    if any_skipped
        println(io, "Skipped files:")
        for job_id in keys(mses)
            mses[job_id] isa Bool && continue
            println(io, "     job_id:`$job_id`, file:`$(mses[job_id])`")
        end
    end
    return any_skipped
end

import OrderedCollections
import JSON
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
