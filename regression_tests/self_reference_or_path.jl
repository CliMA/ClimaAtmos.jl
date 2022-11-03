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

function self_reference_or_path()

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

    ref_counters_main = Vector{Int}(undef, length(sorted_paths))
    ref_counters_main .= -1
    for (i, path) in enumerate(sorted_paths)
        ref_counter_file_main = joinpath(path, "ref_counter.jl")
        !isfile(ref_counter_file_main) && continue
        ref_counters_main[i] =
            parse(Int, first(readlines(ref_counter_file_main)))
    end
    i_oldest_reference = findfirst(ref_counters_main) do ref_counter_main
        ref_counter_main == ref_counter_PR
    end
    if i_oldest_reference == nothing
        @warn "`ref_counter.jl` not found on main, assuming self-reference"
        @info "Please review output results before merging."
        return :self_reference
    end
    # Oldest reference path:
    path = sorted_paths[i_oldest_reference]
    ref_counter_file_main = joinpath(path, "ref_counter.jl")

    @info "Files on main:" # for debugging
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

"""
    get_main_branch_buildkite_path()

Return the folder of the most recent buildkite run that contains a `ref_counter.jl` file,
and therefore also contains hdf5 files with output data.

This is very similar to `self_reference_or_path`, except that it always returns a path, not
`:self_reference`. Functionality could be merged in a future PR.
"""
function get_main_branch_buildkite_path()
    # TODO: Combine this method with `self_reference_or_path`, above.
    cluster_data_prefix = "/central/scratch/esm/slurm-buildkite/climaatmos-main"
    if !ispath(cluster_data_prefix)
        if haskey(ENV, "BUILDKITE_COMMIT") || haskey(ENV, "BUILDKITE_BRANCH")
            error("No path found for comparison against main")
        else
            @warn "Buildkite reference data is not available locally; skipping \
                   comparison against main"
            return ""
        end
    end

    buildkite_paths = readdir(cluster_data_prefix; join = true)
    filter!(buildkite_paths) do path
        isdir(path)
    end
    if isempty(buildkite_paths)
        @warn("no buildkite paths found")
        return ""
    end
    sort!(
        buildkite_paths;
        by = f -> Dates.unix2datetime(stat(f).mtime),
        rev = true,
    )  # most recent build first

    # Find ref counters
    ref_counters = -ones(Int, length(buildkite_paths))
    for (i, path) in enumerate(buildkite_paths)
        ref_counter_file_main = joinpath(path, "ref_counter.jl")
        if isfile(ref_counter_file_main)  # set ref_counter value if file exists.
            ref_counters[i] =
                parse(Int, first(readlines(ref_counter_file_main)))
        end
    end
    if all(ref_counters .== -1)
        @warn("No `ref_counter.jl` files found")
        return ""
    end

    most_recent_max_counter_ind = argmax(ref_counters)
    main_branch_data_path = buildkite_paths[most_recent_max_counter_ind]
    @info "Comparing PR profiles against main with commit id: \
           $(basename(main_branch_data_path))"
    return main_branch_data_path
end
