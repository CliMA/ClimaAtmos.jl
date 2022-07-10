import Dates

function find_latest_dataset_folder(; dir = pwd())
    matching_paths = String[]
    for file in readdir(dir)
        !ispath(joinpath(dir, file)) && continue
        push!(matching_paths, joinpath(dir, file))
    end
    isempty(matching_paths) && return ""
    # sort by timestamp
    sorted_paths =
        sort(matching_paths; by = f -> Dates.unix2datetime(stat(f).mtime))
    return pop!(sorted_paths)
end

cluster_data_prefix = "/central/scratch/esm/slurm-buildkite/climaatmos-main"
path = find_latest_dataset_folder(; dir = cluster_data_prefix)
ref_counter = 0 # (error)
if isempty(path) # no folders found
    ref_counter = 1
    @warn "path: `$path` is empty, setting `ref_counter = 1`"
elseif !isfile(joinpath(path, "ref_counter.jl")) # no file found
    @warn "file `$(joinpath(path, "ref_counter.jl"))` not found"
    @info "readdir(path) = `$(readdir(path))`"
    # This may be the very first self-reference,
    # in which case, verify, allow, and warn.
    ref_counter_file_PR = joinpath(@__DIR__, "ref_counter.jl")
    ref_counter_PR = parse(Int, first(readlines(ref_counter_file_PR)))
    if ref_counter_PR == 1 # The very first self-reference
        @warn "Assuming (very first) self reference"
        ref_counter = 1
    end
else
    @info "Ref counter file found in path:`$path`"
    ref_counter_contents = readlines(joinpath(path, "ref_counter.jl"))
    @info "`$(path)/ref_counter.jl` contents: `$(ref_counter_contents)`"
    ref_counter = parse(Int, first(ref_counter_contents))
    @info "Old reference counter: `$(ref_counter)`"
    ref_counter += 1 # increment counter
    @info "New reference counter: `$(ref_counter)`"
end

ref_counter == 0 && error("Uncaught case")

msg = ""
msg *= "Pull request author:\n"
msg *= "Copy the reference counter below (only\n"
msg *= "the number) and paste into the file:\n\n"
msg *= "    `regression_tests/ref_counter.jl`\n\n"
msg *= "if this PR satisfies one of the following:\n"
msg *= "   - Variable name has changed\n"
msg *= "   - A new regression test was added\n"
msg *= "   - Grid resolution has changed\n\n"
msg *= "For more information, please find\n"
msg *= "`regression_tests/README.md` and read the section\n\n"
msg *= "  `How to merge pull requests (PR) that get approved\n"
msg *= "   but *break* regression tests`\n\n"
msg *= "for how to merge this PR."

@info msg

println("------------")
println("------------")
println("------------")
println("$ref_counter")
println("------------")
println("------------")
println("------------")
