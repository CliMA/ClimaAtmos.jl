all_lines = readlines(joinpath(@__DIR__, "mse_tables.jl"))
lines = deepcopy(all_lines)
filter!(x -> occursin("] = OrderedCollections", x), lines)
job_ids = getindex.(split.(lines, "\""), 2)
@assert count(x -> occursin("OrderedDict", x), all_lines) == length(job_ids) + 1
@assert length(job_ids) ≠ 0 # safety net

if haskey(ENV, "BUILDKITE_COMMIT") && haskey(ENV, "BUILDKITE_BRANCH")
    commit = ENV["BUILDKITE_COMMIT"]
    branch = ENV["BUILDKITE_BRANCH"]
    # Note: cluster_data_prefix is also defined in compute_mse.jl
    cluster_data_prefix = "/central/scratch/esm/slurm-buildkite/climaatmos-main"

    @info "pwd() = $(pwd())"
    @info "branch = $(branch)"
    @info "commit = $(commit)"

    using Glob
    @show readdir(joinpath(@__DIR__, ".."))
    if branch == "staging"
        commit_sha = commit[1:7]
        mkpath(cluster_data_prefix)
        path = joinpath(cluster_data_prefix, commit_sha)
        mkpath(path)
        for folder_name in job_ids
            src = folder_name
            dst = joinpath(path, folder_name)
            @info "Moving $src to $dst"
            mv(src, dst; force = true)
        end
        ref_counter_file_PR = joinpath(@__DIR__, "ref_counter.jl")
        ref_counter_file_main = joinpath(path, "ref_counter.jl")
        mv(ref_counter_file_PR, ref_counter_file_main; force = true)
        @info "readdir(): $(readdir(path))"
    end
else
    @info "ENV keys: $(keys(ENV))"
end
