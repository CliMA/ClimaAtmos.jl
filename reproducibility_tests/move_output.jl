include(joinpath(@__DIR__, "reproducibility_utils.jl"))

all_lines = readlines(joinpath(@__DIR__, "mse_tables.jl"))
job_ids = reproducibility_test_job_ids

move_data_to_save_dir(;
    dirs_src = job_ids,
    ref_counter_file_PR = joinpath(@__DIR__, "ref_counter.jl"),
)

if buildkite_ci && in_merge_queue
    folders = get_reference_dirs_to_delete(; root_dir = cluster_data_prefix)
    for f in folders
        rm(f; recursive = true, force = true)
    end
end
