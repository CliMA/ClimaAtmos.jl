import OrderedCollections
import JSON

# Get cases from JobIDs in mse_tables file:
include(joinpath(@__DIR__, "reproducibility_utils.jl"))
paths = latest_comparable_dirs()
isempty(paths) && @warn string("No comparable references.")

include(joinpath(@__DIR__, "reproducibility_test_job_ids.jl"))
job_ids = reproducibility_test_job_ids()

computed_mses = get_computed_mses(; job_ids)
isempty(computed_mses) && @warn "No MSEs were computed"
print_mse_summary(; mses = computed_mses)
print_skipped_jobs(; mses = computed_mses)

# # Cleanup
# for job_id in job_ids
#     all_files = readdir(job_id)
#     mse_filenames = filter(is_mse_file, all_files)
#     for f in mse_filenames
#         rm(f; force = true)
#     end
# end
