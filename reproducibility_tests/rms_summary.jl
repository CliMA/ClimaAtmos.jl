import OrderedCollections

# Get cases from JobIDs in the reproducibility job list:
include(joinpath(@__DIR__, "reproducibility_utils.jl"))
paths = latest_comparable_dirs()
isempty(paths) && @warn string("No comparable references.")

include(joinpath(@__DIR__, "reproducibility_test_job_ids.jl"))
job_ids = reproducibility_test_job_ids()

# Print summary of computed RMS results for all tracked jobs
computed_results = get_computed_rms(; job_ids)
isempty(computed_results) && @warn "No comparison results were computed"
print_rms_summary(; results = computed_results)
print_skipped_jobs(; results = computed_results)
