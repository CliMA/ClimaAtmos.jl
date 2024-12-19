#=
Please see ClimaAtmos.jl/reproducibility_tests/README.md
for a more detailed information on how reproducibility tests work.
=#
@info "########################################## Reproducibility tests"

include(joinpath(@__DIR__, "reproducibility_test_job_ids.jl"))
include(joinpath(@__DIR__, "reproducibility_tools.jl"))

(; job_id, out_dir, test_broken_report_flakiness) =
    reproducibility_test_params()

computed_mse_filenames = map(filter(default_is_mse_file, readdir(out_dir))) do x
    joinpath(out_dir, x)
end
@show computed_mse_filenames
sources = map(x -> basename(dirname(dirname(x))), computed_mse_filenames)

results = report_reproducibility_results(
    sources,
    computed_mse_filenames;
    test_broken_report_flakiness,
)

if test_broken_report_flakiness
    @test results == :not_yet_reproducible
    @test_broken results == :now_reproducible
else
    @test results == :reproducible
end

@info "##########################################"
