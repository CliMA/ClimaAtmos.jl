#=
Please see ClimaAtmos.jl/reproducibility_tests/README.md
for more detailed information on how reproducibility tests work.
=#
@info "########################################## Reproducibility tests"

include(joinpath(@__DIR__, "reproducibility_test_job_ids.jl"))
include(joinpath(@__DIR__, "reproducibility_tools.jl"))

(; job_id, out_dir, test_broken_report_flakiness) =
    reproducibility_test_params()

repro_dir = joinpath(out_dir, "reproducibility_bundle")

# Find computed RMS comparison files
computed_rms_files = map(filter(default_is_rms_file, readdir(repro_dir))) do x
    joinpath(repro_dir, x)
end

if isempty(computed_rms_files)
    @warn "No reproducibility tests performed, due to non-existent comparable data."
    if debug_reproducibility()
        @show readdir(out_dir)
        @show readdir(repro_dir)
        @show filter(default_is_rms_file, readdir(repro_dir))
    end
    dirs = latest_comparable_dirs()
    if isempty(dirs) # no comparable references
        bins = compute_bins()
        if isempty(bins)
            @warn "No reproducibility data found"
        else
            # Verify the ref counter was incremented
            newest_saved_dir =
                joinpath(bins[1][1], "reproducibility_bundle")
            if !ispath(newest_saved_dir)
                newest_saved_dir = bins[1][1]
            end
            ispath(newest_saved_dir) ||
                error("Newest saved directory not found.")

            newest_saved_ref_counter =
                read_ref_counter(joinpath(newest_saved_dir, "ref_counter.jl"))
            ref_counter_PR =
                read_ref_counter(joinpath(@__DIR__, "ref_counter.jl"))
            if ref_counter_PR ≠ newest_saved_ref_counter + 1
                if debug_reproducibility()
                    @info "    ref_counter_PR=$ref_counter_PR, newest_saved_ref_counter=$newest_saved_ref_counter\n"
                    @info "newest_saved_dir: $newest_saved_dir\n"
                    print_bins(bins)
                end
                error("Reference counter must be incremented by 1.")
            end
        end
    else
        msg = "There were comparable references, but no computed RMS files exist."
        msg *= "\nPlease open an issue with this message."
        error(msg)
    end
else
    @testset "Reproducibility tests" begin
        commit_hashes =
            map(x -> commit_sha_from_rms_file(x), computed_rms_files)
        computed_rms_results = map(x -> parse_file(x), computed_rms_files)

        results = report_reproducibility_results(
            commit_hashes,
            computed_rms_results;
            test_broken_report_flakiness,
        )

        if test_broken_report_flakiness
            @test results == :not_yet_reproducible
            @test_broken results == :now_reproducible
        else
            @test results == :reproducible
        end
    end
end

@info "##########################################"
