#=
Please see ClimaAtmos.jl/reproducibility_tests/README.md
for a more detailed information on how reproducibility tests work.
=#
@info "########################################## Reproducibility tests"

include(joinpath(@__DIR__, "reproducibility_test_job_ids.jl"))
include(joinpath(@__DIR__, "reproducibility_tools.jl"))

(; job_id, out_dir, test_broken_report_flakiness) =
    reproducibility_test_params()

debug = true
repro_dir = joinpath(out_dir, "reproducibility_bundle")
computed_mse_files = map(filter(default_is_mse_file, readdir(repro_dir))) do x
    joinpath(repro_dir, x)
end
if isempty(computed_mse_files)
    @warn "No reproducibility tests performed, due to non-existent comparable data."
    debug && @show readdir(out_dir)
    debug && @show readdir(repro_dir)
    debug && @show filter(default_is_mse_file, readdir(repro_dir))
    dirs = latest_comparable_dirs()
    if isempty(dirs) # no comparable references
        bins = compute_bins() # all reproducible bins, may or may not be comparable
        if isempty(bins)
            @warn "No reproducibility data found"
        else
            # Let's assert that the ref counter was incremented
            newest_saved_dir_new =
                joinpath(bins[1][1], "reproducibility_bundle")
            newest_saved_dir_legacy = joinpath(bins[1][1])
            newest_saved_dir = if ispath(newest_saved_dir_new)
                newest_saved_dir_new
            elseif ispath(newest_saved_dir_legacy)
                newest_saved_dir_legacy
            else
                error("Newest saved directory not found.")
            end
            newest_saved_ref_counter =
                read_ref_counter(joinpath(newest_saved_dir, "ref_counter.jl"))
            ref_counter_PR =
                read_ref_counter(joinpath(@__DIR__, "ref_counter.jl"))
            if ref_counter_PR ≠ newest_saved_ref_counter + 1
                if debug_reproducibility()
                    @info "    ref_counter_PR=$ref_counter_PR, newest_saved_ref_counter=$newest_saved_ref_counter\n"
                    @info "newest_saved_dir: $newest_saved_dir\n"
                    @info "newest_saved_dir_legacy: $newest_saved_dir_legacy\n"
                    @info "newest_saved_dir_new: $newest_saved_dir_new\n"
                    print_bins(bins)
                end
                error("Reference counter must be incremented by 1.")
            end
        end
    else
        msg = "There were comparable references, but no computed mse files exist."
        msg *= "\nPlease open an issue with this message."
        error(msg)
    end
else
    @testset "Reproducibility tests" begin
        commit_hashes =
            map(x -> commit_sha_from_mse_file(x), computed_mse_files)
        computed_mses = map(x -> parse_file(x), computed_mse_files)
        if debug_reproducibility()
            println("------ in test_mse.jl")
            @show computed_mses
            println("------")
        end
        results = report_reproducibility_results(
            commit_hashes,
            computed_mses;
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
