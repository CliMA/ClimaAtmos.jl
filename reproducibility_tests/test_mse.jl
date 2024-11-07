#=
Please see ClimaAtmos.jl/reproducibility_tests/README.md
for a more detailed information on how reproducibility tests work.
=#
@info "##########################################"
@info "########################################## Reproducibility tests"
@info "##########################################"

import OrderedCollections
import JSON
import ArgParse

function parse_commandline()
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
    return (s, parsed_args)
end

function get_params()
    (s, parsed_args) = parse_commandline()
    job_id = parsed_args["job_id"]
    out_dir = parsed_args["out_dir"]
    test_broken_report_flakiness = parsed_args["test_broken_report_flakiness"]
    return (; job_id, out_dir, test_broken_report_flakiness)
end

(; job_id, out_dir, test_broken_report_flakiness) = get_params()
include(joinpath(@__DIR__, "mse_tables.jl"))
best_mse = all_best_mse[job_id]
best_mse_string =
    Dict(map(x -> string(x) => best_mse[x], collect(keys(best_mse))))

import ClimaReproducibilityTests as CRT
using Test
function test_reproducibility_results(
    computed_mse_filenames;
    test_broken_report_flakiness,
)
    @info "computed_mse_filenames: $computed_mse_filenames"
    @info "isfile.(computed_mse_filenames): $(isfile.(computed_mse_filenames))"
    n_passes = 0
    for computed_mse_filename in computed_mse_filenames
        computed_mse = JSON.parsefile(
            computed_mse_filename;
            dicttype = OrderedCollections.OrderedDict,
        )

        if test_broken_report_flakiness
            all_reproducible = true
            for (var, reproducible) in CRT.test_mse(; computed_mse)
                reproducible || (all_reproducible = false)
                @show var, reproducible
            end
            all_reproducible && (n_passes += 1)
        else
            for (var, reproducible) in CRT.test_mse(; computed_mse)
                @test reproducible
            end
        end
    end

    n_allowed_passes = 5
    # If we successfully compare against 5 other jobs,
    # let's error and tell the user that the job now
    # seems reproducible.
    if test_broken_report_flakiness
        if n_passes > n_allowed_passes
            now_reproducible = true
            @test_broken now_reproducible
        else
            n_times_reproducible = n_passes
            n_times_not_reproducible = length(computed_mse_filenames) - n_passes
            @show n_times_reproducible
            @show n_times_not_reproducible
        end
    end
end
is_mse_file(x) = startswith(basename(x), "computed_mse") && endswith(x, ".json")
@show readdir(out_dir)
computed_mse_filenames = filter(is_mse_file, readdir(out_dir))
computed_mse_filenames =
    map(fn -> joinpath(out_dir, fn), computed_mse_filenames)
test_reproducibility_results(
    computed_mse_filenames;
    test_broken_report_flakiness,
)

@info "##########################################"
@info "##########################################"
@info "##########################################"
