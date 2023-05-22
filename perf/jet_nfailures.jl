# Customizing specific jobs / specs in config_parsed_args.jl:
ca_dir = joinpath(dirname(@__DIR__));
include(joinpath(ca_dir, "perf", "config_parsed_args.jl")) # defines parsed_args

ENV["CI_PERF_SKIP_RUN"] = true # we only need haskey(ENV, "CI_PERF_SKIP_RUN") == true

filename = joinpath(ca_dir, "examples", "hybrid", "driver.jl")

try # capture integrator
    include(filename)
catch err
    if err.error !== :exit_profile
        rethrow(err.error)
    end
end

import JET

OrdinaryDiffEq.step!(integrator) # Make sure no errors

# Suggested in: https://github.com/aviatesk/JET.jl/issues/455
macro n_failures(ex)
    return :(
        let result = JET.@report_call $(ex)
            length(JET.get_reports(result.analyzer, result.result))
        end
    )
end

using Test
@testset "Test N-jet failures" begin
    n = @n_failures OrdinaryDiffEq.step!(integrator)
    # This test is intended to provide some friction when we
    # add code to our tendency function that results in degraded
    # inference. By increasing this counter, we acknowledge that
    # we have introduced an inference failure. We hope to drive
    # this number down to 0.
    n_allowed_failures = 27
    @test n ≤ n_allowed_failures
    if n < n_allowed_failures
        @info "Please update the n-failures to $n"
    end
end
