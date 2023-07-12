import Random
Random.seed!(1234)
import ClimaAtmos as CA
config = CA.AtmosPerfConfig()
integrator = CA.get_integrator(config)

import JET

import OrdinaryDiffEq
OrdinaryDiffEq.step!(integrator) # Make sure no errors

# Suggested in: https://github.com/aviatesk/JET.jl/issues/455
macro n_failures(ex)
    return :(
        let result = JET.@report_opt $(ex)
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
    n_allowed_failures = 823
    @test n ≤ n_allowed_failures
    if n < n_allowed_failures
        @info "Please update the n-failures to $n"
    end
end
