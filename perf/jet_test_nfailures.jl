import Random
Random.seed!(1234)
import ClimaAtmos as CA

include("common.jl")

length(ARGS) < 2 && error("Usage: benchmark.jl <target_job> <job_id>")
target_job = ARGS[1]
job_id = get(ARGS, 2, target_job)

config_dict =
    target_job != "default" ? CA.config_from_target_job(target_job) :
    CA.default_config_dict()
config = AtmosCoveragePerfConfig(; config_dict)
integrator = CA.get_integrator(config)

import JET
import SciMLBase
SciMLBase.step!(integrator) # Make sure no errors

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
    n = @n_failures SciMLBase.step!(integrator)
    # This test is intended to provide some friction when we
    # add code to our tendency function that results in degraded
    # inference. By increasing this counter, we acknowledge that
    # we have introduced an inference failure. We hope to drive
    # this number down to 0.
    n_allowed_failures = 680
    @show n
    @test n ≤ n_allowed_failures
    if n < n_allowed_failures
        @info "Please update the n-failures to $n"
    end
end
