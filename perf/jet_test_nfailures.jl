redirect_stderr(IOContext(stderr, :stacktrace_types_limited => Ref(false)))
import ClimaComms
ClimaComms.@import_required_backends
import Random
Random.seed!(1234)
import ClimaAtmos as CA

include("common.jl")

(; config_file, job_id) = CA.commandline_kwargs()
config = CA.AtmosConfig(config_file; job_id)

simulation = CA.get_simulation(config)
(; integrator) = simulation

import JET
import SciMLBase
SciMLBase.step!(integrator) # Make sure no errors

import HDF5, NCDatasets, CUDA
# Suggested in: https://github.com/aviatesk/JET.jl/issues/455
macro n_failures(ex)
    return :(
        let result =
                JET.@report_opt ignored_modules = (HDF5, NCDatasets, CUDA) $(
                    ex
                )
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
    n_allowed_failures = 253
    @show n
    @test n ≤ n_allowed_failures
    if n < n_allowed_failures
        @info "Please update the n-failures to $n"
    end
end
