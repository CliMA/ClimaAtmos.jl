import Profile

ENV["CI_PERF_SKIP_RUN"] = true # we only need haskey(ENV, "CI_PERF_SKIP_RUN") == true

filename = joinpath(dirname(@__DIR__), "examples", "hybrid", "driver.jl")

try
    include(filename)
catch err
    if err.error !== :exit_profile
        rethrow(err.error)
    end
end

import JET

OrdinaryDiffEq.step!(integrator) # Make sure no errors
JET.@test_opt OrdinaryDiffEq.step!(integrator)
