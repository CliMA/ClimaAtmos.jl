# Launch with `julia --project --track-allocation=user`

example_dir = joinpath(dirname(@__DIR__), "examples")

import Profile

filename = joinpath(example_dir, "hybrid", "driver.jl")
ENV["CI_PERF_SKIP_RUN"] = true # we only need haskey(ENV, "CI_PERF_SKIP_RUN") == true
try
    include(filename)
catch err
    if err.error !== :exit_profile
        rethrow(err.error)
    end
end

OrdinaryDiffEq.step!(integrator) # compile first
Profile.clear_malloc_data()
OrdinaryDiffEq.step!(integrator)
