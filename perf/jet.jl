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
JET.@test_opt OrdinaryDiffEq.step!(integrator)
