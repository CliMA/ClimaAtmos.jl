import Profile

ENV["CI_PERF_SKIP_RUN"] = true # we only need haskey(ENV, "CI_PERF_SKIP_RUN") == true

filename = joinpath(dirname(@__DIR__), "examples", "hybrid", "driver.jl")

# Uncomment for customizing specific jobs / specs:
# dict = parsed_args_per_job_id(; trigger = "benchmark.jl"); # if job_id uses benchmark.jl
# dict = parsed_args_per_job_id();                           # if job_id uses driver.jl
# parsed_args = dict["sphere_aquaplanet_rhoe_equilmoist_allsky"];

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
