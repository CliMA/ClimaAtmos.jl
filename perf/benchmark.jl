example_dir = joinpath(dirname(@__DIR__), "examples")


import BenchmarkTools
ENV["CI_PERF_SKIP_RUN"] = true # we only need haskey(ENV, "CI_PERF_SKIP_RUN") == true

filename = joinpath(example_dir, "hybrid", "driver.jl")

try
    include(filename)
catch err
    if err.error !== :exit_profile
        rethrow(err.error)
    end
end

OrdinaryDiffEq.step!(integrator) # compile first
trial = BenchmarkTools.@benchmark OrdinaryDiffEq.step!($integrator)
show(stdout, MIME("text/plain"), trial)
println()
