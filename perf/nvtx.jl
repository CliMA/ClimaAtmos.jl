# Launch with `julia --project --track-allocation=user`

example_dir = joinpath(dirname(@__DIR__), "examples")

filename = joinpath(example_dir, "hybrid", "driver.jl")
ENV["CI_PERF_SKIP_RUN"] = true # we only need haskey(ENV, "CI_PERF_SKIP_RUN") == true
try
    include(filename)
catch err
    if err.error !== :exit_profile
        rethrow(err.error)
    end
end

import NVTX

OrdinaryDiffEq.step!(integrator) # compile first

nvtx_step = NVTX.range_start(; message = "step!", color = colorant"yellow")

OrdinaryDiffEq.step!(integrator)

NVTX.range_end(nvtx_step)

