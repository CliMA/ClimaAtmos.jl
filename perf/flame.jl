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

# The callbacks flame graph is very expensive, so only do 2 steps.
const n_samples = occursin("callbacks", parsed_args["job_id"]) ? 2 : 20

function do_work!(integrator)
    for _ in 1:n_samples
        OrdinaryDiffEq.step!(integrator)
    end
end

do_work!(integrator) # compile first
import Profile
Profile.clear_malloc_data()
Profile.clear()
prof = Profile.@profile begin
    do_work!(integrator)
end

(; output_dir, job_id) = simulation

import ProfileCanvas

if haskey(ENV, "BUILDKITE_COMMIT") || haskey(ENV, "BUILDKITE_BRANCH")
    output_dir = job_id
    mkpath(output_dir)
    ProfileCanvas.html_file(joinpath(output_dir, "flame.html"))
else
    ProfileCanvas.view(Profile.fetch())
end


#####
##### Allocation tests
#####

# We're grouping allocation tests here for convenience.

using Test
# Threaded allocations are not deterministic, so let's add a buffer
# TODO: remove buffer, and threaded tests, when
#       threaded/unthreaded functions are unified
buffer = occursin("threaded", job_id) ? 1.4 : 1

allocs = @allocated OrdinaryDiffEq.step!(integrator)
@timev OrdinaryDiffEq.step!(integrator)
@info "`allocs ($job_id)`: $(allocs)"

allocs_limit = Dict()
allocs_limit["flame_perf_target"] = 9360
allocs_limit["flame_perf_target_tracers"] = 6245350392
allocs_limit["flame_perf_target_edmf"] = 15003862184
allocs_limit["flame_perf_target_threaded"] = 4431840
allocs_limit["flame_perf_target_callbacks"] = 11439104

if allocs < allocs_limit[job_id] * buffer
    @info "TODO: lower `allocs_limit[$job_id]` to: $(allocs)"
end
Δallocs = allocs / allocs_limit[job_id]
@info "Allocation change (allocs/allocs_limit): $Δallocs"

# https://github.com/CliMA/ClimaAtmos.jl/issues/827
@testset "Allocations limit" begin
    @test allocs ≤ allocs_limit[job_id]
end
