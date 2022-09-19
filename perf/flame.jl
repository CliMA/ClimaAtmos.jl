ca_dir = joinpath(dirname(@__DIR__));
include(joinpath(ca_dir, "examples", "hybrid", "cli_options.jl"));

ENV["CI_PERF_SKIP_RUN"] = true # we only need haskey(ENV, "CI_PERF_SKIP_RUN") == true

filename = joinpath(ca_dir, "examples", "hybrid", "driver.jl")
dict = parsed_args_per_job_id(; trigger = "benchmark.jl")
parsed_args_prescribed = parsed_args_from_ARGS(ARGS)

# Start with performance target, but override anything provided in ARGS
parsed_args_target = dict["perf_target_unthreaded"];
parsed_args = merge(parsed_args_target, parsed_args_prescribed);

# The callbacks flame graph is very expensive, so only do 2 steps.
const n_samples = occursin("callbacks", parsed_args["job_id"]) ? 2 : 20

try # capture integrator
    include(filename)
catch err
    if err.error !== :exit_profile
        rethrow(err.error)
    end
end

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
include("profile_canvas_patch.jl")

if haskey(ENV, "BUILDKITE_COMMIT") || haskey(ENV, "BUILDKITE_BRANCH")
    output_dir = job_id
    mkpath(output_dir)
    html_file(joinpath(output_dir, "flame.html"))
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
allocs_limit["flame_perf_target_rhoe"] = 10575616
allocs_limit["flame_perf_target_rhoe_threaded"] = 12078608
allocs_limit["flame_perf_target_rhoe_callbacks"] = 21740792
allocs_limit["flame_perf_target_rhoe_ARS343"] = 24056711

if allocs < allocs_limit[job_id] * buffer
    @info "TODO: lower `allocs_limit[$job_id]` to: $(allocs)"
end
Δallocs = allocs / allocs_limit[job_id]
@info "Allocation change (allocs/allocs_limit): $Δallocs"

# https://github.com/CliMA/ClimaAtmos.jl/issues/827
@testset "Allocations limit" begin
    @test allocs ≤ allocs_limit[job_id]
end
