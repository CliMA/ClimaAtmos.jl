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
@info "running step"

OrdinaryDiffEq.step!(integrator) # compile first
import Profile, ProfileCanvas
(; output_dir, job_id) = simulation
output_dir = job_id
mkpath(output_dir)


@info "collect profile"
Profile.clear()
prof = Profile.@profile OrdinaryDiffEq.step!(integrator)
results = Profile.fetch()
Profile.clear()

ProfileCanvas.html_file(joinpath(output_dir, "flame.html"), results)


#####
##### Allocation tests
#####

# use new allocation profiler
@info "collecting allocations"
Profile.Allocs.clear()
Profile.Allocs.@profile sample_rate = 0.01 OrdinaryDiffEq.step!(integrator)
results = Profile.Allocs.fetch()
Profile.Allocs.clear()
profile = ProfileCanvas.view_allocs(results)
ProfileCanvas.html_file(joinpath(output_dir, "allocs.html"), profile)

# We're grouping allocation tests here for convenience.

@info "testing allocations"
using Test
# Threaded allocations are not deterministic, so let's add a buffer
# TODO: remove buffer, and threaded tests, when
#       threaded/unthreaded functions are unified
buffer = occursin("threaded", job_id) ? 1.4 : 1


## old allocation profiler (TODO: remove this)
allocs = @allocated OrdinaryDiffEq.step!(integrator)
@timev OrdinaryDiffEq.step!(integrator)
@info "`allocs ($job_id)`: $(allocs)"

allocs_limit = Dict()
allocs_limit["flame_perf_target"] = 764512
allocs_limit["flame_perf_target_tracers"] = 5685484032
allocs_limit["flame_perf_target_edmfx"] = 764688
allocs_limit["flame_perf_target_edmf"] = 9799934480
allocs_limit["flame_perf_target_threaded"] = 1680000 # round this up; it can fluctuate
allocs_limit["flame_perf_target_callbacks"] = 11920024

if allocs < allocs_limit[job_id] * buffer
    @info "TODO: lower `allocs_limit[$job_id]` to: $(allocs)"
end
Δallocs = allocs / allocs_limit[job_id]
@info "Allocation change (allocs/allocs_limit): $Δallocs"

# https://github.com/CliMA/ClimaAtmos.jl/issues/827
@testset "Allocations limit" begin
    @test allocs ≤ allocs_limit[job_id]
end
