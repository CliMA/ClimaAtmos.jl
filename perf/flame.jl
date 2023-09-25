import Random
Random.seed!(1234)
import ClimaAtmos as CA
config = CA.AtmosCoveragePerfConfig()
integrator = CA.get_integrator(config)

# The callbacks flame graph is very expensive, so only do 2 steps.
@info "running step"

import SciMLBase
SciMLBase.step!(integrator) # compile first
CA.call_all_callbacks!(integrator) # compile callbacks
import Profile, ProfileCanvas
(; output_dir, job_id) = integrator.p.simulation
output_dir = job_id
mkpath(output_dir)


@info "collect profile"
Profile.clear()
prof = Profile.@profile SciMLBase.step!(integrator)
results = Profile.fetch()
Profile.clear()

ProfileCanvas.html_file(joinpath(output_dir, "flame.html"), results)


#####
##### Allocation tests
#####

# use new allocation profiler
@info "collecting allocations"
Profile.Allocs.clear()
Profile.Allocs.@profile sample_rate = 0.01 SciMLBase.step!(integrator)
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
allocs = @allocated SciMLBase.step!(integrator)
@timev SciMLBase.step!(integrator)
@info "`allocs ($job_id)`: $(allocs)"

allocs_limit = Dict()
allocs_limit["flame_perf_target"] = 12864
allocs_limit["flame_perf_target_tracers"] = 212496
allocs_limit["flame_perf_target_edmfx"] = 304064
allocs_limit["flame_perf_diagnostics"] = 3023192
allocs_limit["flame_perf_target_diagnostic_edmfx"] = 754848
allocs_limit["flame_perf_target_edmf"] = 12459299664
allocs_limit["flame_perf_target_threaded"] = 6175664
allocs_limit["flame_perf_target_callbacks"] = 45111592
allocs_limit["flame_perf_gw"] = 4911463328

if allocs < allocs_limit[job_id] * buffer
    @info "TODO: lower `allocs_limit[$job_id]` to: $(allocs)"
end
Δallocs = allocs / allocs_limit[job_id]
@info "Allocation change (allocs/allocs_limit): $Δallocs"

# https://github.com/CliMA/ClimaAtmos.jl/issues/827
@testset "Allocations limit" begin
    @test allocs ≤ allocs_limit[job_id] * buffer
end
