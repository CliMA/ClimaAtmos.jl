import Random
Random.seed!(1234)
import ClimaAtmos as CA

include("common.jl")

length(ARGS) != 1 && error("Usage: flame.jl <config_file>")
config_file = ARGS[1]
config_dict = YAML.load_file(config_file)
config = AtmosCoveragePerfConfig(config_dict)
job_id = config.parsed_args["job_id"]
integrator = CA.get_integrator(config)

# The callbacks flame graph is very expensive, so only do 2 steps.
@info "running step"

import SciMLBase
SciMLBase.step!(integrator) # compile first
CA.call_all_callbacks!(integrator) # compile callbacks
import Profile, ProfileCanvas
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
allocs_limit["flame_perf_target"] = 4656
allocs_limit["flame_perf_target_tracers"] = 204288
allocs_limit["flame_perf_target_edmfx"] = 253440
allocs_limit["flame_perf_diagnostics"] = 3016328
allocs_limit["flame_perf_target_diagnostic_edmfx"] = 920960
allocs_limit["flame_sphere_baroclinic_wave_rhoe_equilmoist_expvdiff"] =
    67443909648
allocs_limit["flame_perf_target_threaded"] = 5857808
allocs_limit["flame_perf_target_callbacks"] = 46407936
allocs_limit["flame_perf_gw"] = 4844555632

if allocs < allocs_limit[job_id] * buffer
    @info "TODO: lower `allocs_limit[$job_id]` to: $(allocs)"
end
Δallocs = allocs / allocs_limit[job_id]
@info "Allocation change (allocs/allocs_limit): $Δallocs"

# https://github.com/CliMA/ClimaAtmos.jl/issues/827
@testset "Allocations limit" begin
    @test allocs ≤ allocs_limit[job_id] * buffer
end

import ClimaComms
if config.comms_ctx isa ClimaComms.SingletonCommsContext && !isinteractive()
    include(joinpath(pkgdir(CA), "perf", "jet_report_nfailures.jl"))
end
