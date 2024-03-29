redirect_stderr(IOContext(stderr, :stacktrace_types_limited => Ref(false)))
import Random
Random.seed!(1234)
import ClimaAtmos as CA

include("common.jl")

length(ARGS) != 1 && error("Usage: flame.jl <config_file>")
config_file = ARGS[1]
config_dict = YAML.load_file(config_file)
config = AtmosCoveragePerfConfig(config_dict)
job_id = config.parsed_args["job_id"]
simulation = CA.get_simulation(config)
(; integrator) = simulation

# The callbacks flame graph is very expensive, so only do 2 steps.
@info "running step"

import SciMLBase
SciMLBase.step!(integrator) # compile first
SciMLBase.step!(integrator) # compile print_walltime_estimate, which skips the first step to avoid timing compilation
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

allocs_limit = Dict()
allocs_limit["flame_perf_target"] = 51_600
allocs_limit["flame_perf_target_tracers"] = 81_576
allocs_limit["flame_perf_target_edmfx"] = 86_608
allocs_limit["flame_perf_diagnostics"] = 10_876_900
allocs_limit["flame_perf_target_diagnostic_edmfx"] = 86_608
allocs_limit["flame_sphere_baroclinic_wave_rhoe_equilmoist_expvdiff"] =
    4_018_252_656
allocs_limit["flame_perf_target_frierson"] = 4_015_547_056
allocs_limit["flame_perf_target_threaded"] = 1_276_864
allocs_limit["flame_perf_target_callbacks"] = 172_032
allocs_limit["flame_perf_gw"] = 3_268_961_856
allocs_limit["flame_perf_target_prognostic_edmfx_aquaplanet"] = 2_770_296
allocs_limit["flame_gpu_implicit_barowave_moist"] = 199_936
# Ideally, we would like to track all the allocations, but this becomes too
# expensive there is too many of them. Here, we set the default sample rate to
# 1, but lower it to a smaller value when we expect the job to produce lots of
# allocations. Empirically, we find that on the Caltech cluster the limit is 10
# M of allocation.
max_allocs_for_full_sampling = 10e6

# For jobs that we don't track, we set an expected_allocs of
# max_allocs_for_full_sampling, which leads to a sampling rate of 1
expected_allocs = get(allocs_limit, job_id, max_allocs_for_full_sampling)
sampling_rate = expected_allocs <= max_allocs_for_full_sampling ? 1 : 0.01

# Some jobs are problematic (the ones with Krylov mostly)
# https://github.com/pfitzseb/ProfileCanvas.jl/issues/34
if job_id in (
    "flame_sphere_baroclinic_wave_rhoe_equilmoist_expvdiff",
    "flame_perf_target_frierson",
)
    sampling_rate = 0.001
end

# use new allocation profiler
@info "collecting allocations with sampling rate $sampling_rate"
Profile.Allocs.clear()
Profile.Allocs.@profile sample_rate = sampling_rate SciMLBase.step!(integrator)
results = Profile.Allocs.fetch()
Profile.Allocs.clear()
profile = ProfileCanvas.view_allocs(results)
ProfileCanvas.html_file(joinpath(output_dir, "allocs.html"), profile)

# We're grouping allocation tests here for convenience.

@info "testing allocations"
using Test
# Threaded/gpu allocations are not deterministic, so let's add a buffer
# TODO: remove buffer, and threaded tests, when
#       threaded/unthreaded functions are unified
buffer = if any(x -> occursin(x, job_id), ("threaded", "gpu"))
    1.8
else
    1.1
end


## old allocation profiler (TODO: remove this)
allocs = @allocated SciMLBase.step!(integrator)
@timev SciMLBase.step!(integrator)
@info "`allocs ($job_id)`: $(allocs)"

if allocs < allocs_limit[job_id] * buffer
    @info "TODO: lower `allocs_limit[$job_id]` to: $(allocs)"
end
Δallocs = allocs / allocs_limit[job_id]
@info "Allocation change (allocs/allocs_limit): $Δallocs"

# https://github.com/CliMA/ClimaAtmos.jl/issues/827
@testset "Allocations limit" begin
    @test 0.5 * allocs_limit[job_id] * buffer <=
          allocs ≤
          allocs_limit[job_id] * buffer
end

import ClimaComms
if config.comms_ctx isa ClimaComms.SingletonCommsContext && !isinteractive()
    include(joinpath(pkgdir(CA), "perf", "jet_report_nfailures.jl"))
end
