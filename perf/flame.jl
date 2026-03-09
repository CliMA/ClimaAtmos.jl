redirect_stderr(IOContext(stderr, :stacktrace_types_limited => Ref(false)))
import ClimaComms
ClimaComms.@import_required_backends
import Random
Random.seed!(1234)
import ClimaAtmos as CA

include("common.jl")

(; config_file, job_id) = CA.commandline_kwargs()
config = CA.AtmosConfig(config_file; job_id)
simulation = CA.get_simulation(config)
device = ClimaComms.device(config.comms_ctx)
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
allocs_limit["flame_baroclinic_wave_moist_gpu"] = 1_243_048
allocs_limit["flame_default"] = 99_928
allocs_limit["flame_default_1m"] = 663_056
allocs_limit["flame_diagnostics"] = 10_677_144
allocs_limit["flame_aquaplanet_diagedmf"] = 11_644_128
allocs_limit["flame_aquaplanet_progedmf"] = 995_288
allocs_limit["flame_aquaplanet_progedmf_dense_autodiff"] = 195_224
allocs_limit["flame_aquaplanet_progedmf_sparse_autodiff"] = 102_808
allocs_limit["flame_diffusion"] = 138_432
allocs_limit["flame_threaded"] = 2047_736
allocs_limit["flame_callbacks"] = 400_942
allocs_limit["flame_gravity_wave"] = 141_976
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
buffer = if device isa ClimaComms.CPUMultiThreaded
    1.8
elseif device isa ClimaComms.CUDADevice
    5
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
    if device isa ClimaComms.CUDADevice # https://github.com/CliMA/ClimaAtmos.jl/issues/2831
        @test allocs ≤ allocs_limit[job_id] * buffer
    else
        @test 0.25 * allocs_limit[job_id] * buffer <=
              allocs ≤
              allocs_limit[job_id] * buffer
    end
end
