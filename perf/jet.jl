#=
# to get a target configuration:
julia --project=perf
import ClimaAtmos as CA
using Revise; include(joinpath(pkgdir(CA), "perf", "common.jl"))
config = TargetJobConfig("gpu_prognostic_edmfx_aquaplanet")

include(joinpath(pkgdir(CA), "perf", "jet.jl"))
=#
redirect_stderr(IOContext(stderr, :stacktrace_types_limited => Ref(false)))
import Random
import HDF5, NCDatasets, CUDA
Random.seed!(1234)
import ClimaAtmos as CA

include("common.jl")

if !(@isdefined config)
    (; config_file, job_id) = CA.commandline_kwargs()
    config = CA.AtmosConfig(config_file; job_id)
end

import JET

function jet_test(f, args)
    f(args...) # compile first
    JET.@test_opt ignored_modules = (HDF5, CUDA, NCDatasets) f(args...)
end

simulation = CA.get_simulation(config)
(; integrator) = simulation

import SciMLBase
W = get_W(integrator);
(; u, p, dt, t) = integrator;

# jet_test(wfact_fun(integrator), (W, u, p, dt, t))
# jet_test(LA.ldiv!, (X, W, u))
# jet_test(implicit_fun(integrator), implicit_args(integrator))
# jet_test(remaining_fun(integrator), remaining_args(integrator))
# jet_test(CA.additional_tendency!, (X, u, p, t))
# jet_test(CA.hyperdiffusion_tendency!, remaining_args(integrator))
# jet_test(CA.dss!, (u, p, t))
# jet_test(CA.set_precomputed_quantities!, (u, p, t))
jet_test(SciMLBase.step!, (integrator,))
