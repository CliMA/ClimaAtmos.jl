ca_dir = joinpath(dirname(@__DIR__));
include(joinpath(ca_dir, "examples", "hybrid", "cli_options.jl"));

ENV["CI_PERF_SKIP_RUN"] = true # we only need haskey(ENV, "CI_PERF_SKIP_RUN") == true

# Uncomment for customizing specific jobs / specs:
# dict = parsed_args_per_job_id(; trigger = "benchmark.jl"); # if job_id uses benchmark.jl
# dict = parsed_args_per_job_id();                           # if job_id uses driver.jl
# parsed_args = dict["sphere_aquaplanet_rhoe_equilmoist_allsky"];

filename = joinpath(ca_dir, "examples", "hybrid", "driver.jl")

try # capture integrator
    include(filename)
catch err
    if err.error !== :exit_profile
        rethrow(err.error)
    end
end

import OrdinaryDiffEq as ODE
import ClimaTimeSteppers as CTS
ODE.step!(integrator) # compile first
X = similar(integrator.u)

(; sol, cache, u, p, dt, t) = integrator
f = sol.prob.f
if integrator isa CTS.DistributedODEIntegrator
    W = cache.newtons_method_cache.j
    linsolve = cache.newtons_method_cache.linsolve!
else
    (; W, linsolve) = cache
end
f1_args =
    f.f1 isa CTS.ForwardEulerODEFunction ? (copy(u), u, p, t, dt) : (X, u, p, t)
f2_args =
    f.f2 isa CTS.ForwardEulerODEFunction ? (copy(u), u, p, t, dt) : (X, u, p, t)

include("benchmark_utils.jl")

import OrderedCollections
trials = OrderedCollections.OrderedDict()
#! format: off
trials["Wfact"] = get_trial(f.f1.Wfact, (W, u, p, dt, t), "Wfact");
trials["linsolve"] = get_trial(linsolve, (X, W, u), "linsolve");
trials["implicit_tendency!"] = get_trial(f.f1, f1_args, "implicit_tendency!");
trials["remaining_tendency!"] = get_trial(f.f2, f2_args, "remaining_tendency!");
trials["additional_tendency!"] = get_trial(additional_tendency!, (X, u, p, t), "additional_tendency!");
trials["hyperdiffusion_tendency!"] = get_trial(hyperdiffusion_tendency!, (X, u, p, t), "hyperdiffusion_tendency!");
trials["step!"] = get_trial(ODE.step!, (integrator, ), "step!");
#! format: on

summary = OrderedCollections.OrderedDict()
for k in keys(trials)
    summary[k] = get_summary(trials[k])
end
tabulate_summary(summary)

if get(ENV, "BUILDKITE", "") == "true"
    # Export summary
    import JSON
    job_id = parsed_args["job_id"]
    path = ca_dir
    open(joinpath(path, "perf_benchmark_$job_id.json"), "w") do io
        JSON.print(io, summary)
    end
end
