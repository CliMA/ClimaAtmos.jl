ca_dir = joinpath(dirname(@__DIR__));
include(joinpath(ca_dir, "examples", "hybrid", "cli_options.jl"));

ENV["CI_PERF_SKIP_RUN"] = true # we only need haskey(ENV, "CI_PERF_SKIP_RUN") == true

filename = joinpath(ca_dir, "examples", "hybrid", "driver.jl")

try # capture integrator
    include(filename)
catch err
    if err.error !== :exit_profile
        rethrow(err.error)
    end
end

import OrdinaryDiffEq as ODE
ODE.step!(integrator) # compile first
X = similar(integrator.u)

(; cache, u, p, dt, t) = integrator
(; W) = cache

include("benchmark_utils.jl")

import OrderedCollections
trials = OrderedCollections.OrderedDict()
#! format: off
trials["Wfact"] = get_trial(integrator.f.f1.Wfact, (W, u, p, dt, t), "Wfact");
trials["linsolve"] = get_trial(integrator.cache.linsolve, (X, W, u), "linsolve");
trials["implicit_tendency!"] = get_trial(integrator.f.f1, (X, u, p, t), "implicit_tendency!");
trials["remaining_tendency!"] = get_trial(integrator.f.f2, (X, u, p, t), "remaining_tendency!");
trials["default_remaining_tendency!"] = get_trial(p.default_remaining_tendency!, (X, u, p, t), "default_remaining_tendency!");
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
