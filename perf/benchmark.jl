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

import ClimaAtmos as CA
import OrdinaryDiffEq as ODE
import ClimaTimeSteppers as CTS
ODE.step!(integrator) # compile first

(; sol, u, p, dt, t) = integrator

get_W(i::CTS.DistributedODEIntegrator) = i.cache.newtons_method_cache.j
get_W(i) = i.cache.W
f_args(i, f::CTS.ForwardEulerODEFunction) = (copy(i.u), i.u, i.p, i.t, i.dt)
f_args(i, f) = (similar(i.u), i.u, i.p, i.t)
implicit_args(i::CTS.DistributedODEIntegrator) = f_args(i, i.sol.prob.f.T_imp!)
implicit_args(i) = f_args(i, i.f.f1)
remaining_args(i::CTS.DistributedODEIntegrator) = f_args(i, i.sol.prob.f.T_exp!)
remaining_args(i) = f_args(i, i.f.f2)
wfact_fun(i) = implicit_fun(i).Wfact
implicit_fun(i::CTS.DistributedODEIntegrator) = i.sol.prob.f.T_imp!
implicit_fun(i) = i.sol.prob.f.f1
remaining_fun(i::CTS.DistributedODEIntegrator) = i.sol.prob.f.T_exp!
remaining_fun(i) = i.sol.prob.f.f2

W = get_W(integrator)
X = similar(u)

include("benchmark_utils.jl")

import OrderedCollections
import LinearAlgebra as LA
trials = OrderedCollections.OrderedDict()
#! format: off
trials["Wfact"] = get_trial(wfact_fun(integrator), (W, u, p, dt, t), "Wfact");
trials["linsolve"] = get_trial(LA.ldiv!, (X, W, u), "linsolve");
trials["implicit_tendency!"] = get_trial(implicit_fun(integrator), implicit_args(integrator), "implicit_tendency!");
trials["remaining_tendency!"] = get_trial(remaining_fun(integrator), remaining_args(integrator), "remaining_tendency!");
trials["additional_tendency!"] = get_trial(additional_tendency!, (X, u, p, t), "additional_tendency!");
trials["hyperdiffusion_tendency!"] = get_trial(CA.hyperdiffusion_tendency!, (X, u, p, t), "hyperdiffusion_tendency!");
trials["step!"] = get_trial(ODE.step!, (integrator, ), "step!");
#! format: on

table_summary = OrderedCollections.OrderedDict()
for k in keys(trials)
    table_summary[k] = get_summary(trials[k])
end
tabulate_summary(table_summary)

if get(ENV, "BUILDKITE", "") == "true"
    # Export table_summary
    import JSON
    job_id = parsed_args["job_id"]
    path = ca_dir
    open(joinpath(path, "perf_benchmark_$job_id.json"), "w") do io
        JSON.print(io, table_summary)
    end
end
