redirect_stderr(IOContext(stderr, :stacktrace_types_limited => Ref(false)))
import YAML

"""
    TargetJobConfig(target_job)

Creates a full model configuration from the given target job.
"""
TargetJobConfig(target_job) =
    CA.AtmosConfig(CA.config_from_target_job(target_job))

import ClimaTimeSteppers as CTS
get_W(i::CTS.DistributedODEIntegrator) =
    hasproperty(i.cache, :W) ? i.cache.W : i.cache.newtons_method_cache.j
get_W(i::CTS.RosenbrockAlgorithm) = i.cache.W
get_W(i) = i.cache.W
f_args(i, f::CTS.ForwardEulerODEFunction) = (copy(i.u), i.u, i.p, i.t, i.dt)
f_args(i, f) = (similar(i.u), i.u, i.p, i.t)

r_args(i, f::CTS.ForwardEulerODEFunction) =
    (copy(i.u), copy(i.u), i.u, i.p, i.t, i.dt)
r_args(i, f) = (similar(i.u), similar(i.u), i.u, i.p, i.t)

implicit_args(i::CTS.DistributedODEIntegrator) = f_args(i, i.sol.prob.f.T_imp!)
implicit_args(i) = f_args(i, i.f.f1)
remaining_args(i::CTS.DistributedODEIntegrator) =
    r_args(i, i.sol.prob.f.T_exp_T_lim!)
remaining_args(i) = r_args(i, i.f.f2)
wfact_fun(i) = implicit_fun(i).Wfact
implicit_fun(i::CTS.DistributedODEIntegrator) = i.sol.prob.f.T_imp!
implicit_fun(i) = i.sol.prob.f.f1
remaining_fun(i::CTS.DistributedODEIntegrator) = i.sol.prob.f.T_exp_T_lim!
remaining_fun(i) = i.sol.prob.f.f2
