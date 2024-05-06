redirect_stderr(IOContext(stderr, :stacktrace_types_limited => Ref(false)))
import YAML
import ArgParse

function parse_commandline()
    s = ArgParse.ArgParseSettings()
    ArgParse.@add_arg_table! s begin
        "--target_config"
        help = "Target job configuration [e.g., ``]"
        arg_type = String
    end
    parsed_args = ArgParse.parse_args(ARGS, s)
    return (s, parsed_args)
end

"""
    AtmosCoveragePerfConfig()
    AtmosCoveragePerfConfig(; config_dict)

Creates a model configuration that covers many physics components.
The configuration precedence is as follows:
    1. Configuration from the given config file/dict (highest precendence)
    2. Default perf configuration (to increase coverage)
    3. Default configuration (lowest precedence)
"""
function AtmosCoveragePerfConfig(config_dict = Dict())
    perf_default_config = perf_config_dict()
    config_dict = merge(perf_default_config, config_dict)
    return CA.AtmosConfig(config_dict)
end

"""
    TargetConfig(target_config)

Creates a full model configuration from the given target job.
"""
function TargetConfig(target_config)
    (; config, config_file) = CA.config_from_target_config(target_config)
    CA.AtmosConfig(config; config_file)
end


"""
    perf_config_dict()

Loads the default performance configuration from a file into a Dict.
"""
function perf_config_dict()
    perf_defaults = joinpath(
        dirname(@__FILE__),
        "..",
        "config",
        "default_configs",
        "default_perf.yml",
    )
    return YAML.load_file(perf_defaults)
end

import ClimaTimeSteppers as CTS
get_W(i::CTS.DistributedODEIntegrator) = i.cache.newtons_method_cache.j
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
