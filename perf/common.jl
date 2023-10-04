import YAML

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
    TargetJobConfig(target_job)

Creates a full model configuration from the given target job.
"""
TargetJobConfig(target_job) =
    CA.AtmosConfig(CA.config_from_target_job(target_job))


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
