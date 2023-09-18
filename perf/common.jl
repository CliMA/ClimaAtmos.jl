import YAML

"""
    AtmosCoveragePerfConfig()
    AtmosCoveragePerfConfig(; config_dict)
Creates a model configuration that covers many physics components.
The configuration precedence is as follows:
1. Default perf configuration (to increase coverage)
2. Configuration from the given config file/dict
3. Default configuration (lowest precedence)

There is an exception for the job `flame_perf_target_edmf`, which requires that 
the performance configuration be overridden for certain components.
"""
function AtmosCoveragePerfConfig(; config_dict = CA.default_config_dict())
    perf_default_config = perf_config_dict()
    config_dict = merge(config_dict, perf_default_config)
    if get(config_dict, "job_id", "") == "flame_perf_target_edmf"
        config_dict["precip_model"] = nothing
        config_dict["rad"] = nothing
        config_dict["apply_limiter"] = false
    end
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
