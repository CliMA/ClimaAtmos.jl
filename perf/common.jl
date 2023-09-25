import ClimaAtmos as CA
import YAML
"""
    AtmosCoveragePerfConfig()
    AtmosCoveragePerfConfig(; config_dict)

Creates a model configuration that covers many physics components.

The configuration precedence is as follows:
1. Configuration from the given config file/dict (highest precedence)
2. Default perf configuration (to increase coverage)
3. Default configuration (lowest precedence)
"""
function AtmosCoveragePerfConfig(; config_dict = CA.default_config_dict())
    perf_defaults = joinpath(
        dirname(@__FILE__),
        "..",
        "config",
        "default_configs",
        "default_perf.yml",
    )
    perf_default_config = YAML.load_file(perf_defaults)
    config_dict = merge(perf_default_config, config_dict)
    return CA.AtmosConfig(; config_dict)
end

"""
    AtmosTargetConfig()
    AtmosTargetConfig(; target_job)

Creates an AtmosConfig given the job ID (target_job).
"""
AtmosTargetConfig(; target_job) =
    CA.AtmosConfig(; config_dict = CA.config_from_target_job(target_job))
