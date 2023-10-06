import YAML

"""
    default_config_dict()
    default_config_dict(config_path)

Loads the default configuration from files into a Dict for use in AtmosConfig().
"""
function default_config_dict(
    config_path = joinpath(
        dirname(@__FILE__),
        "..",
        "..",
        "config",
        "default_configs",
    ),
)
    default_config_file = joinpath(config_path, "default_config.yml")
    config = YAML.load_file(default_config_file)
    edmf_config_file = joinpath(config_path, "default_edmf_config.yml")
    edmf_config = YAML.load_file(edmf_config_file)
    # Combine base config with EDMF config - Don't allow duplicate entries.
    !isempty(intersect(keys(config), keys(edmf_config))) &&
        error("Duplicate keys in default config and EDMF config.")
    merge!(config, edmf_config)
    # Strip out help messages
    for (k, v) in config
        config[k] = v["value"]
    end
    return config
end

"""
    config_from_target_job(target_job)

Given a job id string, returns the configuration for that job.
Does not include the default configuration dictionary.
"""
function config_from_target_job(target_job)
    for (job_id, config) in configs_per_job_id()
        if job_id == target_job
            return config
        end
    end
    error("Could not find job with id $target_job")
end

"""
    override_default_config(override_config)

Takes in a Dict, vector of Dicts or filepaths and returns a Dict with the
default configuration overridden by the given dicts or parsed YAML files.
"""
override_default_config(config_files::AbstractString) =
    override_default_config(YAML.load_file(config_files))

override_default_config(config_files::Vector{<:AbstractString}) =
    override_default_config(YAML.load_file.(config_files))

override_default_config(config_dicts::Vector{<:AbstractDict}) =
    override_default_config(merge(config_dicts...))

function override_default_config(::Nothing)
    @info "Using default configuration"
    return default_config_dict()
end

function override_default_config(config_dict::AbstractDict;)
    default_config = default_config_dict()
    config = deepcopy(default_config)
    # Allow unused keys in config_dict for coupler
    for k in intersect(keys(config_dict), keys(default_config))
        default_type = typeof(default_config[k])
        v = config_dict[k]
        config[k] = isnothing(default_config[k]) ? v : default_type(v)
    end

    # The "diagnostics" entry is a more complex type that doesn't fit the schema described in
    # the previous lines. So, we manually add it.
    if haskey(config_dict, "diagnostics")
        config["diagnostics"] = config_dict["diagnostics"]
    end

    config == default_config && @info "Using default configuration"

    return config
end

"""
    non_default_config_entries(config)

Given a configuration Dict, returns a Dict of the non-default values.
"""
function non_default_config_entries(config, defaults = default_config_dict())
    non_defaults = Dict()
    for k in keys(config)
        defaults[k] == config[k] && continue
        non_defaults[k] = config[k]
    end
    return non_defaults
end

"""
    configs_per_job_id(directory)

Walks a directory and reads all of the yaml files that are used to configure the driver,
then parses them into a vector of dictionaries. Does not include the default configuration.
To filter only configurations with a certain key/value pair,
use the `filter_name` keyword argument with a Pair.
"""
function configs_per_job_id(
    directory::AbstractString = joinpath(
        dirname(@__FILE__),
        "..",
        "..",
        "config",
    ),
    filter_name = nothing,
)
    cmds = Dict()
    for (root, _, files) in walkdir(directory)
        for f in files
            file = joinpath(root, f)
            !endswith(file, ".yml") && continue
            occursin("default_configs", file) && continue
            config = YAML.load_file(file)
            cmds[config["job_id"]] = config
        end
    end
    if !isnothing(filter_name)
        (key, value) = filter_name
        filter!(cmds) do (job_id, dict)
            get(dict, key, "") == value
        end
    end
    return cmds
end

"""
    job_id_from_config(config)

Returns a unique name (`String`) given
`config`, the Dict containing the model configuration
"""
function job_id_from_config(config, defaults = default_config_dict())
    # Use only keys from the default ArgParseSettings
    _config = deepcopy(config)
    s = ""
    warn = false
    for k in keys(_config)
        # Skip defaults to alleviate verbose names
        !haskey(defaults, k) && continue
        defaults[k] == _config[k] && continue

        if _config[k] isa String
            # We don't need keys if the value is a string
            # (alleviate verbose names)
            s *= _config[k]
        elseif _config[k] isa Int
            s *= k * "_" * string(_config[k])
        elseif _config[k] isa AbstractFloat
            warn = true
        else
            s *= k * "_" * string(_config[k])
        end
        s *= "_"
    end
    s = replace(s, "/" => "_")
    s = strip(s, '_')
    warn && @warn "Truncated job ID:$s may not be unique due to use of Real"
    return s
end
