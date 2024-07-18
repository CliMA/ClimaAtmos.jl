import YAML

const config_path = joinpath(dirname(@__FILE__), "..", "..", "config")

const default_config_file =
    joinpath(config_path, "default_configs", "default_config.yml")

strip_help_message(v::Dict) = v["value"]
strip_help_message(v) = v
strip_help_messages(d) =
    Dict(map(k -> Pair(k, strip_help_message(d[k])), collect(keys(d)))...)

function load_yaml_file(f)
    filesize(f) == 0 && error("File $f is empty or missing.")
    return YAML.load_file(f)
end

"""
    default_config_dict()
    default_config_dict(config_path)

Loads the default configuration from files into a Dict for use in AtmosConfig().
"""
function default_config_dict(config_file = default_config_file)
    config = load_yaml_file(config_file)
    return strip_help_messages(config)
end

"""
    config_from_target_job(target_job)

Given a job id string, returns the configuration for that job.
Does not include the default configuration dictionary.
"""
function config_from_target_job(target_job)
    for (job_id, config) in configs_per_config_id()
        if job_id == target_job
            return config
        end
    end
    error("Could not find job with id $target_job")
end

ContainerType(T) = Union{Tuple{<:T, Vararg{T}}, Vector{<:T}}

"""
    override_default_config(override_config)

Takes in a Dict, vector of Dicts or filepaths and returns a Dict with the
default configuration overridden by the given dicts or parsed YAML files.
"""
override_default_config(config_files::AbstractString) =
    override_default_config(load_yaml_file(config_files))

override_default_config(config_files::ContainerType(AbstractString)) =
    override_default_config(load_yaml_file.(config_files))

override_default_config(config_dicts::ContainerType(AbstractDict)) =
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

        # Attempt to convert user value `v` to the same type as
        # the default. If that fails, throw an informative warning.
        config[k] = try
            isnothing(default_config[k]) ? v : convert(default_type, v)
        catch e
            # A failed conversion should result in a MethodError
            e isa MethodError || rethrow(e)
            @warn """Failed to convert `config_dict["$k"] = $v` to default type $default_type, keeping original value"""
            v
        end
    end

    unused_keys = filter(k -> !haskey(default_config, k), keys(config_dict))
    if !isempty(unused_keys)
        @warn "The configuration passed to ClimaAtmos contains unused keys: $(join(unused_keys, ", "))"
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
    configs_per_config_id(directory)

Walks a directory and reads all of the yaml files that are used to configure the driver,
then parses them into a vector of dictionaries. Does not include the default configuration.
To filter only configurations with a certain key/value pair,
use the `filter_name` keyword argument with a Pair.
"""
function configs_per_config_id(
    directory::AbstractString = config_path,
    filter_name = nothing,
)
    cmds = Dict()
    for (root, _, files) in walkdir(directory)
        for f in files
            file = joinpath(root, f)
            !endswith(file, ".yml") && continue
            occursin("default_configs", file) && continue
            config = load_yaml_file(file)
            name = config_id_from_config_file(file)
            cmds[name] = (; config, config_file = file)
        end
    end
    if !isnothing(filter_name)
        (key, value) = filter_name
        filter!(cmds) do (config_id, nt)
            get(nt.config, key, "") == value
        end
    end
    return cmds
end

function is_unique_basename(file, bname = first(splitext(basename(file))))
    is_unique = true
    for (root, _, files) in walkdir(config_path)
        for f in files
            file = joinpath(root, f)
            if basename(f) == bname
                is_unique = false
            end
        end
    end
    return is_unique
end

function config_id_from_config_file(config_file::String)
    @assert isfile(config_file)
    bname = first(splitext(basename(config_file)))
    if is_unique_basename(config_file, bname)
        return bname
    else
        return replace(config_file, path_sep => "_")
    end
end

# Can we do better?
config_id_from_config_files(config_files::Union{Tuple, Vector}) =
    join(map(x -> config_id_from_config_file(x), config_files), "_")
