import YAML
import ClimaUtilities.ClimaArtifacts: @clima_artifact
import LazyArtifacts

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
    return default_config_dict()
end

# Keys that bypass scalar coercion: `named String | nothing` unions
# (hyperdiff has a String default but accepts `~` to disable) and structured
# values (`diagnostics` is a list of dicts).
const EXCEPTED_KEYS = Set([
    "hyperdiff",
    "diagnostics",
])

"""
    coerce_to_default(::Type{T}, v) -> T

Coerce a user-supplied YAML value `v` to the type `T` of the corresponding
default in `default_config.yml`. Used by [`override_default_config`](@ref) to
enforce a single canonical type per key.

Dispatch order, most specific first:

1. `coerce_to_default(::Type{T}, v::T) = v` — same type, identity. The
   common path; most YAML values already load with the expected type.
2. `coerce_to_default(::Type{Bool}, v::AbstractString)` — `"true"`/`"false"`
   → `true`/`false` via `parse(Bool, v)`. Anything else throws
   `ArgumentError`.
3. `coerce_to_default(::Type{<:Integer}, v::AbstractString)` — `"42"` → `42`.
4. `coerce_to_default(::Type{<:AbstractFloat}, v::AbstractString)` — `"3.14"`
   → `3.14`. Note `"42"` also parses as a `Float64` here if the schema
   default is float.
5. `coerce_to_default(::Type{T}, v) = convert(T, v)` — fallback. Catches
   things like `Int → Float64` (schema default `1.0`, user wrote `1`).
   For unrelated types this throws `MethodError`.

# Examples

```julia-repl
julia> coerce_to_default(Bool, "true")       # 2
true

julia> coerce_to_default(Int, "42")          # 3
42

julia> coerce_to_default(Float64, "3.14")    # 4
3.14

julia> coerce_to_default(Float64, 1)         # 5 (Int → Float)
1.0

julia> coerce_to_default(Bool, true)         # 1 (identity)
true

julia> coerce_to_default(Bool, "yes")        # 2, throws
ERROR: ArgumentError: invalid value for Bool: "yes"
```

Keys whose schema default is `nothing` or that appear in [`EXCEPTED_KEYS`]
bypass coercion entirely and pass through unchanged.
"""
coerce_to_default(::Type{T}, v::T) where {T} = v
coerce_to_default(::Type{Bool}, v::AbstractString) = parse(Bool, v)
coerce_to_default(::Type{T}, v::AbstractString) where {T <: Integer} =
    parse(T, v)
coerce_to_default(::Type{T}, v::AbstractString) where {T <: AbstractFloat} =
    parse(T, v)
coerce_to_default(::Type{T}, v) where {T} = convert(T, v)

function override_default_config(config_dict::AbstractDict;)
    default_config = default_config_dict()
    config = deepcopy(default_config)
    # Allow unused keys in config_dict for coupler
    for k in intersect(keys(config_dict), keys(default_config))
        v = config_dict[k]
        # `nothing` defaults and excepted keys pass through unchanged;
        # everything else must coerce cleanly to the default's type.
        if isnothing(default_config[k]) || k in EXCEPTED_KEYS
            config[k] = v
        else
            default_type = typeof(default_config[k])
            config[k] = try
                coerce_to_default(default_type, v)
            catch e
                e isa Union{MethodError, ArgumentError} || rethrow(e)
                error(
                    "Cannot coerce `$k = $(repr(v))` to expected type $default_type.",
                )
            end
        end
    end

    # `job_id` is set by the AtmosConfig constructor, not the YAML schema.
    unused_keys = filter(
        k -> !haskey(default_config, k) && k != "job_id",
        keys(config_dict),
    )
    if !isempty(unused_keys)
        @warn "The configuration passed to ClimaAtmos contains unused keys: $(join(unused_keys, ", "))"
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
    load_all_configs([with_pair])

Loads all available configs, excluding the default config, and stores them in a
Dict that maps each one to a unique string. A key/value pair can also be used to
filter out all configs which do not associate that key with the specified value.
"""
function load_all_configs(with_pair = nothing)
    configs = Dict()
    for (root, _, files) in walkdir(config_path)
        for f in files
            file = joinpath(root, f)
            (endswith(file, ".yml") && file != default_config_file) || continue
            config = load_yaml_file(file)
            if !isnothing(with_pair)
                (key, value) = with_pair
                (haskey(config, key) && config[key] == value) || continue
            end
            configs[job_id_from_config_file(file)] = config
        end
    end
    return configs
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

function job_id_from_config_file(config_file::String)
    @assert isfile(config_file)
    bname = first(splitext(basename(config_file)))
    if is_unique_basename(config_file, bname)
        return bname
    else
        return replace(config_file, path_sep => "_")
    end
end

job_id_from_config_files(config_files::Union{Tuple, Vector}) =
    join(map(x -> job_id_from_config_file(x), config_files), "_")

"""
    maybe_resolve_and_acquire_artifacts(input_str::AbstractString, context)

When given a string of the form `artifact"name"/something/else`, resolve the
artifact path and download it (if not already available).

In all the other cases, return the input unchanged.
"""
function maybe_resolve_and_acquire_artifacts(
    input_str::AbstractString,
    context,
)
    matched = match(r"artifact\"([a-zA-Z0-9_]+)\"(\/.*)?", input_str)
    if isnothing(matched)
        return input_str
    else
        artifact_name, other_path = matched
        return joinpath(
            @clima_artifact(string(artifact_name), context),
            lstrip(other_path, '/'),
        )
    end
end

function maybe_resolve_and_acquire_artifacts(
    input,
    _,
)
    return input
end

"""
    config_with_resolved_and_acquired_artifacts(input_str::AbstractString, context)

Substitute strings of the form `artifact"name"/something/else` with the actual
artifact path.
"""
function config_with_resolved_and_acquired_artifacts(
    config::AbstractDict,
    context,
)
    return Dict(
        k => maybe_resolve_and_acquire_artifacts(v, context) for
        (k, v) in config
    )
end

function config_summary(io::IO, config_files)
    print(io, '\n')
    for x in config_files
        println(io, "   $x")
    end
end
