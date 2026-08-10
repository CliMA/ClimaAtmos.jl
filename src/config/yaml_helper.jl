import YAML
import ClimaUtilities.ClimaArtifacts: @clima_artifact
import LazyArtifacts

const config_path = joinpath(dirname(@__FILE__), "..", "..", "config")

const default_config_file =
    joinpath(config_path, "default_configs", "default_config.yml")

"""
    strip_help_message(v)
    strip_help_messages(d)

Reduce schema-style config entries to their bare values.

Entries in `default_config.yml` are `Dict`s of the form `(help = ..., value = ...)`;
`strip_help_message` returns the `"value"` field of such an entry and passes anything
else through unchanged. `strip_help_messages` applies this to every entry of the config
`Dict` `d`.
"""
strip_help_message(v::Dict) = v["value"]
strip_help_message(v) = v
strip_help_messages(d) =
    Dict(map(k -> Pair(k, strip_help_message(d[k])), collect(keys(d)))...)

"""
    load_yaml_file(f)

Parse the YAML file `f` into a `Dict`, erroring if the file is empty or missing.
"""
function load_yaml_file(f)
    filesize(f) == 0 && error("File $f is empty or missing.")
    return YAML.load_file(f)
end

"""
    default_config_dict(config_file = default_config_file)

Load the default configuration into a `Dict` of `key => value` pairs, with the schema's
help messages stripped out.
"""
function default_config_dict(config_file = default_config_file)
    config = load_yaml_file(config_file)
    return strip_help_messages(config)
end

ContainerType(T) = Union{Tuple{<:T, Vararg{T}}, Vector{<:T}}

"""
    override_default_config(config_dict::AbstractDict)
    override_default_config(config_file::AbstractString)
    override_default_config(config_files)
    override_default_config(config_dicts)
    override_default_config(::Nothing)

Return the default configuration with the given overrides applied.

The argument may be a `Dict`, a YAML file path, or a tuple/vector of either (merged in
order, later entries winning); `nothing` returns the defaults unchanged. Only keys that
exist in `default_config.yml` are overridden, and each value is coerced to the type of
the corresponding default via `coerce_to_default`. Keys absent from the schema (other
than `job_id`) are reported: an error if `strict_config` is `true`, a warning otherwise.
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
default in `default_config.yml`. Used by `override_default_config` to
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

```julia
coerce_to_default(Bool, "true")       # 2 → true
coerce_to_default(Int, "42")          # 3 → 42
coerce_to_default(Float64, "3.14")    # 4 → 3.14
coerce_to_default(Float64, 1)         # 5 → 1.0 (Int → Float)
coerce_to_default(Bool, true)         # 1 → true (identity)
coerce_to_default(Bool, "yes")        # 2 → throws ArgumentError
```

# Notes

Keys whose schema default is `nothing` or that appear in `EXCEPTED_KEYS` bypass
coercion entirely and pass through unchanged.
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
        if isnothing(default_config[k]) || isnothing(v) || k in EXCEPTED_KEYS
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
        msg = "The configuration passed to ClimaAtmos contains unused keys: $(join(unused_keys, ", "))"
        config["strict_config"] ? error(msg) : @warn msg
    end

    config == default_config && @info "Using default configuration"

    return config
end

"""
    non_default_config_entries(config, defaults = default_config_dict())

Return a `Dict` with the entries of `config` whose values differ from `defaults`.
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
    load_all_configs(with_pair = nothing)

Load every `.yml` file under `config/` except the default config, keyed by job id.

Files are located by walking `config_path`; keys come from `job_id_from_config_file`.
When `with_pair` is a `key => value` pair, only configs that set `key` to `value` are
kept.
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

"""
    is_unique_basename(file, bname = first(splitext(basename(file))))

Return `true` if no other configuration file under `config/` shares the base name
`bname`. Called from `job_id_from_config_file`.

# Notes

`bname` has its extension stripped while the file names it is compared against do not,
so the comparison currently never matches and the result is always `true`.
"""
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

"""
    job_id_from_config_file(config_file::String)
    job_id_from_config_files(config_files)

Derive a job id from configuration file names.

For a single file the id is its base name without extension, or, if that base name is
not unique within `config/`, the full path with the path separator replaced by `_`.
For several files the individual ids are joined with `_`.
"""
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
    maybe_resolve_and_acquire_artifacts(input, context)

Resolve a string of the form `artifact"name"/something/else` to a local path,
downloading the artifact if it is not already available.

Inputs that are not strings, or strings that do not match the artifact pattern, are
returned unchanged. `context` is the `ClimaComms` context used to download the artifact
on one rank only.
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
    config_with_resolved_and_acquired_artifacts(config::AbstractDict, context)

Return a copy of `config` in which every value of the form `artifact"name"/some/path`
has been replaced by the resolved (and, if needed, downloaded) artifact path.

Values that are not artifact strings pass through unchanged; see
`maybe_resolve_and_acquire_artifacts`.
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

"""
    config_summary(io::IO, config_files)

Print one configuration file name per line to `io`, for logging by the `AtmosConfig`
constructor.
"""
function config_summary(io::IO, config_files)
    print(io, '\n')
    for x in config_files
        println(io, "   $x")
    end
end
