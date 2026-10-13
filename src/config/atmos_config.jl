import ClimaParams as CP

"""
    AtmosConfig{FT, TD, PA, C, CF}

A fully resolved ClimaAtmos configuration, used to build an `AtmosModel` and
`AtmosSimulation`.

# Fields

  - `toml_dict`: the merged ClimaParams TOML parameter dictionary (built by
    `ClimaParams.create_toml_dict` from the files listed under the `toml` config
    key). It holds the physical parameter values (with units and defaults) that
    the model reads, as opposed to `parsed_args`, which holds the run/model
    configuration options. `eltype(toml_dict)` determines the float type `FT`.
  - `parsed_args`: the run configuration as a `key => value` dictionary, obtained
    by overriding `default_config.yml` with the user-supplied configuration.
  - `comms_ctx`: the `ClimaComms` context (device and MPI information).
  - `config_files`: the configuration files that were merged to build this config.
  - `job_id`: a unique identifier for the run, used e.g. for the output directory.
"""
struct AtmosConfig{FT, TD, PA, C, CF}
    toml_dict::TD
    parsed_args::PA
    comms_ctx::C
    config_files::CF
    job_id::String
end

Base.eltype(::AtmosConfig{FT}) where {FT} = FT

TupleOrVector(T) = Union{Tuple{<:T, Vararg{T}}, Vector{<:T}}

# Use short, relative paths, if possible.
function normrelpath(file)
    rfile = normpath(relpath(file, dirname(config_path)))
    return if isfile(rfile) && samefile(rfile, file)
        rfile
    else
        file
    end
end

"""
    maybe_add_default(config_files, default_config_file)

Prepend `default_config_file` to `config_files` unless it is already among them.

Called from the file-based `AtmosConfig` constructor.
"""
function maybe_add_default(config_files, default_config_file)
    return if any(x -> samefile(x, default_config_file), config_files)
        config_files
    else
        (default_config_file, config_files...)
    end
end

"""
    AtmosConfig(config_file::String = default_config_file; job_id = nothing, comms_ctx = nothing)
    AtmosConfig(config_files; job_id = nothing, comms_ctx = nothing)

Build an `AtmosConfig` from one or more YAML configuration files.

Each file is parsed into a `Dict`, and the results are merged (later files override
earlier ones) on top of `default_config.yml`, which is prepended automatically when not
already among `config_files`.

# Keyword Arguments

  - `job_id = nothing`: Run identifier. When `nothing`, it is taken from the `job_id` key
    in the merged configuration (if present), and otherwise derived from the config file
    names.
  - `comms_ctx = nothing`: `ClimaComms` context. When `nothing`, it is inferred from the
    `device` config key (see `get_comms_context`).

# Examples

```julia
import ClimaAtmos as CA
config = CA.AtmosConfig("config/model_configs/held_suarez.yml")
```
"""
AtmosConfig(
    config_file::String = default_config_file;
    job_id = nothing,
    comms_ctx = nothing,
) = AtmosConfig((config_file,); job_id, comms_ctx)

function AtmosConfig(
    config_files::TupleOrVector(String);
    job_id = nothing,
    comms_ctx = nothing,
)

    all_config_files =
        normrelpath.(maybe_add_default(config_files, default_config_file))
    configs = map(all_config_files) do config_file
        strip_help_messages(load_yaml_file(config_file))
    end
    # Resolve `job_id` here to derive it from the user-provided `config_files`.
    job_id = @something(
        job_id,
        get(merge(configs...), "job_id", nothing),
        job_id_from_config_files(config_files),
    )
    return AtmosConfig(
        configs;
        comms_ctx,
        config_files = all_config_files,
        job_id,
    )
end

"""
    AtmosConfig(config::AbstractDict; comms_ctx = nothing, config_files = [default_config_file], job_id = nothing)
    AtmosConfig(configs; comms_ctx = nothing, config_files = [default_config_file], job_id = nothing)

Build an `AtmosConfig` from one or more configuration `Dict`s.

The dicts are merged (later ones override earlier ones), and the result overrides the
defaults from `default_config.yml` (see `override_default_config`). The float type `FT`
is set by the `FLOAT_TYPE` key (`"Float64"` gives `Float64`; anything else gives
`Float32`), the parameter files listed under the `toml` key are merged into the
ClimaParams TOML dictionary, and `artifact"name"` strings in config values are resolved
to local artifact paths.

# Keyword Arguments

  - `comms_ctx = nothing`: `ClimaComms` context. When `nothing`, it is inferred from the
    `device` config key (see `get_comms_context`).
  - `config_files = [default_config_file]`: File names recorded in the resulting config,
    used for logging and for deriving `job_id`; the dicts themselves are the data source.
  - `job_id = nothing`: Run identifier. Resolution order: this keyword if given, then the
    `job_id` key in the merged dicts, then a name derived from `config_files`.
"""
AtmosConfig(configs::AbstractDict; kwargs...) =
    AtmosConfig((configs,); kwargs...)
function AtmosConfig(
    configs::TupleOrVector(AbstractDict);
    comms_ctx = nothing,
    config_files = [default_config_file],
    job_id = nothing,
)
    config_files = map(x -> normrelpath(x), config_files)

    # using config_files = [default_config_file] as a default
    # relies on the fact that override_default_config uses
    # default_config_file.
    config = merge(configs...)
    # Resolve `job_id` (before `override_default_config` drops non-schema keys)
    job_id = @something(
        job_id,
        get(config, "job_id", nothing),
        job_id_from_config_files(config_files),
    )
    comms_ctx = isnothing(comms_ctx) ? get_comms_context(config) : comms_ctx
    config = override_default_config(config)

    FT = config["FLOAT_TYPE"] == "Float64" ? Float64 : Float32
    atmos_toml = map(config["toml"]) do file
        isfile(file) ? file :
        isfile(joinpath(pkgdir(@__MODULE__), file)) ?
        joinpath(pkgdir(@__MODULE__), file) : error("Parameter file $file not found.")
    end
    override_file = CP.merge_toml_files(atmos_toml)
    toml_dict = CP.create_toml_dict(FT; override_file)
    config = config_with_resolved_and_acquired_artifacts(config, comms_ctx)

    isempty(job_id) &&
        @warn "`job_id` is empty and likely not passed to AtmosConfig"

    @info "Making AtmosConfig with config files: $(sprint(config_summary, config_files))"

    C = typeof(comms_ctx)
    TD = typeof(toml_dict)
    PA = typeof(config)
    CF = typeof(config_files)
    return AtmosConfig{FT, TD, PA, C, CF}(
        toml_dict,
        config,
        comms_ctx,
        config_files,
        job_id,
    )
end
