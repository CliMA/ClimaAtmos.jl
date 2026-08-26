"""
Parameter composition for a SOCRATES run.

Every source of parameter overrides — the case's prescribed droplet number, a calibrated
`parameters.toml`, an EKI-sampled ensemble member's TOML, an inline tweak from the REPL — enters
through one ordered list. Later sources win. There is no second path and no temporary file is
written for any of it.
"""

using ClimaAtmos: ClimaAtmos as CA
using ClimaParams: ClimaParams as CP
using TOML: TOML

"""
Base parameter TOML for the SOCRATES column: ClimaAtmos's prognostic EDMFX + 1-moment
microphysics set. Overrides are layered on top of it.
"""
default_base_toml() = joinpath(pkgdir(CA), "toml", "prognostic_edmfx_1M.toml")

"""
A single parameter override source: a path to a TOML file, or an already-parsed
`name => Dict("value" => …, "type" => …)` dictionary.
"""
const ParamSource = Union{AbstractString, AbstractDict}

# One source -> a plain override dictionary. Files are parsed; dictionaries pass through with
# their keys normalized to `String` so sources merge regardless of how the caller keyed them.
function _override_dict(source::AbstractString)
    isfile(source) || error("Parameter TOML not found: $source")
    endswith(source, ".toml") || error("Parameter source is not a .toml file: $source")
    return TOML.parsefile(source)
end
_override_dict(source::AbstractDict) =
    Dict{String, Any}(string(k) => v for (k, v) in source)

_source_list(params::ParamSource) = ParamSource[params]
_source_list(params::AbstractVector) = ParamSource[p for p in params]
_source_list(::Nothing) = ParamSource[]

"""
    n_ccn_override(case)

The case's prescribed cloud droplet number as a parameter-override dictionary, in the
`ClimaParams` entry format. Layered below any caller-supplied source, so calibrating or
overriding `prescribed_cloud_droplet_number_concentration` works.
"""
n_ccn_override(case::SocratesCase) = Dict{String, Any}(
    "prescribed_cloud_droplet_number_concentration" =>
        Dict{String, Any}("value" => n_ccn(case), "type" => "float"),
)

"""
    merge_param_sources(sources)

Merge parameter override `sources` left to right into one override dictionary; later sources
overwrite earlier ones. Accepts TOML paths and dictionaries interchangeably.

This generalizes `ClimaParams.merge_toml_files`, which takes paths only, so that in-memory
overrides need no temporary file.
"""
function merge_param_sources(sources)
    merged = Dict{String, Any}()
    for source in sources
        d = _override_dict(source)
        for (name, entry) in d
            if haskey(merged, name)
                @debug "Parameter `$name` overridden by a later source" source
            end
            merged[name] = entry
        end
    end
    return merged
end

"""
    socrates_toml_dict(FT, case; params, base_toml)

The `ClimaParams.ParamDict{FT}` for `case`: the `base_toml` defaults, then the case's
prescribed droplet number, then each entry of `params` in order.

`params` accepts a TOML path, an override dictionary, or a vector mixing both.
"""
function socrates_toml_dict(
    ::Type{FT},
    case::SocratesCase;
    params = nothing,
    base_toml::AbstractString = default_base_toml(),
) where {FT <: AbstractFloat}
    sources = ParamSource[base_toml, n_ccn_override(case), _source_list(params)...]
    return CP.create_toml_dict(FT; override_file = merge_param_sources(sources))
end

"""
    socrates_params(FT, case; params, base_toml, microphysics_model)

`ClimaAtmosParameters` for `case` in float type `FT`, with parameters composed as described in
[`socrates_toml_dict`](@ref).

`microphysics_model` must match the model's own microphysics choice; passing it lets
`ClimaAtmosParameters` drop the parameter sets the model cannot use.

# Examples
```julia
socrates_params(Float64, case)                                  # case defaults only
socrates_params(Float64, case; params = "calibrated.toml")       # a calibrated set
socrates_params(Float64, case; params = [                        # ordered, later wins
    "calibrated.toml",
    Dict("rain_autoconversion_timescale" => Dict("value" => 500.0, "type" => "float")),
])
```
"""
function socrates_params(
    ::Type{FT},
    case::SocratesCase;
    params = nothing,
    base_toml::AbstractString = default_base_toml(),
    microphysics_model = CA.NonEquilibriumMicrophysics1M(),
) where {FT <: AbstractFloat}
    toml_dict = socrates_toml_dict(FT, case; params, base_toml)
    return CA.ClimaAtmosParameters(toml_dict; microphysics_model)
end