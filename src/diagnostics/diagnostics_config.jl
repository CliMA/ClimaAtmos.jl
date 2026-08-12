"""
    DiagnosticsConfig(; default = true, additional = (),
                      interpolation_num_points = nothing, output_at_levels = true)

Specify which diagnostics a simulation produces and how their NetCDF output is shaped.

A single `DiagnosticsConfig` value is passed to
[`AtmosSimulation`](@ref ClimaAtmos.AtmosSimulation) through its `diagnostics` keyword
argument. A simulation produces no diagnostics when `default = false`, `debug_tendency = false`,
and `additional` is empty. The type parameter `A` is the type of the `additional` collection.

# Fields

  - `default::Bool = true`: Whether to include the built-in ClimaAtmos diagnostic set for
    the chosen `AtmosModel`, as returned by `default_diagnostics`.
  - `additional::A = ()`: Extra user-supplied diagnostics. Mixed collections are allowed;
    each entry is normalized by `normalize_diag_entry` and can be:
      + a `ClimaDiagnostics.ScheduledDiagnostic`, used as-is for full control;
      + a `Pair` of short name to options, e.g.
        `"ua" => (; period = "30mins", reduction = "average")`;
      + a `NamedTuple` with at least `short_name` and `period`, e.g.
        `(; short_name = "ts", period = "1hours")`;
      + a YAML-style `Dict{String, Any}`, the shape produced by the `diagnostics:` YAML key.
  - `interpolation_num_points = nothing`: Override for the NetCDF remap grid, e.g.
    `(180, 90, 10)`. When `nothing`, the default for the underlying space is used.
  - `output_at_levels::Bool = true`: Whether to write on model levels, applying no vertical
    interpolation. Set to `false` to interpolate to pressure levels instead.
  - `debug_tendency::Bool = false`: include the column-integrated per-process
    tendency diagnostics (short names of the form `<field>_tend_<process>_colint`).
    Debug-only; each sample allocates a full `Y`-sized `FieldVector` and runs
    one extra tendency evaluation.

A simulation produces no diagnostics when `default = false`, `debug_tendency = false`,
and `additional` is empty.

# Examples

```julia
import ClimaAtmos as CA

# Defaults only.
config = CA.DiagnosticsConfig()

# Defaults plus half-hourly mean zonal wind and hourly instantaneous surface temperature.
config = CA.DiagnosticsConfig(;
    additional = (
        "ua" => (; period = "30mins", reduction = "average"),
        (; short_name = "ts", period = "1hours"),
    ),
)

simulation = CA.AtmosSimulation{Float64}(; diagnostics = config)
```
"""
@kwdef struct DiagnosticsConfig{A}
    default::Bool = true
    additional::A = ()
    interpolation_num_points::Union{Nothing, Tuple, AbstractVector} = nothing
    output_at_levels::Bool = true
    debug_tendency::Bool = false
end

# `reduction` is a friendlier alias for the YAML schema's `reduction_time` key.
_diag_key(k::Symbol) = k === :reduction ? "reduction_time" : String(k)

"""
    normalize_diag_entry(entry)

Normalize one user-supplied diagnostic spec to a form the simulation setup understands.

A `ScheduledDiagnostic` is passed through unchanged; everything else becomes a
`Dict{String, Any}` matching the YAML diagnostic schema, with the friendlier `reduction`
key renamed to the schema's `reduction_time`. Throws on an unrecognized type, and on a
bare `"short_name" => "period"` pair, whose reduction would be ambiguous.

Called from `setup_diagnostics_and_writers`, so that the `additional` field of
[`DiagnosticsConfig`](@ref) can be written in several convenient forms.
"""
normalize_diag_entry(sd::ClimaDiagnostics.ScheduledDiagnostic) = sd
normalize_diag_entry(d::AbstractDict) = Dict{String, Any}(String(k) => v for (k, v) in d)
normalize_diag_entry(nt::NamedTuple) =
    Dict{String, Any}(_diag_key(k) => v for (k, v) in pairs(nt))
normalize_diag_entry(p::Pair{<:AbstractString, <:AbstractString}) = error(
    "Ambiguous diagnostic spec $(repr(p)): the reduction is unspecified. \
    Use `\"$(p.first)\" => (; period = $(repr(p.second)), reduction = \"average\")` \
    (or another reduction such as \"inst\", \"min\", \"max\") to be explicit.",
)
function normalize_diag_entry(p::Pair{<:AbstractString, <:NamedTuple})
    d = Dict{String, Any}("short_name" => p.first)
    for (k, v) in pairs(p.second)
        d[_diag_key(k)] = v
    end
    return d
end
normalize_diag_entry(x) = error(
    "Cannot interpret $(typeof(x)) as a diagnostic. Expected a \
    ScheduledDiagnostic, a Pair (short_name => period or short_name => NamedTuple), \
    a NamedTuple, or a Dict.",
)

"""
    extract_diagnostic_periods(diagnostics)

Collect the accumulation periods of the diagnostics that perform a time reduction.

Diagnostics without a `reduction_time_func`, and those whose output schedule has no fixed
period (such as `EveryStepSchedule` and `DivisorSchedule`), contribute nothing. The result
is checked against the checkpointing frequency by
`validate_checkpoint_diagnostics_consistency`.

# Returns

A `Set` of `Dates.Period`.
"""
function extract_diagnostic_periods(diagnostics)
    schedule_period(s::EveryDtSchedule) = Dates.Second(s.dt)
    schedule_period(s::EveryCalendarDtSchedule) = s.dt
    schedule_period(_) = nothing

    return Set(
        Iterators.filter(
            !isnothing,
            schedule_period(d.output_schedule_func) for
            d in diagnostics if !isnothing(d.reduction_time_func)
        ),
    )
end
