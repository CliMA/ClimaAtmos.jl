"""
    DiagnosticsConfig(; default = true, additional = (), interpolation_num_points = nothing, output_at_levels = true)

Specification of which diagnostics a simulation produces and how their NetCDF
output is shaped. A single `DiagnosticsConfig` value is passed to
[`AtmosSimulation`](@ref ClimaAtmos.AtmosSimulation) via the `diagnostics`
keyword argument.

# Fields

- `default::Bool = true`: include the built-in ClimaAtmos diagnostic set for
  the chosen `AtmosModel`.
- `additional = ()`: extra user-supplied diagnostics. Each entry can be:
    - a `ClimaDiagnostics.ScheduledDiagnostic` (full control);
    - a `Pair` like `"ua" => (; period = "30mins", reduction = "average")`
      (short_name => options as a NamedTuple);
    - a NamedTuple with at least `short_name` and `period`,
      e.g. `(; short_name = "ts", period = "1hours")`;
    - a YAML-style `Dict{String,Any}` (the same shape produced by the
      `diagnostics:` YAML key).

  Mixed lists are allowed.
- `interpolation_num_points = nothing`: override the NetCDF remap grid (e.g.
  `(180, 90, 10)`). When `nothing`, falls back to the default chosen from the
  underlying space.
- `output_at_levels::Bool = true`: write at model levels (no vertical
  interpolation). Set `false` to interpolate to pressure levels.

A simulation produces no diagnostics when `default = false` and `additional`
is empty.
"""
@kwdef struct DiagnosticsConfig{A}
    default::Bool = true
    additional::A = ()
    interpolation_num_points::Union{Nothing, Tuple, AbstractVector} = nothing
    output_at_levels::Bool = true
end

# `reduction` is a friendlier alias for the YAML schema's `reduction_time` key.
_diag_key(k::Symbol) = k === :reduction ? "reduction_time" : String(k)

"""
    normalize_diag_entry(entry)

Convert a user-supplied diagnostic spec into either a
`ClimaDiagnostics.ScheduledDiagnostic` (passed through unchanged) or a
`Dict{String,Any}` matching the YAML diagnostic schema. Used internally by
`setup_diagnostics_and_writers` so callers can write diagnostics in several
convenient forms — see [`DiagnosticsConfig`](@ref).
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

Extract accumulation periods from diagnostics that have reduction functions.
Returns a Set of Period objects.
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
