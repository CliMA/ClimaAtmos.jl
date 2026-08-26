"""
SSCF forcing as an in-memory ClimaAtmos column dataset.

`ColumnDatasets` is ClimaAtmos's pluggable column-forcing-file interface: a format is a singleton
type that teaches generic machinery how to read one layout. Nothing in that interface requires the
data to be on disk — the reads and the `TimeVaryingInput` builders are all format-dispatched — so a
format whose "file" is a NamedTuple of arrays in memory plugs straight into the stock
`ExternalDrivenTVForcing` and gets its composed forcing terms, surface wiring, and error messages
for free.

The arrays are sampled onto the model's own levels (see [`socrates_z`](@ref)) once per
`(flight, forcing, grid, FT)` and memoized, because the forcing does not depend on any calibrated
parameter and is otherwise rebuilt for every ensemble member of every iteration.
"""

using ClimaAtmos: ClimaAtmos as CA
using SOCRATESSingleColumnForcings: SOCRATESSingleColumnForcings as SSCF

"""
    SocratesColumnFormat

Column-forcing format whose data lives in memory: the `ColumnDataset`'s `options` carry the
sampled arrays, and its `path` is a label (`"sscf:RF09_Obs"`) rather than a file.
"""
struct SocratesColumnFormat <: CA.ColumnDatasets.AbstractColumnFormat end

CA.ColumnDatasets.format_name(::SocratesColumnFormat) = "SOCRATES (in-memory SSCF)"

# Canonical names are used as-is: the sampled arrays are keyed by them directly.
CA.ColumnDatasets.format_variable_name(::SocratesColumnFormat, name::Symbol) = name

# The "dataset" is the options NamedTuple itself; there is no file to open.
CA.ColumnDatasets.open_dataset(f, ::SocratesColumnFormat, path, options) = f(options)

CA.ColumnDatasets.height_profile(::SocratesColumnFormat, ds, options) = ds.z

CA.ColumnDatasets.dates(::SocratesColumnFormat, ds) =
    ds.start_date .+ Dates.Millisecond.(round.(Int, 1000 .* ds.times))

CA.ColumnDatasets.has_variable(::SocratesColumnFormat, ds, name::Symbol) =
    haskey(ds.column, name) || haskey(ds.surface, name)

CA.ColumnDatasets.read_profile(::SocratesColumnFormat, ds, name::Symbol, time_index) =
    ds.column[name][:, time_index]

CA.ColumnDatasets.read_series(::SocratesColumnFormat, ds, name::Symbol) = ds.surface[name]

CA.ColumnDatasets.site_location(::SocratesColumnFormat, ds) =
    (; latitude = ds.lat, longitude = ds.lon)

"""
    column_timevaryinginputs(cd::ColumnDataset{SocratesColumnFormat}, names, target_space, start_date; method)

Column inputs built from the in-memory arrays. The generic implementation builds file-backed
inputs through a `DataHandler`; this is the documented override for a format whose layout the file
readers cannot consume.

The arrays are already on the model's levels, so no vertical regridding happens here — the
constructor errors if they do not match `target_space`.
"""
function CA.ColumnDatasets.column_timevaryinginputs(
    cd::CA.ColumnDatasets.ColumnDataset{SocratesColumnFormat},
    names,
    target_space,
    start_date;
    method = CA.ColumnDatasets.time_interpolation_method(cd.format),
)
    names = Tuple(names)
    inputs = CA.ColumnDatasets.open_dataset(cd) do ds
        start_date == ds.start_date || error(
            "SOCRATES forcing was sampled with start_date $(ds.start_date) but the simulation \
             uses $start_date; the time axes would not line up.",
        )
        map(names) do name
            haskey(ds.column, name) || error(
                "SOCRATES forcing has no column variable `$name`; it provides \
                 $(sort(collect(keys(ds.column)))).",
            )
            ColumnMemoryTimeVaryingInput(ds.times, ds.column[name], target_space; method)
        end
    end
    return NamedTuple{names}(inputs)
end

# --- sampling SSCF onto the model grid ------------------------------------------------------ #

"""
SSCF column outputs this experiment needs, and the canonical `ColumnDatasets` name each maps to.
`tntva`/`tnhusva` have no SSCF counterpart and are supplied as zeros (see
[`socrates_forcing_arrays`](@ref)).
"""
const SSCF_TO_CANONICAL = (
    dTdt_hadv = :tntha,
    dqtdt_hadv = :tnhusha,
    T_nudge = :ta,
    qt_nudge = :hus,
    u_nudge = :ua,
    v_nudge = :va,
    subsidence = :wa,
)

const SSCF_COLUMN_KEYS = Tuple(keys(SSCF_TO_CANONICAL))

"""Default spacing [s] of the sampled forcing time axis."""
const DEFAULT_FORCING_DT = 300.0

# Memo of sampled forcing, keyed by everything that changes it. The forcing is independent of any
# calibrated parameter, so on a calibration worker this is built once per case instead of once per
# (member, case, iteration).
const _FORCING_CACHE = Dict{Any, Any}()

_forcing_cache_key(FT, case, z, dt_sec, start_date) =
    (FT, case.flight_number, forcing_label(case), hash(z), dt_sec, start_date)

"""
    socrates_forcing_arrays(FT, case; z, dt_sec, start_date)

Sample SSCF forcing for `case` onto levels `z` and a time axis of spacing `dt_sec` covering the
whole run, returning the NamedTuple that backs a [`SocratesColumnFormat`](@ref) dataset:

  - `z`, `times` (seconds since `start_date`), `start_date`, `lat`, `lon`
  - `column`: canonical `(n_levels, n_times)` matrices
  - `surface`: canonical length-`n_times` series (`ts`, `coszen`, `rsdt`)

The time axis extends one `dt_sec` past `t_end` so the final integrator step cannot fall off the
end of the data.

Results are memoized; pass `refresh = true` to rebuild.
"""
function socrates_forcing_arrays(
    ::Type{FT},
    case::SocratesCase;
    z::AbstractVector,
    dt_sec::Real = DEFAULT_FORCING_DT,
    start_date::Dates.DateTime = simulation_start_date(case),
    refresh::Bool = false,
) where {FT <: AbstractFloat}
    key = _forcing_cache_key(FT, case, z, dt_sec, start_date)
    if !refresh && haskey(_FORCING_CACHE, key)
        return _FORCING_CACHE[key]
    end
    arrays = _build_forcing_arrays(FT, case; z, dt_sec, start_date)
    _FORCING_CACHE[key] = arrays
    return arrays
end

"""Number of memoized forcing entries (diagnostic for the per-worker cache)."""
forcing_cache_size() = length(_FORCING_CACHE)

"""Drop all memoized forcing."""
empty_forcing_cache!() = (empty!(_FORCING_CACHE); nothing)

function _build_forcing_arrays(
    ::Type{FT},
    case::SocratesCase;
    z::AbstractVector,
    dt_sec::Real,
    start_date::Dates.DateTime,
) where {FT <: AbstractFloat}
    validate(case)
    ft = case.forcing_type
    flight = case.flight_number
    z_model = collect(FT, z)
    thermo = _thermodynamics_backend(FT)

    # One extra sample past t_end so the last integrator step stays inside the data.
    times = collect(FT, 0:FT(dt_sec):(FT(t_end(case)) + FT(dt_sec)))

    column = SSCF.get_column_forcing(
        flight,
        ft,
        SSCF_COLUMN_KEYS;
        new_z = z_model,
        thermodynamics_backend = thermo,
    )
    surface = SSCF.get_surface_forcing(flight, ft; thermodynamics_backend = thermo)

    sampled = Dict{Symbol, Matrix{FT}}()
    for (sscf_name, canonical) in pairs(SSCF_TO_CANONICAL)
        sampled[canonical] = _sample_levels(FT, column[sscf_name], z_model, times)
    end
    # SSCF carries no vertical eddy-fluctuation tendency; the composed forcing may still request
    # these names, so they are supplied explicitly as zeros rather than left absent.
    sampled[:tntva] = zeros(FT, length(z_model), length(times))
    sampled[:tnhusva] = zeros(FT, length(z_model), length(times))
    sampled[:rho] = _initial_density(FT, case, z_model, length(times))

    lat, lon = _site_location(FT, case)
    coszen, rsdt = _insolation_series(FT, case, times, lat, lon)
    surface_series = Dict{Symbol, Vector{FT}}(
        :ts => FT[FT(surface.Tsfc(t)) for t in times],
        :coszen => coszen,
        :rsdt => rsdt,
    )

    return (;
        z = z_model,
        times,
        start_date,
        lat,
        lon,
        column = sampled,
        surface = surface_series,
    )
end

# SSCF returns one time interpolant per level; sample them onto the (level, time) grid.
function _sample_levels(
    ::Type{FT},
    level_interpolants,
    z_model::AbstractVector,
    times::AbstractVector,
) where {FT}
    length(level_interpolants) == length(z_model) || error(
        "SSCF returned $(length(level_interpolants)) level interpolants for a $(length(z_model))-level \
         grid; `new_z` was not honored.",
    )
    out = Matrix{FT}(undef, length(z_model), length(times))
    for (j, t) in enumerate(times), k in eachindex(z_model)
        out[k, j] = FT(level_interpolants[k](t))
    end
    return out
end

"""
    _initial_density(FT, case, z_model, n_times)

Initial air density [kg m^-3] on the model levels, linearly interpolated from the Atlas LES
density at the first LES output time, then held constant in time.

`ColumnProfiles` uses this to set the initial state directly (`column_profiles_ic` passes `ρ`), so
it is the LES density rather than a hydrostatic reconstruction. Only column 1 is ever read — the
initial condition — but it is stored as a `(level, time)` matrix so it satisfies the same
`read_profile` contract as everything else.
"""
function _initial_density(
    ::Type{FT},
    case::SocratesCase,
    z_model::AbstractVector,
    n_times::Int,
) where {FT}
    les = SSCF.open_atlas_les_output(case.flight_number, case.forcing_type)
    z_les = FT.(vec(Array(les.data["z"])))
    ρ_les = FT.(Array(les.data["RHO"])[:, 1])
    ρ = _linear_interp(z_model, z_les, ρ_les)
    return repeat(reshape(ρ, :, 1), 1, n_times)
end

# Linear interpolation with flat extrapolation; `xs` ascending.
function _linear_interp(x_out::AbstractVector, xs::AbstractVector, ys::AbstractVector)
    issorted(xs) || error("_linear_interp requires ascending source coordinates")
    FT = eltype(ys)
    out = similar(x_out, FT)
    for (i, x) in enumerate(x_out)
        if x <= first(xs)
            out[i] = first(ys)
        elseif x >= last(xs)
            out[i] = last(ys)
        else
            j = searchsortedlast(xs, x)
            w = (x - xs[j]) / (xs[j + 1] - xs[j])
            out[i] = (1 - w) * ys[j] + w * ys[j + 1]
        end
    end
    return out
end

function _site_location(::Type{FT}, case::SocratesCase) where {FT}
    inp = SSCF.open_atlas_les_input(case.flight_number, case.forcing_type)
    lat = FT(Array(inp.data["lat"])[1])
    lon = FT(Array(inp.data["lon"])[1])
    return (FT(lat), FT(mod(lon + 180, 360) - 180))
end

"""
Insolation (`coszen`, `rsdt`), held constant at the case's reference time — Atlas et al. (2020)
section 4.2: "The solar zenith angle is held constant at the reference time of the case."

`les_start_datetime` is the reference time minus 12 h, so the reference time is 12 h after it.
Stored as a series so it satisfies the same `read_series` contract as the other surface fields.
"""
function _insolation_series(
    ::Type{FT},
    case::SocratesCase,
    times::AbstractVector,
    lat,
    lon,
) where {FT}
    params = CA.Insolation.Parameters.InsolationParameters(FT)
    reference = les_start_datetime(case) + Dates.Hour(12)
    F, _, μ, _ = CA.Insolation.insolation(reference, FT(lat), FT(lon), params)
    return fill(FT(μ), length(times)), fill(FT(F), length(times))
end

# SSCF's accurate thermodynamics path needs a ThermodynamicsParameters; build it in FT.
_thermodynamics_backend(::Type{FT}) where {FT} =
    CA.Thermodynamics.Parameters.ThermodynamicsParameters(CP.create_toml_dict(FT))

"""
    socrates_forcing(FT, case; z, dt_sec, start_date, forcing_terms, refresh)

The `ExternalDrivenTVForcing` for `case`, backed by the in-memory dataset.

`forcing_terms` selects which composed terms are active. The default matches the SOCRATES setup:
horizontal advection, nudging of `(ta, hus)` and `(ua, va)` on the Atlas timescales, and subsidence.
Vertical eddy fluctuation is omitted because SSCF supplies no such tendency.
"""
function socrates_forcing(
    ::Type{FT},
    case::SocratesCase;
    z::AbstractVector,
    dt_sec::Real = DEFAULT_FORCING_DT,
    start_date::Dates.DateTime = simulation_start_date(case),
    forcing_terms = default_socrates_forcing_terms(case),
    refresh::Bool = false,
) where {FT <: AbstractFloat}
    arrays = socrates_forcing_arrays(FT, case; z, dt_sec, start_date, refresh)
    dataset = CA.ColumnDatasets.ColumnDataset(
        "sscf:$(case_name(case))";
        format = SocratesColumnFormat(),
        arrays...,
    )
    return CA.ExternalDrivenTVForcing(dataset; forcing = forcing_terms)
end

"""Scalar and wind nudging timescales [s], from Atlas et al. (2020) section 4.2."""
const SCALAR_NUDGE_TIMESCALE = 20.0 * 60.0
const OBS_WIND_NUDGE_TIMESCALE = 20.0 * 60.0
const ERA5_WIND_NUDGE_TIMESCALE = 60.0 * 60.0

wind_nudge_timescale(case::SocratesCase) =
    case.forcing_type isa SSCF.ObsForcing ? OBS_WIND_NUDGE_TIMESCALE :
    ERA5_WIND_NUDGE_TIMESCALE

"""
The forcing composition for SOCRATES: no `VerticalFluctuation` term, because SSCF provides no
vertical eddy-fluctuation tendency for these cases.

A numeric `timescale` relaxes over the whole column. `DefaultTimescale()` would instead apply
ClimaAtmos's GCM-driven profile, which is zero below `gcmdriven_relaxation_minimum_height` and so
leaves the boundary layer unconstrained.
"""
default_socrates_forcing_terms(case::SocratesCase) = (
    CA.HorizontalAdvection(),
    CA.Nudging(:ta, :hus; timescale = SCALAR_NUDGE_TIMESCALE),
    CA.Nudging(:ua, :va; timescale = wind_nudge_timescale(case)),
    CA.Subsidence(),
)