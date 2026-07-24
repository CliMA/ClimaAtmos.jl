"""
    ColumnDatasets

Data access for single-column (SCM) forcing files.

The file format is a singleton type ([`ClimaColumnFile`](@ref), the native
ClimaColumn schema) whose module extends a small interface: canonical variable
names to file names, the file layout (profile, surface series, height
coordinate), and per-variable unit/derivation hooks. Generic machinery (the
[`ColumnDataset`](@ref) handle, `TimeVaryingInput` builders, initial-profile
reads) consumes it through that interface.

# Adding a format

Define a singleton subtype of [`AbstractColumnFormat`](@ref) in a new module
under `src/column_datasets/`, extend the three required methods
([`format_name`](@ref), [`format_variable_name`](@ref),
[`height_profile`](@ref)) plus any optional ones (`open_dataset`, `preprocess`,
`dates`, `read_profile`, `read_series`, `extrapolation_bc`,
`time_interpolation_method`, `site_location`, `validate`), and pass it via the
`format` keyword of [`ColumnDataset`](@ref).
"""
module ColumnDatasets

import NCDatasets as NC
import Dates
import Interpolations as Intp
import ClimaCore
import ClimaUtilities.TimeVaryingInputs
import ClimaUtilities.TimeVaryingInputs: TimeVaryingInput
import ClimaUtilities.Utils: period_to_seconds_float

# ============================================================================
# Canonical vocabulary (CMIP short names, SI units)
# ============================================================================

"""
    CANONICAL_COLUMN_VARS

The canonical column `(z, time)` forcing variables, named after their CMIP
short names and stored in SI units. A file-driven forcing requires only the
subset needed by the terms it composes.
"""
const CANONICAL_COLUMN_VARS = (
    :tntha,    # temperature horizontal-advection tendency [K s-1]
    :tnhusha,  # humidity horizontal-advection tendency [kg kg-1 s-1]
    :tntva,    # temperature vertical/eddy tendency [K s-1]
    :tnhusva,  # humidity vertical/eddy tendency [kg kg-1 s-1]
    :ta,       # air temperature nudging target [K]
    :hus,      # specific humidity nudging target [kg kg-1]
    :ua,       # eastward wind nudging target [m s-1]
    :va,       # northward wind nudging target [m s-1]
    :wa,       # large-scale subsidence velocity [m s-1]
)

"""
    CANONICAL_SURFACE_VARS

The canonical surface `(time,)` forcing variables, named after their CMIP short
names and stored in SI units.
"""
const CANONICAL_SURFACE_VARS = (
    :ts,      # surface (skin) temperature [K]
    :hfls,    # surface upward latent heat flux [W m-2]
    :hfss,    # surface upward sensible heat flux [W m-2]
    :coszen,  # cosine of the solar zenith angle [1]
    :rsdt,    # TOA incoming shortwave radiation [W m-2]
)

"""
    CANONICAL_IC_VARS

The canonical variables a file must carry for
`read_initial_profiles` to build a column initial condition:
temperature, both wind components, specific humidity, and density.
"""
const CANONICAL_IC_VARS = (:ta, :ua, :va, :hus, :rho)

# ============================================================================
# The format interface
# ============================================================================

"""
    AbstractColumnFormat

Supertype of the column forcing-file formats. A format is a singleton subtype
that teaches the generic machinery how to read one on-disk layout: it extends
`format_name`, `format_variable_name`, and `height_profile` (plus optional
hooks), and is passed via the `format` keyword of `ColumnDataset`. The native
format is `ClimaColumnFiles.ClimaColumnFile`.
"""
abstract type AbstractColumnFormat end

# --- required methods (every format module defines all three) ---

"""
    format_name(format)

The display name of the format, used in error messages.
"""
function format_name end

"""
    format_variable_name(format, name::Symbol)

The file variable name for the canonical variable `name`, or `nothing` when
the format cannot represent it directly (e.g. a derived variable, handled by
a `read_*` override instead).
"""
function format_variable_name end

"""
    height_profile(format, ds, options)

The heights of the file's column levels, in the file's storage order [m].

This is where a format absorbs its vertical convention, e.g. dividing a
geopotential by `g` or converting pressure levels to height.
"""
function height_profile end

# --- optional methods with generic defaults ---

"""
    open_dataset(f, format, path, options)
    open_dataset(f, cd::ColumnDataset)

Open the file and apply `f` to the format-resolved `NCDataset`, closing it
afterwards. A format whose data lives in a subgroup rather than at the root
overrides this to pass the group to `f` instead.
"""
open_dataset(f, ::AbstractColumnFormat, path, options) =
    NC.NCDataset(f, path, "r")

"""
    preprocess(format, name::Symbol)

Elementwise function applied to every value of the canonical variable `name`
read from this format (unit conversions, fill-value handling). Applied both by
the direct `read_*` methods and, through the file reader's `preprocess_func`
hook, by file-backed `TimeVaryingInput`s.
"""
preprocess(::AbstractColumnFormat, name::Symbol) = identity

"""
    dates(format, ds)

The file's time axis as a vector of `DateTime`s. The default requires a
CF-decodable `time` variable; formats with nonstandard time conventions
(e.g. `base_time` offsets) override this.
"""
function dates(d::AbstractColumnFormat, ds)
    times = vec(ds["time"][:])
    eltype(times) <: Dates.AbstractDateTime || error(
        "The `time` variable of $(NC.path(ds)) does not decode to dates, \
         the $(format_name(d)) format requires CF time (units + calendar).",
    )
    return times
end

"""
    has_variable(format, ds, name::Symbol)

Whether the canonical variable `name` is available from this file. Formats
with derived variables override this alongside the corresponding `read_*`
method.
"""
function has_variable(d::AbstractColumnFormat, ds, name::Symbol)
    var = format_variable_name(d, name)
    return !isnothing(var) && haskey(ds, var)
end

"""
    read_profile(format, ds, name::Symbol, time_index)

The vertical profile of canonical variable `name` at one time index, in the
file's level order, with `preprocess` applied.
"""
function read_profile(d::AbstractColumnFormat, ds, name::Symbol, time_index)
    return preprocess(d, name).(ds[format_variable_name(d, name)][:, time_index])
end

"""
    read_series(format, ds, name::Symbol)

The full time series of the surface variable `name`, with `preprocess`
applied.
"""
function read_series(d::AbstractColumnFormat, ds, name::Symbol)
    return preprocess(d, name).(vec(ds[format_variable_name(d, name)][:]))
end

"""
    extrapolation_bc(format)

Extrapolation setting for file-backed `TimeVaryingInput`s of this format,
matching the dimensionality of its stored variables.
"""
extrapolation_bc(::AbstractColumnFormat) = (Intp.Flat(),)

"""
    time_interpolation_method(format)

Return the `method` for this format's `TimeVaryingInput`s, which fixes the
time-interpolation and the out-of-range extrapolation policy. The default is
plain `LinearInterpolation()`: it interpolates linearly within the file's time
span and errors out of range, so a finite campaign cannot fabricate forcing by
wrapping around. A case whose file stores one repeating period (e.g. the
monthly-averaged-diurnal ERA5 file, one day) overrides this with
`periodic_calendar_method`.
"""
time_interpolation_method(::AbstractColumnFormat) =
    TimeVaryingInputs.LinearInterpolation()

"""
    periodic_calendar_method()

The `TimeVaryingInput` method that repeats a file's time axis periodically, for
a file that stores exactly one period. Passed as an `ExternalDrivenTVForcing`
`time_interpolation_method` by the monthly-averaged-diurnal ERA5 case, whose
file stores a single day.
"""
periodic_calendar_method() =
    TimeVaryingInputs.LinearInterpolation(TimeVaryingInputs.PeriodicCalendar())

"""
    site_location(format, ds)
    site_location(cd::ColumnDataset)

Return a NamedTuple `(; latitude, longitude)` of the column site in degrees.
Consumed by whoever constructs behavior that needs the location (e.g. an
astronomical insolation model). The default is an informative error.
"""
site_location(d::AbstractColumnFormat, ds) = error(
    "format $(format_name(d)) does not record a site location, pass \
     latitude/longitude explicitly",
)

"""
    validate(format, path)

Check `path` against the format's specification, throwing a
descriptive error listing all violations. The default is a no-op for formats
without a formal specification.
"""
validate(::AbstractColumnFormat, path) = nothing

# ============================================================================
# Column data sources
# ============================================================================

"""
    AbstractColumnData

Supertype of the column forcing data sources consumed by the file-driven
forcing and `ForcingFromFile`: [`ColumnDataset`](@ref) (an on-disk file read
through a format) and [`InMemoryColumnData`](@ref) (in-memory arrays, e.g. the
steady GCM-driven profiles). Both provide the same reader interface
(`column_timevaryinginputs`, `surface_timevaryinginputs`,
`read_surface_series`, `read_initial_profiles`, `require_forcing_variables`,
`time_interpolation_method`, `site_location`) and expose `column_vars` and
`surface_vars`.
"""
abstract type AbstractColumnData end

"""
    ColumnDataset(path; format = nothing, options...)

Handle to one column forcing file.

Construction runs the format's [`validate`](@ref) method and then probes, once,
which canonical variables the file carries, so that a non-conforming or
incomplete file errors here rather than mid-simulation.

# Fields

  - `format`: The [`AbstractColumnFormat`](@ref) of the file; the native
    [`ClimaColumnFile`](@ref) unless the `format` keyword says otherwise.
  - `path`: Path to the file.
  - `options`: Format-specific options passed through as keywords at
    construction.
  - `column_vars`, `surface_vars`: The canonical column and surface variables the
    file actually carries.
"""
struct ColumnDataset{F <: AbstractColumnFormat, O <: NamedTuple} <:
       AbstractColumnData
    format::F
    path::String
    options::O
    column_vars::Vector{Symbol}
    surface_vars::Vector{Symbol}
end

function ColumnDataset(path::AbstractString; format = nothing, options...)
    options_nt = (; options...)
    fmt = isnothing(format) ? ClimaColumnFile() : format
    validate(fmt, path)
    probe = open_dataset(fmt, path, options_nt) do ds
        (;
            column_vars = [
                name for
                name in CANONICAL_COLUMN_VARS if has_variable(fmt, ds, name)
            ],
            surface_vars = [
                name for
                name in CANONICAL_SURFACE_VARS if has_variable(fmt, ds, name)
            ],
        )
    end
    return ColumnDataset(
        fmt,
        String(path),
        options_nt,
        probe.column_vars,
        probe.surface_vars,
    )
end

open_dataset(f, cd::ColumnDataset) =
    open_dataset(f, cd.format, cd.path, cd.options)

validate(cd::ColumnDataset) = validate(cd.format, cd.path)

site_location(cd::ColumnDataset) =
    open_dataset(ds -> site_location(cd.format, ds), cd)

time_interpolation_method(cd::ColumnDataset) =
    time_interpolation_method(cd.format)

source_name(cd::ColumnDataset) = "$(cd.path) ($(format_name(cd.format)) format)"

# ============================================================================
# Generic machinery (shared by every `AbstractColumnData`)
# ============================================================================

# The variables in `column_vars`/`surface_vars` that the source does not carry.
function _missing_required(data::AbstractColumnData, column_vars, surface_vars)
    return (
        column = setdiff(column_vars, data.column_vars),
        surface = setdiff(surface_vars, data.surface_vars),
    )
end

"""
    missing_forcing_variables(cd, column_vars, surface_vars)
    missing_forcing_variables(cd)

The requested forcing variables absent from the file: those in the given
`column_vars`/`surface_vars`, or (single-argument form) the full canonical
vocabulary.
"""
function missing_forcing_variables(
    data::AbstractColumnData,
    column_vars,
    surface_vars,
)
    m = _missing_required(data, column_vars, surface_vars)
    return Symbol[m.column; m.surface]
end
missing_forcing_variables(data::AbstractColumnData) = missing_forcing_variables(
    data,
    CANONICAL_COLUMN_VARS,
    CANONICAL_SURFACE_VARS,
)

"""
    require_forcing_variables(cd, column_vars, surface_vars)

Error, naming what is absent, unless the file carries every variable in
`column_vars` (needed by the composed forcing terms) and `surface_vars`
(needed by the resolved model).
"""
function require_forcing_variables(
    data::AbstractColumnData,
    column_vars,
    surface_vars,
)
    m = _missing_required(data, column_vars, surface_vars)
    isempty(m.column) && isempty(m.surface) && return nothing
    advice = String[]
    isempty(m.column) || push!(
        advice,
        "$(join(m.column, ", ")) required by the composed forcing terms. \
         Provide the data or drop the term that needs it when constructing \
         `ExternalDrivenTVForcing`",
    )
    isempty(m.surface) || push!(
        advice,
        "$(join(m.surface, ", ")) required by the resolved model (`ts` by an \
         `ExternalTemperature` surface, `coszen`/`rsdt` by `ExternalTVInsolation` \
         with RRTMGP radiation). Provide the data or change those model choices",
    )
    error(
        "Forcing data $(source_name(data)) is missing required variables: \
         $(join(advice, "; ")).",
    )
end

"""
    time_index_closest(format, ds, date)

Index of the file time closest to `date`.
"""
time_index_closest(d::AbstractColumnFormat, ds, date) =
    argmin(abs.(dates(d, ds) .- date))

"""
    simulation_times(format, ds, start_date)

The file's time axis as simulation time in seconds, with `t = 0` at
`start_date`.
"""
simulation_times(d::AbstractColumnFormat, ds, start_date) =
    period_to_seconds_float.(dates(d, ds) .- start_date)

"""
    file_time_span(cd, start_date)

The time in seconds from `start_date` to the file's last time. A simulation
longer than this runs past the end of the file's data.
"""
file_time_span(cd::ColumnDataset, start_date) =
    open_dataset(cd) do ds
        maximum(simulation_times(cd.format, ds, start_date))
    end

"""
    wraps_periodically(method)

Whether a `TimeVaryingInput` `method` repeats its data past the file's time
range (a `PeriodicCalendar` boundary) instead of erroring out of range.
"""
wraps_periodically(method) =
    TimeVaryingInputs.extrapolation_bc(method) isa
    TimeVaryingInputs.PeriodicCalendar

"""
    read_initial_profiles(cd, ds, start_date)

The initial-condition profiles at the file time closest to `start_date`.

# Returns

`(; z, ta, ua, va, hus, rho)`: the height coordinate [m] and the
`CANONICAL_IC_VARS` profiles, all sorted ascending in `z`. Errors,
naming what is absent, when the file lacks a variable needed to build an
initial condition.
"""
function read_initial_profiles(cd::ColumnDataset, ds, start_date)
    d = cd.format
    missing_vars = [n for n in CANONICAL_IC_VARS if !has_variable(d, ds, n)]
    isempty(missing_vars) || error(
        "Initializing a column from $(cd.path) requires the variables \
         $(join(CANONICAL_IC_VARS, ", ")); missing: \
         $(join(missing_vars, ", ")).",
    )
    time_index = time_index_closest(d, ds, start_date)
    z = Float64.(height_profile(d, ds, cd.options))
    order = sortperm(z)
    profile(name) = Float64.(read_profile(d, ds, name, time_index))[order]
    return (;
        z = z[order],
        ta = profile(:ta),
        ua = profile(:ua),
        va = profile(:va),
        hus = profile(:hus),
        rho = profile(:rho),
    )
end

"""
    read_initial_profiles(data, start_date)

The initial-condition profiles for `data`, opening the file (for a
[`ColumnDataset`](@ref)) or reading the stored arrays (for an
[`InMemoryColumnData`](@ref)).
"""
read_initial_profiles(cd::ColumnDataset, start_date) =
    open_dataset(cd) do ds
        read_initial_profiles(cd, ds, start_date)
    end

"""
    column_timevaryinginputs(cd, names, target_space, start_date; method)

A `NamedTuple` of `TimeVaryingInput`s, one per requested column variable,
targeting `target_space`, the model's center column space.

The default builds file-backed inputs, applying the format's
`extrapolation_bc` and `preprocess` hooks. A format whose
on-disk layout the file readers cannot consume directly — a grouped file, or a
non-height vertical coordinate — overrides this to build in-memory inputs
instead.
"""
function column_timevaryinginputs(
    cd::ColumnDataset,
    names,
    target_space,
    start_date;
    method = time_interpolation_method(cd.format),
)
    names = Tuple(names)
    d = cd.format
    inputs = map(names) do name
        prep = preprocess(d, name)
        file_reader_kwargs =
            prep === identity ? (;) : (; preprocess_func = prep)
        TimeVaryingInput(
            cd.path,
            format_variable_name(d, name),
            target_space;
            start_date,
            regridder_kwargs = (; extrapolation_bc = extrapolation_bc(d)),
            file_reader_kwargs,
            method,
        )
    end
    return NamedTuple{names}(inputs)
end

"""
    read_surface_series(cd, names, start_date)

Read the surface variables `names` in a single file open.

The data layer that both `surface_timevaryinginputs` and data-backed
surface components, such as a prescribed-flux scheme, build on.

# Returns

`(; times, name₁ = series₁, ...)`, where `times` is the simulation time axis in
seconds with `t = 0` at `start_date`, and each series has `preprocess`
applied.
"""
function read_surface_series(cd::ColumnDataset, names, start_date)
    names = Tuple(names)
    d = cd.format
    return open_dataset(cd) do ds
        times = Float64.(simulation_times(d, ds, start_date))
        series = map(name -> Float64.(read_series(d, ds, name)), names)
        (; times, NamedTuple{names}(series)...)
    end
end

"""
    surface_timevaryinginputs(cd, names, target_space, start_date; method)

A `NamedTuple` of `TimeVaryingInput`s, one per requested surface variable,
read into in-memory inputs on the simulation time axis (`t = 0` at
`start_date`) from a single file open.
"""
function surface_timevaryinginputs(
    cd::ColumnDataset,
    names,
    target_space,
    start_date;
    method = time_interpolation_method(cd.format),
)
    names = Tuple(names)
    FT = ClimaCore.Spaces.undertype(target_space)
    read = read_surface_series(cd, names, start_date)
    times = FT.(read.times)
    inputs = map(name -> TimeVaryingInput(times, FT.(read[name]); method), names)
    return NamedTuple{names}(inputs)
end

# ============================================================================
# In-memory column data
# ============================================================================

"""
    InMemoryColumnData(; z, column, surface, site_location = nothing, source)

A column data source backed by in-memory arrays instead of a file. `z` is the
ascending height coordinate [m], `column` a `NamedTuple` of `(z,)` profiles
(canonical variable => vector), and `surface` a `NamedTuple` of scalars
(canonical variable => value). The data is steady: the reader interface returns
constant-in-time inputs. Used for sources with no on-disk ClimaColumn file, such
as the steady GCM-driven profiles built by `GCMColumnData`.
"""
struct InMemoryColumnData{C <: NamedTuple, S <: NamedTuple, L} <:
       AbstractColumnData
    z::Vector{Float64}
    column::C
    surface::S
    site_location::L
    source::String
    column_vars::Vector{Symbol}
    surface_vars::Vector{Symbol}
end

function InMemoryColumnData(;
    z,
    column,
    surface,
    site_location = nothing,
    source = "in-memory column data",
)
    issorted(z) ||
        error("`InMemoryColumnData` requires an ascending `z` coordinate")
    return InMemoryColumnData(
        Float64.(z),
        column,
        surface,
        site_location,
        source,
        collect(Symbol, keys(column)),
        collect(Symbol, keys(surface)),
    )
end

source_name(d::InMemoryColumnData) = d.source

time_interpolation_method(::InMemoryColumnData) =
    TimeVaryingInputs.LinearInterpolation()

file_time_span(::InMemoryColumnData, _) = Inf

site_location(d::InMemoryColumnData) =
    isnothing(d.site_location) ?
    error("in-memory column source ($(d.source)) has no site location") :
    d.site_location

read_initial_profiles(d::InMemoryColumnData, start_date) = (;
    z = d.z,
    ta = d.column.ta,
    ua = d.column.ua,
    va = d.column.va,
    hus = d.column.hus,
    rho = d.column.rho,
)

# Interpolate an ascending `(z_src,) -> vals_src` profile onto the column of
# `target_space`, flat beyond the source range, returning a Field.
function _interp_column(z_src, vals_src, target_space)
    ᶜz = ClimaCore.Fields.coordinate_field(target_space).z
    field = similar(ᶜz)
    itp = Intp.extrapolate(
        Intp.interpolate(
            (Float64.(z_src),),
            Float64.(vals_src),
            Intp.Gridded(Intp.Linear()),
        ),
        Intp.Flat(),
    )
    parent(field) .= itp.(parent(ᶜz))
    return field
end

# Steady in-memory profiles become constant-in-time analytic inputs: the
# pre-interpolated field is copied into the destination on every `evaluate!`.
function column_timevaryinginputs(
    d::InMemoryColumnData,
    names,
    target_space,
    start_date;
    method = nothing,
)
    names = Tuple(names)
    inputs = map(names) do name
        field = _interp_column(d.z, d.column[name], target_space)
        TimeVaryingInput(Returns(field))
    end
    return NamedTuple{names}(inputs)
end

function surface_timevaryinginputs(
    d::InMemoryColumnData,
    names,
    target_space,
    start_date;
    method = nothing,
)
    names = Tuple(names)
    FT = ClimaCore.Spaces.undertype(target_space)
    inputs = map(name -> TimeVaryingInput(Returns(FT(d.surface[name]))), names)
    return NamedTuple{names}(inputs)
end

function read_surface_series(d::InMemoryColumnData, names, start_date)
    names = Tuple(names)
    series = map(name -> [Float64(d.surface[name])], names)
    return (; times = [0.0], NamedTuple{names}(series)...)
end

# ============================================================================
# Format modules
# ============================================================================

include("ClimaColumnFiles.jl")
# Converter-only: reads a source format and writes a ClimaColumn-schema file
# that the reader above then consumes.
include("VaranalFiles.jl")
# Reader-only: builds a steady `InMemoryColumnData` from a GCM cfsite file.
include("GCMColumnData.jl")

using .ClimaColumnFiles: ClimaColumnFile

end # module
