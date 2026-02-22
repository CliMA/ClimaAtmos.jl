#####
##### Utility functions
#####
import ClimaComms
import ClimaCore: Spaces, Topologies, Fields, Geometry
import LinearAlgebra: norm_sqr
using Dates: DateTime, @dateformat_str
import StaticArrays: SVector, SMatrix

"""
    enforce_mass_energy_consistency!(Y, p, ᶜΔρq_tot)

Updates density (ρ) and total energy (ρe_tot) to maintain mass and energy consistency
when ρq_tot changes by ᶜΔρq_tot due to limiter application.

Arguments:
- `Y`: The state vector, modified in place.
- `p`: Cache containing parameters and precomputed fields.
- `ᶜΔρq_tot`: The change in total water density due to limiter application.
"""
function enforce_mass_energy_consistency!(Y, p, ᶜΔρq_tot)
    thp = CAP.thermodynamics_params(p.params)
    ᶜT = p.precomputed.ᶜT
    ᶜΦ = p.core.ᶜΦ
    @. Y.c.ρ += ᶜΔρq_tot
    @. Y.c.ρe_tot += ᶜΔρq_tot * (TD.internal_energy_vapor(thp, ᶜT) + ᶜΦ)
    return nothing
end

is_energy_var(symbol) = symbol in (:ρe_tot,)
is_momentum_var(symbol) = symbol in (:uₕ, :u₃)
is_sgs_var(symbol) = symbol in (:sgsʲs,)
is_tracer_var(symbol) = !(
    symbol == :ρ ||
    symbol == :ρtke ||
    is_energy_var(symbol) ||
    is_momentum_var(symbol) ||
    is_sgs_var(symbol)
)

# we may be hitting a slow path:
# https://stackoverflow.com/questions/14687665/very-slow-stdpow-for-bases-very-close-to-1
fast_pow(x, y) = exp(y * log(x))

"""
    geopotential(grav, z)

Compute the geopotential (Φ) at height `z` using gravitational acceleration `grav`.

Φ = g * z

where:
- `grav` is the gravitational acceleration
- `z` is the height
"""
geopotential(grav, z) = grav * z

"""
    time_from_filename(file)

Returns the time (Float64) from a given filename

## Example
```
time_from_filename("day4.46906.hdf5")
392506.0
```
"""
function time_from_filename(file)
    arr = split(basename(file), ".")
    day = parse(Float64, replace(arr[1], "day" => ""))
    sec = parse(Float64, arr[2])
    return day * (60 * 60 * 24) + sec
end

"""
    sort_files_by_time(files)

Sorts an array of files by the time,
determined by the filename.
"""
sort_files_by_time(files) =
    permute!(files, sortperm(time_from_filename.(files)))

"""
    κ .= compute_kinetic(uₕ::Field, uᵥ::Field)

Compute the specific kinetic energy at cell centers, resulting in `κ` from
individual velocity components:

 - `κ = 1/2 (uₕ⋅uʰ + 2uʰ⋅ᶜI(uᵥ) + ᶜI(uᵥ⋅uᵛ))`
 - `uₕ` should be a `Covariant1Vector` or `Covariant12Vector`-valued field at
    cell centers, and
 - `uᵥ` should be a `Covariant3Vector`-valued field at cell faces.
"""
function compute_kinetic(uₕ, uᵥ)
    @assert eltype(uₕ) <: Union{C1, C2, C12}
    @assert eltype(uᵥ) <: C3
    FT = Spaces.undertype(axes(uₕ))
    onehalf = FT(1 / 2)
    return @. lazy(
        onehalf * (
            dot(C123(uₕ), CT123(uₕ)) +
            ᶜinterp(dot(C123(uᵥ), CT123(uᵥ))) +
            2 * dot(CT123(uₕ), ᶜinterp(C123(uᵥ)))
        ),
    )
end

"""
    compute_kinetic(Y::FieldVector)

Compute the specific kinetic energy at cell centers, where `Y` is the model
state.
"""
compute_kinetic(Y::Fields.FieldVector) = compute_kinetic(Y.c.uₕ, Y.f.u₃)

"""
    ϵ .= compute_strain_rate_center_vertical(ᶠu)

Compute the strain rate at cell centers from velocity at cell faces, with vertical gradients only.

Returns a lazy representation of the strain rate tensor.
"""
function compute_strain_rate_center_vertical(ᶠu)
    axis_uvw = Geometry.UVWAxis()
    return @. lazy(
        (
            Geometry.project((axis_uvw,), ᶜgradᵥ(UVW(ᶠu))) +
            adjoint(Geometry.project((axis_uvw,), ᶜgradᵥ(UVW(ᶠu))))
        ) / 2,
    )
end

"""
    ϵ .= compute_strain_rate_face_vertical(ᶜu::Field)

Compute the strain rate at cell faces from velocity at cell centers, with vertical gradients only.

Returns a lazy representation of the strain rate tensor.
"""
function compute_strain_rate_face_vertical(ᶜu)
    ∇ᵥuvw_boundary = Geometry.outer(Geometry.WVector(0), Geometry.UVWVector(0, 0, 0))
    ∇bc = Operators.SetGradient(∇ᵥuvw_boundary)
    ᶠgradᵥ = Operators.GradientC2F(bottom = ∇bc, top = ∇bc)
    axis_uvw = Geometry.UVWAxis()
    return @. lazy(
        (
            Geometry.project((axis_uvw,), ᶠgradᵥ(UVW(ᶜu))) +
            adjoint(Geometry.project((axis_uvw,), ᶠgradᵥ(UVW(ᶜu))))
        ) / 2,
    )
end

"""
    compute_strain_rate_center_full!(ᶜε, ᶜu, ᶠu)

Compute the full strain rate tensor at cell centers from velocity

# Arguments
 - `ᶜε`: Preallocated `UVW x UVW` tensor field for the strain rate at cell centers
 - `ᶜu, ᶠu`: Velocity field at cell centers and faces, respectively.
    Both reconstructions are needed to compute the full strain rate tensor.

See also [`compute_strain_rate_face_full!`](@ref) for the analogous calculation on cell faces.

# Notes:
- it is recommended to use `ᶠu` and `ᶜu` as computed by
    [`set_velocity_quantities!`](@ref) and [`set_implicit_precomputed_quantities_part1!`](@ref)
- Because the computation involves both vertical and horizontal gradients, this
    calculation cannot be lazified (for now). It requires a pre-allocated output field.
"""
function compute_strain_rate_center_full!(ᶜε, ᶜu, ᶠu)
    axis_uvw = (Geometry.UVWAxis(),)
    @. ᶜε = Geometry.project(axis_uvw, ᶜgradᵥ(UVW(ᶠu)))  # vertical component
    @. ᶜε += Geometry.project(axis_uvw, gradₕ(UVW(ᶜu)))  # horizontal component
    @. ᶜε = (ᶜε + adjoint(ᶜε)) / 2
    return ᶜε
end

"""
    compute_strain_rate_face_full!(ᶠε, ᶜu, ᶠu)

Compute the full strain rate tensor at cell faces from velocity

# Arguments
 - `ᶠε`: Preallocated `UVW x UVW` tensor field for the strain rate at cell centers
 - `ᶜu, ᶠu`: Velocity field at cell centers and faces, respectively.
    Both reconstructions are needed to compute the full strain rate tensor.

See also [`compute_strain_rate_center_full!`](@ref) for the analogous calculation on cell centers.

# Notes:
- it is recommended to use `ᶠu` and `ᶜu` as computed by
    [`set_velocity_quantities!`](@ref) and [`set_implicit_precomputed_quantities_part1!`](@ref)
- Because the computation involves both vertical and horizontal gradients, this
    calculation cannot be lazified (for now). It requires a pre-allocated output field.
- The calculation assumes zero vertical gradient boundary conditions
"""
function compute_strain_rate_face_full!(ᶠε, ᶜu, ᶠu)
    ∇ᵥuvw_boundary = Geometry.outer(Geometry.WVector(0), UVW(0, 0, 0))
    ∇bc = Operators.SetGradient(∇ᵥuvw_boundary)
    ᶠgradᵥ = Operators.GradientC2F(bottom = ∇bc, top = ∇bc)
    axis_uvw = (Geometry.UVWAxis(),)
    @. ᶠε = Geometry.project(axis_uvw, ᶠgradᵥ(UVW(ᶜu)))  # vertical component
    @. ᶠε += Geometry.project(axis_uvw, gradₕ(UVW(ᶠu)))  # horizontal component
    @. ᶠε = (ᶠε + adjoint(ᶠε)) / 2
    return ᶠε
end

"""
    strain_rate_norm(S, axis = Geometry.UVWAxis())

Return a lazy representation of the strain rate norm `|S| = √(2 ∘ S : S)`

If `axis` is provided, project the strain rate tensor `S` onto the specified axis
before computing the norm.

For example,
- `axis = Geometry.UVAxis()` computes the horizontal strain rate norm, while
- `axis = Geometry.WAxis()` computes the vertical strain rate norm.
"""
function strain_rate_norm(S, axis = Geometry.UVWAxis())
    S_proj = @. lazy(Geometry.project((axis,), S, (axis,)))
    S_norm = @. lazy(sqrt(2 * norm_sqr(S_proj)))
    return S_norm
end

"""
    g³³_field(space)

Extracts the value of `g³³`, the 3rd component of the metric terms that convert
Covariant AxisTensors to Contravariant AxisTensors, from the given space.
"""
function g³³_field(space)
    g_field = Fields.local_geometry_field(space).gⁱʲ.components.data
    end_index = fieldcount(eltype(g_field)) # This will be 4 in 2D and 9 in 3D.
    return g_field.:($end_index) # For both 2D and 3D spaces, g³³ = g[end].
end

"""
    g³³(gⁱʲ)

Extracts the `g³³` sub-tensor from the `gⁱʲ` tensor.
"""
g³³(gⁱʲ) = Geometry.AxisTensor(
    (Geometry.Contravariant3Axis(), Geometry.Contravariant3Axis()),
    Geometry.components(gⁱʲ)[end],
)


"""
    g³ʰ(gⁱʲ)

Extracts the `g³ʰ` sub-tensor from the `gⁱʲ` tensor.
"""
function g³ʰ(gⁱʲ)
    full_CT_axis = axes(gⁱʲ)[1]
    N = length(full_CT_axis)
    gⁱʲ_components = Geometry.components(gⁱʲ)
    FT = eltype(gⁱʲ_components)
    g³ʰ_components = if full_CT_axis == Geometry.Contravariant123Axis()
        @inbounds SMatrix{1, 2, FT, 2}(
            gⁱʲ_components[N, 1],
            gⁱʲ_components[N, 2],
        )
    elseif full_CT_axis == Geometry.Contravariant13Axis()
        @inbounds val = gⁱʲ_components[N, 1]
        SMatrix{1, 2, FT, 2}(val, zero(FT))
    elseif full_CT_axis == Geometry.Contravariant23Axis()
        @inbounds val = gⁱʲ_components[N, 1]
        SMatrix{1, 2, FT, 2}(zero(FT), val)
    else
        error("$full_CT_axis is missing either vertical or horizontal sub-axes")
    end
    axes_tuple = (Geometry.Contravariant3Axis(), Geometry.Contravariant12Axis())
    return Geometry.AxisTensor(axes_tuple, g³ʰ_components)
end

has_topography(space::Spaces.FiniteDifferenceSpace) = false
has_topography(space) = Spaces.grid(space).hypsography != Spaces.Grids.Flat()

"""
    unit_basis_vector_data(type, local_geometry)

The component of the vector of the specified type with length 1 in physical units.
The type should correspond to a vector with only one component, i.e., a basis vector.
"""
function unit_basis_vector_data(::Type{V}, local_geometry) where {V}
    FT = Geometry.undertype(typeof(local_geometry))
    return FT(1) / Geometry._norm(V(FT(1)), local_geometry)
end

"""
    projected_vector_data(::Type{V}, vector, local_geometry)

Projects the given vector onto the axis of V, then extracts the component data and rescales it to physical units.
The type should correspond to a vector with only one component, i.e., a basis vector.
"""
projected_vector_data(::Type{V}, vector, local_geometry) where {V} =
    V(vector, local_geometry)[1] / unit_basis_vector_data(V, local_geometry)

function projected_vector_buoy_grad_vars(::Type{V}, v1, v2, lg) where {V}
    ubvd = unit_basis_vector_data(V, lg)
    return (;
        ∂qt∂z = V(v1, lg)[1] / ubvd,
        ∂θli∂z = V(v2, lg)[1] / ubvd,
    )
end

"""
    get_physical_w(u, local_geometry)

Return physical vertical velocity - a projection of full velocity vector
onto the vertical axis.
"""
get_physical_w(u, local_geometry) = Geometry.WVector(u, local_geometry)[1]

time_to_seconds(t::Number) =
    t == Inf ? t : error("Uncaught case in computing time from given string.")

"""
    time_to_seconds(s::String)

Convert a string representing a time to seconds. Supported units: seconds, minutes, hours, days, weeks as
`s`, `secs`, `m`, `mins`, `h`, `hours`, `d`, `days`, `weeks`.
"""
function time_to_seconds(s::String)
    s == "Inf" && return Inf
    # match a number followed by one of the supported units of time
    m = match(r"^(\d+(?:\.\d+)?)(s|secs|m|mins|h|hours|d|days|weeks)$", s)
    isnothing(m) &&
        error(
            "Bad format for flag $s. Examples: `10secs`, `20mins`, `30hours`, `40days`, `50weeks`",
        )
    value = parse(Float64, m.captures[1])
    unit = m.captures[2]
    factor_groups = Dict(
        ["s", "secs"] => 1,
        ["m", "mins"] => 60,
        ["h", "hours"] => 3600,
        ["d", "days"] => 86400,
        ["weeks"] => 604800,
    )
    factors = Dict(unit => val for (units, val) in factor_groups for unit in units)
    return value * factors[unit]
end

function error_if_crashed(ret_code)
    ret_code == :simulation_crashed &&
        error("The ClimaAtmos simulation has crashed. See the stack trace for details.")
end

function verify_callbacks(t)
    if length(t) ≠ length(unique(t))
        @show length(t)
        @show length(unique(t))
        error(
            string(
                "Saving duplicate solutions at the same time.",
                "Please change the callbacks to not save ",
                "duplicate solutions at the same timestep.",
            ),
        )
    end
end


"""
    do_dss(space)

Return whether the underlying horizontal space required DSS or not.
"""
function do_dss(space::Spaces.AbstractSpace)
    return Spaces.quadrature_style(Spaces.horizontal_space(space)) isa
           Quadratures.GLL
end

function do_dss(::Spaces.FiniteDifferenceSpace)
    return false
end

using ClimaComms
is_distributed(::ClimaComms.SingletonCommsContext) = false
is_distributed(::ClimaComms.MPICommsContext) = true

"""
    summary_string(x)

Returns a string that is similar to the output of `dump(x)`, but without any
type parameters.
"""
summary_string(x) = summary_string(x, 0)
summary_string(x, depth) =
    fieldcount(typeof(x)) == 0 ? repr(x) :
    (string(nameof(typeof(x))) * '(') *
    mapreduce(*, 1:fieldcount(typeof(x))) do i
        field =
            x isa Tuple ? ':' * string(i) : string(fieldname(typeof(x), i))
        ('\n' * "  "^(depth + 1) * field * " = ") *
        (summary_string(getfield(x, i), depth + 1) * ',')
    end *
    ('\n' * "  "^depth * ')')

# From BenchmarkTools
function prettytime(t)
    if t < 1e3
        value, units = t, "ns"
    elseif t < 1e6
        value, units = t / 1e3, "μs"
    elseif t < 1e9
        value, units = t / 1e6, "ms"
    else
        value, units = t / 1e9, "s"
    end
    return "$(round(value, digits=3)) $units"
end

import Dates

"""
    time_and_units_str(x::Real)

Returns a truncated string of time and units,
given a time `x` in Seconds.
"""
time_and_units_str(x::Real) =
    trunc_time(string(compound_period(x, Dates.Second)))

"""
    compound_period(x::Real, ::Type{T}) where {T <: Dates.Period}

A canonicalized `Dates.CompoundPeriod` given a real value
`x`, and its units via the period type `T`.
"""
function compound_period(x::Real, ::Type{T}) where {T <: Dates.Period}
    nf = Dates.value(convert(Dates.Nanosecond, T(1)))
    ns = Dates.Nanosecond(ceil(x * nf))
    return Dates.canonicalize(Dates.CompoundPeriod(ns))
end

trunc_time(s::String) = count(',', s) > 1 ? join(split(s, ",")[1:2], ",") : s


function prettymemory(b)
    if b < 1024
        return string(b, " bytes")
    elseif b < 1024^2
        value, units = b / 1024, "KiB"
    elseif b < 1024^3
        value, units = b / 1024^2, "MiB"
    else
        value, units = b / 1024^3, "GiB"
    end
    return "$(round(value, digits=2)) $units"
end

"""
    @timed_str expr
    @timed_str "description" expr

Returns a string containing `@timed` information.
"""
macro timed_str(ex)
    quote
        local stats = @timed $(esc(ex))
        "$(prettytime(stats.time*1e9)) ($(Base.gc_alloc_count(stats.gcstats)) allocations: $(prettymemory(stats.gcstats.allocd)))"
    end
end

struct AllNothing end
const all_nothing = AllNothing()
Base.getproperty(::AllNothing, ::Symbol) = nothing

"""
    horizontal_integral_at_boundary(f::Fields.Field, lev)

Compute the horizontal integral of a 2d or 3d `Fields.Field` `f` at a given vertical level
index `lev`. The underlying vertical space of the 2d field is required to be `FaceFiniteDifferenceSpace`.
"""
function horizontal_integral_at_boundary(f::Fields.Field, lev)
    @assert axes(f) isa Spaces.FaceExtrudedFiniteDifferenceSpace
    horizontal_integral_at_boundary(Spaces.level(f, lev))
end

function horizontal_integral_at_boundary(f::Fields.Field)
    @assert axes(f) isa Spaces.SpectralElementSpace2D
    sum(f ./ Fields.Δz_field(axes(f)) .* 2) # TODO: is there a way to ensure this is derived from face z? The 2d topology doesn't contain this info
end

"""
    isdivisible(dt_large::Dates.Period, dt_small::Dates.Period)

Check if two periods are evenly divisible, i.e., if the larger period can be
expressed as an integer multiple of the smaller period.

In this, take into account the case when periods do not have fixed size, e.g.,
one month is a variable number of days.

# Examples
```
julia> isdivisible(Dates.Year(1), Dates.Month(1))
true

julia> isdivisible(Dates.Month(1), Dates.Day(1))
true

julia> isdivisible(Dates.Month(1), Dates.Week(1))
false
```

## Notes

Not all the combinations are fully implemented. If something is missing, please
consider adding it.
"""
function isdivisible(dt_large::Dates.Period, dt_small::Dates.Period)
    @warn "The combination $(typeof(dt_large)) and $(dt_small) was not covered. Please add a method to handle this case."
    return false
end

# For FixedPeriod and OtherPeriod, it is easy, we can directly divide the two
# (as long as they are both the same)
function isdivisible(dt_large::Dates.FixedPeriod, dt_small::Dates.FixedPeriod)
    return isinteger(dt_large / dt_small)
end

function isdivisible(dt_large::Dates.OtherPeriod, dt_small::Dates.OtherPeriod)
    return isinteger(dt_large / dt_small)
end

function isdivisible(
    dt_large::Union{Dates.Month, Dates.Year},
    dt_small::Dates.FixedPeriod,
)
    # The only case where periods are commensurate for Month/Year is when we
    # have a Day or an integer divisor of a day. (Note that 365 and 366 don't
    # have any common divisor)
    return isinteger(Dates.Day(1) / dt_small)
end

"""
    promote_period(period::Dates.Period)

Promote a period to the largest possible period type.

This function attempts to represent a given `Period` using the largest possible
unit of time. For example, a period of 24 hours will be promoted to 1 day. If
a clean promotion is not possible, return the input as it is.

# Examples
```julia-repl
julia> promote_period(Hour(24))
1 day

julia> promote_period(Day(14))
2 weeks

julia> promote_period(Second(86401))
86401 seconds

julia> promote_period(Millisecond(1))
1 millisecond
```
"""
function promote_period(period::Dates.Period)
    ms = Int(Dates.toms(period))
    # Hard to do this with varying periods like Month/Year...
    PeriodTypes = [
        Dates.Week,
        Dates.Day,
        Dates.Hour,
        Dates.Minute,
        Dates.Second,
        Dates.Millisecond,
    ]
    for PeriodType in PeriodTypes
        period_ms = Int(Dates.toms(PeriodType(1)))
        if ms % period_ms == 0
            # Millisecond will always match, if nothing else matches
            return PeriodType(ms // period_ms)
        end
    end
end

function promote_period(period::Dates.OtherPeriod)
    # For varying periods, we just return them as they are
    return period
end

"""
    parse_date(date_str)

Parse a date string into a `DateTime` object. Currently, only the following formats are supported:
- yyyymmdd
- yyyymmdd-HHMM
"""
function parse_date(date_str)
    # Define a mapping between allowed formats and corresponding date format
    date_format_mapping = Dict(
        r"^\d{8}$" => dateformat"yyyymmdd",
        r"^\d{8}-\d{4}$" => dateformat"yyyymmdd-HHMM",
    )
    for (pattern, format) in date_format_mapping
        !isnothing(match(pattern, date_str)) &&
            return DateTime(date_str, format)
    end
    error(
        "Date string $date_str does not match any of the allowed formats: yyyymmdd or yyyymmdd-HHMM",
    )
end

iscolumn(space::Spaces.FiniteDifferenceSpace) = true
iscolumn(space) = false

function issphere(space)
    return Meshes.domain(Spaces.topology(Spaces.horizontal_space(space))) isa
           Domains.SphereDomain
end

function isbox(space)
    h_space = Spaces.horizontal_space(space)
    return Meshes.domain(Spaces.topology(h_space)) isa Domains.RectangleDomain
end

# Check if space is a single-column model (true column or minimal box used as column)
iscolumn_or_box(space) = iscolumn(space) || isbox(space)

"""
    clima_to_era5_name_dict()

Returns a dictionary mapping ClimaAtmos variable names to ERA5 variable names.
"""
function clima_to_era5_name_dict()
    Dict(
        "ua" => "u",
        "va" => "v",
        "wap" => "w", # era5 w is in Pa/s, this is confusing notation
        "hus" => "q",
        "ta" => "t",
        "zg" => "z", # era5 z is geopotential in m^2/s^2, this is confusing notation
        "clw" => "clwc",
        "cli" => "ciwc",
        "ts" => "skt",
        "hfls" => "slhf",
        "hfss" => "sshf",
    )
end
