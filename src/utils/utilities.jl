#####
##### Utility functions
#####
import ClimaComms
import ClimaCore: Spaces, Topologies, Fields, Geometry
import LinearAlgebra: norm_sqr

is_energy_var(symbol) = symbol in (:ρe_tot, :ρae_tot)
is_momentum_var(symbol) = symbol in (:uₕ, :ρuₕ, :u₃, :ρw)
is_turbconv_var(symbol) = symbol in (:turbconv, :sgsʲs, :sgs⁰)
is_tracer_var(symbol) = !(
    symbol == :ρ ||
    symbol == :ρa ||
    is_energy_var(symbol) ||
    is_momentum_var(symbol) ||
    is_turbconv_var(symbol)
)

# we may be hitting a slow path:
# https://stackoverflow.com/questions/14687665/very-slow-stdpow-for-bases-very-close-to-1
fast_pow(x, y) = exp(y * log(x))

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
    bc_kinetic = compute_kinetic(uₕ::Field, uᵥ::Field)
    @. κ = bc_kinetic

Compute the specific kinetic energy at cell centers, resulting in `κ` from
individual velocity components:

 - `κ = 1/2 (uₕ⋅uʰ + 2uʰ⋅ᶜI(uᵥ) + ᶜI(uᵥ⋅uᵛ))`
 - `uₕ` should be a `Covariant1Vector` or `Covariant12Vector`-valued field at
    cell centers, and
 - `uᵥ` should be a `Covariant3Vector`-valued field at cell faces.
"""
function compute_kinetic(uₕ::Fields.Field, uᵥ::Fields.Field)
    @assert eltype(uₕ) <: Union{C1, C2, C12}
    @assert eltype(uᵥ) <: C3
    return @lazy @. 1 / 2 * (
        dot(C123(uₕ), CT123(uₕ)) +
        ᶜinterp(dot(C123(uᵥ), CT123(uᵥ))) +
        2 * dot(CT123(uₕ), ᶜinterp(C123(uᵥ)))
    )
end

"""
    compute_kinetic(Y::FieldVector)

Compute the specific kinetic energy at cell centers, where `Y` is the model
state.
"""
compute_kinetic(Y::Fields.FieldVector) = compute_kinetic(Y.c.uₕ, Y.f.u₃)

"""
    bc_ϵ = compute_strain_rate_center(u::Field)
    @. ϵ = bc_ϵ

Compute the strain_rate at cell centers from velocity at cell faces.
"""
function compute_strain_rate_center(u::Fields.Field)
    @assert eltype(u) <: C123
    axis_uvw = Geometry.UVWAxis()
    return @lazy @. (
        Geometry.project((axis_uvw,), ᶜgradᵥ(UVW(u))) +
        adjoint(Geometry.project((axis_uvw,), ᶜgradᵥ(UVW(u))))
    ) / 2
end

"""
    bc_ϵ = compute_strain_rate_face(u::Field)
    @. ϵ = bc_ϵ

Compute the strain_rate at cell faces from velocity at cell centers.
"""
function compute_strain_rate_face(u::Fields.Field)
    @assert eltype(u) <: C123
    ∇ᵥuvw_boundary =
        Geometry.outer(Geometry.WVector(0), Geometry.UVWVector(0, 0, 0))
    ᶠgradᵥ = Operators.GradientC2F(
        bottom = Operators.SetGradient(∇ᵥuvw_boundary),
        top = Operators.SetGradient(∇ᵥuvw_boundary),
    )
    axis_uvw = Geometry.UVWAxis()
    return @lazy @. (
        Geometry.project((axis_uvw,), ᶠgradᵥ(UVW(u))) +
        adjoint(Geometry.project((axis_uvw,), ᶠgradᵥ(UVW(u))))
    ) / 2
end

"""
    g³³_field(field)

Extracts the value of `g³³` from `Fields.local_geometry_field(field)`.
"""
function g³³_field(field)
    g_field = Fields.local_geometry_field(field).gⁱʲ.components.data
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
    CTh_axis = if full_CT_axis == Geometry.Contravariant123Axis()
        Geometry.Contravariant12Axis()
    elseif full_CT_axis == Geometry.Contravariant13Axis()
        Geometry.Contravariant1Axis()
    elseif full_CT_axis == Geometry.Contravariant23Axis()
        Geometry.Contravariant2Axis()
    else
        error("$full_CT_axis is missing either vertical or horizontal sub-axes")
    end
    N = length(full_CT_axis)
    return Geometry.AxisTensor(
        (Geometry.Contravariant3Axis(), CTh_axis),
        view(Geometry.components(gⁱʲ), N:N, 1:(N - 1)),
    )
end

"""
    CTh_vector_type(space)

Extracts the (abstract) horizontal contravariant vector type from the given
`AbstractSpace`.
"""
function CTh_vector_type(space)
    full_CT_axis = axes(eltype(Fields.local_geometry_field(space).gⁱʲ))[1]
    return if full_CT_axis == Geometry.Contravariant123Axis()
        Geometry.Contravariant12Vector
    elseif full_CT_axis == Geometry.Contravariant13Axis()
        Geometry.Contravariant1Vector
    elseif full_CT_axis == Geometry.Contravariant23Axis()
        Geometry.Contravariant2Vector
    else
        error("$full_CT_axis is missing either vertical or horizontal sub-axes")
    end
end

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

function projected_vector_buoy_grad_vars(::Type{V}, v1, v2, v3, lg) where {V}
    ubvd = unit_basis_vector_data(V, lg)
    return (;
        ∂θv∂z_unsat = V(v1, lg)[1] / ubvd,
        ∂qt∂z_sat = V(v2, lg)[1] / ubvd,
        ∂θl∂z_sat = V(v3, lg)[1] / ubvd,
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

function time_to_seconds(s::String)
    factor = Dict(
        "secs" => 1,
        "mins" => 60,
        "hours" => 60 * 60,
        "days" => 60 * 60 * 24,
    )
    s == "Inf" && return Inf
    if count(occursin.(keys(factor), Ref(s))) != 1
        error(
            "Bad format for flag $s. Examples: [`10secs`, `20mins`, `30hours`, `40days`]",
        )
    end
    for match in keys(factor)
        occursin(match, s) || continue
        return parse(Float64, first(split(s, match))) * factor[match]
    end
    error("Uncaught case in computing time from given string.")
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


using ClimaComms
is_distributed(::ClimaComms.SingletonCommsContext) = false
is_distributed(::ClimaComms.MPICommsContext) = true

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

function iscolumn(space)
    # TODO: Our columns are 2+1D boxes with one element at the base. Fix this
    isbox =
        Meshes.domain(Spaces.topology(Spaces.horizontal_space(space))) isa
        Domains.RectangleDomain
    isbox || return false
    has_one_element =
        Meshes.nelements(
            Spaces.topology(Spaces.horizontal_space(space)).mesh,
        ) == 1
    has_one_element && return true
end

function issphere(space)
    return Meshes.domain(Spaces.topology(Spaces.horizontal_space(space))) isa
           Domains.SphereDomain
end
