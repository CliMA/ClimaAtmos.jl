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
fast_pow(x::FT, y::FT) where {FT <: AbstractFloat} = exp(y * log(x))

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
    compute_kinetic!(κ::Field, uₕ::Field, uᵥ::Field)

Compute the specific kinetic energy at cell centers, storing in `κ` from
individual velocity components:
κ = 1/2 (uₕ⋅uʰ + 2uʰ⋅ᶜI(uᵥ) + ᶜI(uᵥ⋅uᵛ))
- `uₕ` should be a `Covariant1Vector` or `Covariant12Vector`-valued field at
    cell centers, and
- `uᵥ` should be a `Covariant3Vector`-valued field at cell faces.
"""
function compute_kinetic!(κ::Fields.Field, uₕ::Fields.Field, uᵥ::Fields.Field)
    @assert eltype(uₕ) <: Union{C1, C2, C12}
    @assert eltype(uᵥ) <: C3
    @. κ =
        1 / 2 * (
            dot(C123(uₕ), CT123(uₕ)) +
            ᶜinterp(dot(C123(uᵥ), CT123(uᵥ))) +
            2 * dot(CT123(uₕ), ᶜinterp(C123(uᵥ)))
        )
end

"""
    compute_kinetic!(κ::Field, Y::FieldVector)

Compute the specific kinetic energy at cell centers, storing in `κ`, where `Y`
is the model state.
"""
compute_kinetic!(κ::Fields.Field, Y::Fields.FieldVector) =
    compute_kinetic!(κ, Y.c.uₕ, Y.f.u₃)

"""
    compute_strain_rate_center!(ϵ::Field, u::Field)

Compute the strain_rate at cell centers, storing in `ϵ` from
velocity at cell faces.
"""
function compute_strain_rate_center!(ϵ::Fields.Field, u::Fields.Field)
    @assert eltype(u) <: C123
    axis_uvw = Geometry.UVWAxis()
    @. ϵ =
        (
            Geometry.project((axis_uvw,), ᶜgradᵥ(UVW(u))) +
            adjoint(Geometry.project((axis_uvw,), ᶜgradᵥ(UVW(u))))
        ) / 2
end

"""
    compute_strain_rate_face!(ϵ::Field, u::Field)

Compute the strain_rate at cell faces, storing in `ϵ` from
velocity at cell centers.
"""
function compute_strain_rate_face!(ϵ::Fields.Field, u::Fields.Field)
    @assert eltype(u) <: C123
    ∇ᵥuvw_boundary =
        Geometry.outer(Geometry.WVector(0), Geometry.UVWVector(0, 0, 0))
    ᶠgradᵥ = Operators.GradientC2F(
        bottom = Operators.SetGradient(∇ᵥuvw_boundary),
        top = Operators.SetGradient(∇ᵥuvw_boundary),
    )
    axis_uvw = Geometry.UVWAxis()
    @. ϵ =
        (
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

using ClimaComms
is_distributed(::ClimaComms.SingletonCommsContext) = false
is_distributed(::ClimaComms.MPICommsContext) = true

using Printf
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
    return @sprintf("%.3f %s", value, units)
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
    return @sprintf("%.2f %s", value, units)
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

"""
    dump_string(x)

Returns a string that contains the output of `dump(x)`.
"""
function dump_string(x)
    buffer = IOBuffer()
    dump(buffer, x)
    result = String(take!(buffer))
    close(buffer)
    return result
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
    gaussian_smooth(arr, sigma = 1)

Smooth the given 2D array by applying a Gaussian blur.

Edges are not properly smoothed out: the edge value is extended to infinity.
"""
function gaussian_smooth(arr::AbstractArray, sigma::Int = 1)
    n1, n2 = size(arr)

    # We assume that the Gaussian goes to zero at 10 sigma and ignore contributions outside of that window
    window = Int(ceil(10 * sigma))

    # Prepare the 2D Gaussian kernel
    gauss = [
        exp.(-(x .^ 2 .+ y .^ 2) / (2 * sigma^2)) for
        x in range(-window, window), y in range(-window, window)
    ]

    # Normalize it
    gauss = gauss / sum(gauss)

    smoothed_arr = zeros(size(arr))

    # 2D convolution
    for i in 1:n1
        for j in 1:n2
            # For each point, we "look left and right (up and down)" within our window
            for wx in (-window):window
                for wy in (-window):window
                    # For values at the edge, we keep using the edge value
                    k = clamp(i + wx, 1, n1)
                    l = clamp(j + wy, 1, n2)

                    # gauss has size 2window + 1, so its midpoint (when the gaussian is max)
                    # is at 1 + window
                    #
                    # Eg, for window of 3, wx will go through the values -3, -2, 1, 0, 1, 2, 3,
                    # and the midpoint is 4 (= 1 + window)
                    mid_gauss_idx = 1 + window

                    smoothed_arr[i, j] +=
                        arr[k, l] *
                        gauss[mid_gauss_idx + wx, mid_gauss_idx + wy]
                end
            end
        end
    end

    return smoothed_arr
end
