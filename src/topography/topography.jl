module Topography

using ClimaCore: Geometry

export topography_dcmip200, topography_hughes2023
export topography_agnesi, agnesi_params
export topography_schar, schar_params
export topography_cosine_2d, topography_cosine_3d
export topography_cosine, cosine_params

"""
    topography_dcmip200(coord)

Surface elevation for the DCMIP-2-0-0 test problem.
"""
function topography_dcmip200(coord)
    FT = Geometry.float_type(coord)
    λ, ϕ = coord.long, coord.lat
    ϕₘ = FT(0) # degrees (equator)
    λₘ = FT(3 / 2 * 180)  # degrees
    rₘ = FT(acos(sind(ϕₘ) * sind(ϕ) + cosd(ϕₘ) * cosd(ϕ) * cosd(λ - λₘ))) # Great circle distance (rads)
    Rₘ = FT(3π / 4) # Moutain radius
    ζₘ = FT(π / 16) # Mountain oscillation half-width
    h₀ = FT(2000)
    zₛ = ifelse(
        rₘ < Rₘ,
        FT(h₀ / 2) * (1 + cospi(rₘ / Rₘ)) * (cospi(rₘ / ζₘ))^2,
        FT(0),
    )
    return zₛ
end

"""
    topography_hughes2023(coord)

Surface elevation for baroclinic wave test from Hughes and Jablonowski (2023).
"""
function topography_hughes2023(coord)
    FT = Geometry.float_type(coord)
    λ, ϕ = coord.long, coord.lat
    h₀ = FT(2e3)
    # Angles in degrees
    ϕ₁ = FT(45)
    ϕ₂ = FT(45)
    λ_min = minimum(λ)
    λ₁ = FT(72)
    λ₂ = FT(140)
    λₘ = FT(7)
    ϕₘ = FT(40)
    d = ϕₘ / 2 * (-log(0.1))^(-1 / 6)
    c = λₘ / 2 * (-log(0.1))^(-1 / 2)
    d₁ = @. (λ - λ_min) - λ₁
    d₂ = @. (λ - λ_min) - λ₂
    l₁ = @. λ - λ₁
    l₂ = @. λ - λ₂
    zₛ = @. FT(
        h₀ * (
            exp(-(((ϕ - ϕ₁) / d)^6 + (l₁ / c)^2)) +
            exp(-(((ϕ - ϕ₂) / d)^6 + (l₂ / c)^2))
        ),
    )
end

##
## Topography profiles for 2D and 3D boxes
##

# The parameters of these profiles should be defined separately so that they
# can also be used to compute analytic solutions.

"""
    topography_agnesi(coord)

Surface elevation for a 2D Witch of Agnesi mountain, centered at `x = 50 km`.
"""
function topography_agnesi(coord)
    FT = Geometry.float_type(coord)
    (; x) = coord
    (; h_max, x_center, a) = agnesi_params(FT)
    return h_max / (1 + ((x - x_center) / a)^2)
end
agnesi_params(::Type{FT}) where {FT} =
    (; h_max = FT(25), x_center = FT(50e3), a = FT(5e3))

"""
    topography_schar(coord)

Surface elevation for a 2D Schar mountain, centered at `x = 50 km`.
"""
function topography_schar(coord)
    FT = Geometry.float_type(coord)
    (; x) = coord
    (; h_max, x_center, λ, a) = schar_params(FT)
    return h_max * exp(-(x - x_center)^2 / a^2) * cospi((x - x_center) / λ)^2
end
schar_params(::Type{FT}) where {FT} =
    (; h_max = FT(25), x_center = FT(50e3), λ = FT(4e3), a = FT(5e3))

"""
    topography_cosine_2d(coord)

Surface elevation for 2D cosine hills.
"""
function topography_cosine_2d(coord)
    FT = Geometry.float_type(coord)
    (; x) = coord
    (; h_max, λ) = cosine_params(FT)
    return topography_cosine(x, FT(0), λ, FT(Inf), h_max)
end

"""
    topography_cosine_3d(coord)

Surface elevation for 3D cosine hills.
"""
function topography_cosine_3d(coord)
    FT = Geometry.float_type(coord)
    (; x, y) = coord
    (; h_max, λ) = cosine_params(FT)
    return topography_cosine(x, y, λ, λ, h_max)
end

topography_cosine(x, y, λ_x, λ_y, h_max) =
    h_max * cospi(2 * x / λ_x) * cospi(2 * y / λ_y)
cosine_params(::Type{FT}) where {FT} = (; h_max = FT(25), λ = FT(25e3))

abstract type AbstractTopography end

topo_func(t::AbstractTopography) = error("No topography function for topography $t")

struct NoTopography <: AbstractTopography end

# Analytical topography types for idealized test cases

"""
    CosineTopography(; h_max = 25, λ = 25e3, dim = 2)

Cosine hill topography in 2D or 3D.

# Arguments
- `h_max::FT`: Maximum elevation (m)
- `λ::FT`: Wavelength of the cosine hills (m)
- `dim::Int`: Spatial dimension (2 or 3)
"""
Base.@kwdef struct CosineTopography <: AbstractTopography
    h_max = 25
    λ = 25e3
    dim::Int = 2
end

topo_func(t::CosineTopography) = t.dim == 2 ? topography_cosine_2d : topography_cosine_3d

"""
    AgnesiTopography(; h_max = 25, x_center = 50e3, a = 5e3)

Witch of Agnesi mountain topography for 2D simulations.

# Arguments
- `h_max`: Maximum elevation (m)
- `x_center`: Center position (m)
- `a`: Mountain width parameter (m)
"""
Base.@kwdef struct AgnesiTopography <: AbstractTopography
    h_max = 25
    x_center = 50e3
    a = 5e3
end

topo_func(::AgnesiTopography) = topography_agnesi


"""
    ScharTopography(; h_max = 25, x_center = 50e3, λ = 4e3, a = 5e3)

Schar mountain topography for 2D simulations.

# Arguments
- `h_max`: Maximum elevation (m)
- `x_center`: Center position (m)
- `λ`: Wavelength parameter (m)
- `a`: Mountain width parameter (m)
"""
Base.@kwdef struct ScharTopography <: AbstractTopography
    h_max = 25
    x_center = 50e3
    λ = 4e3
    a = 5e3
end
topo_func(::ScharTopography) = topography_schar


# Data-based topography types

"""
    EarthTopography()

Earth topography from ETOPO2022 data files.
"""
struct EarthTopography <: AbstractTopography end

"""
    DCMIP200Topography()

Surface elevation for the DCMIP-2-0-0 test problem.
"""
struct DCMIP200Topography <: AbstractTopography end

topo_func(::DCMIP200Topography) = topography_dcmip200

"""
    Hughes2023Topography()

Surface elevation for baroclinic wave test from Hughes and Jablonowski (2023).
"""
struct Hughes2023Topography <: AbstractTopography end

topo_func(::Hughes2023Topography) = topography_hughes2023

end
