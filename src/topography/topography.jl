using ClimaCore: Geometry, Spaces, Fields
export CosineTopography,
    AgnesiTopography,
    ScharTopography,
    EarthTopography,
    DCMIP200Topography,
    Hughes2023Topography,
    LinearWarp, SLEVEWarp

##
## Topography profiles for 2D and 3D boxes
##

# The parameters of these profiles should be defined separately so that they
# can also be used to compute analytic solutions.

abstract type AbstractTopography end
Base.broadcastable(t::AbstractTopography) = tuple(t)
topography_function(topo) = Base.Fix1(topography_function, topo)

struct NoTopography <: AbstractTopography end

# Analytical topography types for idealized test cases

"""
    CosineTopography{D, FT}(; h_max = 25, λ = 25e3)
    
Cosine hill topography in 2D or 3D.

# Arguments
- `h_max::FT`: Maximum elevation (m)
- `λ::FT`: Wavelength of the cosine hills (m)
"""
Base.@kwdef struct CosineTopography{D, FT} <: AbstractTopography
    h_max::FT = 25.0
    λ::FT = 25e3
end

topography_function(t::CosineTopography{2}, coord) =
    topography_cosine(coord.x, zero(coord.x), t.λ, oftype(t.λ, Inf), t.h_max)

topography_function(t::CosineTopography{3}, coord) =
    topography_cosine(coord.x, coord.y, t.λ, t.λ, t.h_max)

topography_cosine(x, y, λ_x, λ_y, h_max) =
    h_max * cospi(2 * x / λ_x) * cospi(2 * y / λ_y)

"""
    AgnesiTopography{FT}(; h_max = 25, x_center = 50e3, a = 5e3)

Witch of Agnesi mountain topography for 2D simulations.

# Arguments
- `h_max`: Maximum elevation (m)
- `x_center`: Center position (m)
- `a`: Mountain width parameter (m)
"""
Base.@kwdef struct AgnesiTopography{FT} <: AbstractTopography
    h_max::FT = 25.0
    x_center::FT = 50e3
    a::FT = 5e3
end

topography_function((; h_max, x_center, a)::AgnesiTopography, (; x)) =
    h_max / (1 + ((x - x_center) / a)^2)

"""
    ScharTopography{FT}(; h_max = 25, x_center = 50e3, λ = 4e3, a = 5e3)

Schar mountain topography for 2D simulations.

# Arguments
- `h_max`: Maximum elevation (m)
- `x_center`: Center position (m)
- `λ`: Wavelength parameter (m)
- `a`: Mountain width parameter (m)
"""
Base.@kwdef struct ScharTopography{FT} <: AbstractTopography
    h_max::FT = 25.0
    x_center::FT = 50e3
    λ::FT = 4e3
    a::FT = 5e3
end

topography_function((; h_max, x_center, λ, a)::ScharTopography, (; x)) =
    h_max * exp(-(x - x_center)^2 / a^2) * cospi((x - x_center) / λ)^2

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

topography_function(::DCMIP200Topography, coord) = topography_dcmip200(coord)

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
    Hughes2023Topography()

Surface elevation for baroclinic wave test from Hughes and Jablonowski (2023).
"""
struct Hughes2023Topography <: AbstractTopography end

topography_function(::Hughes2023Topography, coord) = topography_hughes2023(coord)

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
    d₁ = (λ - λ_min) - λ₁
    d₂ = (λ - λ_min) - λ₂
    l₁ = λ - λ₁
    l₂ = λ - λ₂
    zₛ = FT(
        h₀ * (
            exp(-(((ϕ - ϕ₁) / d)^6 + (l₁ / c)^2)) +
            exp(-(((ϕ - ϕ₂) / d)^6 + (l₂ / c)^2))
        ),
    )
end

##
## Mesh warping types for topography
##

abstract type MeshWarpType end

"""
    LinearWarp()

Linear mesh warping that uniformly distributes vertical levels between the
surface and top of the domain.
"""
struct LinearWarp <: MeshWarpType end

"""
    SLEVEWarp(; eta = 0.7, s = 10.0)

Smooth Level Vertical (SLEVE) coordinate warping for terrain-following meshes.

# Arguments
- `eta`: Threshold parameter (if z/z_top > eta, no warping is applied). Default: 0.7
- `s`: Decay scale parameter controlling how quickly the warping decays with height. Default: 10.0

# References
Schär et al. (2002), "A new terrain-following vertical coordinate formulation 
for atmospheric prediction models", Mon. Wea. Rev.
"""
Base.@kwdef struct SLEVEWarp{FT <: AbstractFloat} <: MeshWarpType
    eta::FT = 0.7
    s::FT = 10.0
end
