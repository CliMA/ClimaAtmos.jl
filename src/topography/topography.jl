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

"""
    AbstractTopography

Surface elevation profile used to warp the vertical grid.

Subtypes:

  - `NoTopography`: flat surface; the grid is not warped.
  - [`CosineTopography`](@ref): periodic cosine hills, in 2D or 3D.
  - [`AgnesiTopography`](@ref): a single witch-of-Agnesi mountain, 2D.
  - [`ScharTopography`](@ref): a Gaussian envelope of cosine ridges, 2D.
  - [`EarthTopography`](@ref): Earth orography from the ETOPO2022 dataset.
  - [`DCMIP200Topography`](@ref): the DCMIP-2-0-0 mountain, on the sphere.
  - [`Hughes2023Topography`](@ref): the two-ridge mountain of Hughes and
    Jablonowski (2023), on the sphere.

Every analytic subtype extends `topography_function(topography, coord)`, which
returns the surface elevation [m] at `coord`. `EarthTopography` has no analytic
form and is instead read from a file when the grid is built; `NoTopography`
short-circuits grid warping entirely. The parameters live on the type rather
than inside the elevation function so that the analytic steady-state solutions
in `steady_state_solutions.jl` can reuse them.
"""
abstract type AbstractTopography end
Base.broadcastable(t::AbstractTopography) = tuple(t)

"""
    topography_function(topography)
    topography_function(topography, coord)

Return the surface elevation [m] of `topography` at `coord`, or, given only
`topography`, the callable `coord -> elevation` that a `SpaceVaryingInput` uses
to fill the surface elevation field.
"""
topography_function(topo) = Base.Fix1(topography_function, topo)

"""
    NoTopography()

Flat lower boundary: the vertical grid is built without hypsography, so the
mesh-warping choice has no effect.
"""
struct NoTopography <: AbstractTopography end

# Analytical topography types for idealized test cases

"""
    CosineTopography{D, FT}(; h_max = 25, λ = 25e3)

Periodic cosine hills in a 2D (`D = 2`) or 3D (`D = 3`) box.

The elevation is `h_max cos(2πx/λ)` in 2D and `h_max cos(2πx/λ) cos(2πy/λ)` in
3D, so the same wavelength is used along both horizontal directions. Steady-state
solutions for this profile are available from `steady_state_velocity`.

# Fields

  - `h_max = 25`: Amplitude of the hills, the maximum elevation [m].
  - `λ = 25e3`: Wavelength of the hills [m].

# Examples

```julia
topography = CosineTopography{2, Float64}(; h_max = 100, λ = 20e3)
```
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

Witch-of-Agnesi mountain for 2D simulations.

The elevation is ``h_{max} / (1 + ((x - x_c)/a)^2)``, the standard profile for
mountain-wave tests. Steady-state solutions for this profile are available from
`steady_state_velocity`.

# Fields

  - `h_max = 25`: Peak elevation [m].
  - `x_center = 50e3`: Horizontal position of the peak [m].
  - `a = 5e3`: Half-width of the mountain [m].

# Examples

```julia
topography = AgnesiTopography{Float64}(; h_max = 400, a = 10e3)
```
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

Schär mountain for 2D simulations: cosine ridges of wavelength `λ` under a
Gaussian envelope of half-width `a`.

The elevation is ``h_{max} \\exp(-((x - x_c)/a)^2) \\cos^2(π (x - x_c)/λ)``, so
the profile carries both a resolved-scale and a small-scale response.
Steady-state solutions for this profile are available from
`steady_state_velocity`.

# Fields

  - `h_max = 25`: Peak elevation [m].
  - `x_center = 50e3`: Horizontal position of the central peak [m].
  - `λ = 4e3`: Wavelength of the ridges [m].
  - `a = 5e3`: Half-width of the Gaussian envelope [m].

# Examples

```julia
topography = ScharTopography{Float64}(; h_max = 250, λ = 4e3, a = 5e3)
```
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

Earth orography, regridded from the ETOPO2022 ice-surface elevation dataset.

Unlike the analytic profiles, this one has no `topography_function`: the
elevation is read from the `earth_orography` artifact onto the horizontal space
when the grid is built, then smoothed by horizontal diffusion (the number of
iterations follows the `topography_damping_factor` configuration) and clipped
at zero. See the [Topography in ClimaAtmos](@ref "Topography in ClimaAtmos")
page.
"""
struct EarthTopography <: AbstractTopography end

"""
    DCMIP200Topography()

Surface elevation for the DCMIP-2-0-0 test problem: a 2 km circular mountain
centered on the equator at 270° longitude, on the sphere.

Inside a great-circle radius of 3π/4, the elevation is a cosine bell modulated
by cosine ridges of half-width π/16; outside it is zero.
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

Surface elevation for the baroclinic-wave test of Hughes and Jablonowski
(2023): two 2 km ridges centered at 45°N, at 72° and 140° longitude, on the
sphere.

Each ridge is a super-Gaussian in latitude and a Gaussian in longitude, with
widths set so that the elevation falls to a tenth of its peak at 20° in latitude
and 3.5° in longitude.

# References

Hughes, O. K. and Jablonowski, C. (2023), "A Mountain-Induced Moist Baroclinic
Wave Test Case for the Dynamical Cores of Atmospheric General Circulation
Models", Mon. Wea. Rev.
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

"""
    MeshWarpType

Strategy for warping the vertical grid to follow the surface elevation.

Subtypes:

  - [`LinearWarp`](@ref): terrain following at the surface, decaying linearly to
    flat at the model top.
  - [`SLEVEWarp`](@ref): smooth-level vertical coordinate, decaying the
    small-scale terrain faster than the large-scale terrain.

Has no effect when the topography is `NoTopography`.
"""
abstract type MeshWarpType end

"""
    LinearWarp()

Terrain-following warping in which the terrain influence decays linearly with
height, vanishing at the top of the domain.
"""
struct LinearWarp <: MeshWarpType end

"""
    SLEVEWarp(; eta = 0.7, s = 10.0)

Smooth Level Vertical (SLEVE) coordinate warping for terrain-following meshes.

The terrain influence decays like `sinh((ηₕ - η) / (s ηₕ)) / sinh(1 / s)` in the
normalized height `η = z / z_top`, so levels relax to flat faster than the
linear decay of [`LinearWarp`](@ref).

# Fields

  - `eta = 0.7`: Normalized height `ηₕ` above which no warping is applied, i.e.
    levels with `z / z_top > eta` are flat [-].
  - `s = 10.0`: Decay scale as a fraction of the domain height; smaller values
    confine the terrain influence closer to the surface [-]. Grid construction
    errors unless `s * z_top` exceeds the maximum surface elevation.

# References

Schär et al. (2002), "A new terrain-following vertical coordinate formulation
for atmospheric prediction models", Mon. Wea. Rev.
"""
Base.@kwdef struct SLEVEWarp{FT <: AbstractFloat} <: MeshWarpType
    eta::FT = 0.7
    s::FT = 10.0
end
