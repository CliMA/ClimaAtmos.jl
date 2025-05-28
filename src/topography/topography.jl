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
