"""
    topography_dcmip200(λ,ϕ)
λ = longitude (degrees)
ϕ = latitude (degrees)
Given horizontal coordinates in lon-lat coordinates,
compute and return the local elevation of the surface
consistent with the test problem DCMIP-2-0-0.
"""
function topography_dcmip200(coords)
    λ, ϕ = coords.long, coords.lat
    FT = eltype(λ)
    ϕₘ = FT(0) # degrees (equator)
    λₘ = FT(3 / 2 * 180)  # degrees
    rₘ = @. FT(acos(sind(ϕₘ) * sind(ϕ) + cosd(ϕₘ) * cosd(ϕ) * cosd(λ - λₘ))) # Great circle distance (rads)
    Rₘ = FT(3π / 4) # Moutain radius
    ζₘ = FT(π / 16) # Mountain oscillation half-width
    h₀ = FT(2000)
    zₛ = @. ifelse(
        rₘ < Rₘ,
        FT(h₀ / 2) * (1 + cospi(rₘ / Rₘ)) * (cospi(rₘ / ζₘ))^2,
        FT(0),
    )
    return zₛ
end

"""
    topography_hughes2023(λ,ϕ)
λ = longitude (degrees)
ϕ = latitude (degrees)
Returns the surface elevation profile used in the baroclinic wave
test problem defined by Hughes and Jablonowski (2023).
"""
function topography_hughes2023(coords)
    λ, ϕ = coords.long, coords.lat
    FT = eltype(λ)
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

function generate_topography_warp(earth_spline)
    function topography_earth(coords)
        λ, Φ = coords.long, coords.lat
        FT = eltype(λ)
        @info "Load spline"
        elevation = @. FT(earth_spline(λ, Φ))
        zₛ = @. ifelse(elevation > FT(0), elevation, FT(0))
        @info "Assign elevation"
        return zₛ
    end
    return topography_earth
end

"""
    topography_agnesi(x,z)
x = horizontal coordinate [m]
z = vertical coordinate [m]
h_c = 1 [m]
a_c = 10000 [m]
x_c = 120000 [m]
Generate a single mountain profile (Agnesi mountain)
for use with tests of gravity waves with topography.
"""
function topography_agnesi(coords)
    x = coords.x
    FT = eltype(x)
    h_c = FT(1)
    a_c = FT(10000)
    x_c = FT(120000)
    zₛ = @. h_c / (1 + ((x - x_c) / a_c)^2)
    return zₛ
end

"""
    topography_schar(x,z)
x = horizontal coordinate [m]
z = vertical coordinate [m]
h_c = 250 [m]
a_c = 5000 [m]
x_c = 60000 [m]
Assumes [0, 120] km domain.
Generate a single mountain profile (Schar mountain)
for use with tests of gravity waves with topography.
"""
function topography_schar(coords)
    x = coords.x
    FT = eltype(x)
    h_c = FT(250)
    λ_c = FT(4000)
    a_c = FT(5000)
    x_c = FT(60000)
    zₛ = @. h_c * exp(-((x - x_c) / a_c)^2) * (cospi((x - x_c) / λ_c))^2
    return zₛ
end
