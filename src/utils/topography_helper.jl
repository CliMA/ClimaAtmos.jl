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
    h₀ = FT(200)
    zₛ = @. ifelse(
        rₘ < Rₘ,
        FT(h₀ / 2) * (1 + cospi(rₘ / Rₘ)) * (cospi(rₘ / ζₘ))^2,
        FT(0),
    )
    return zₛ
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
    h_c = FT(1000)
    a_c = FT(1000)
    x_c = FT(25600 - 6000)
    zₛ = @. h_c / (1 + ((x - x_c) / a_c)^2)
    return zₛ
end

"""
    topography_schar(x,z)
x = horizontal coordinate [m]
z = vertical coordinate [m]
h_c = 250 [m]
a_c = 5000 [m]
x_c = 30000 [m]
Assumes [0, 60] km domain.
Generate a single mountain profile (Schar mountain)
for use with tests of gravity waves with topography.
"""
function topography_schar(coords)
    x = coords.x
    FT = eltype(x)
    h_c = FT(250)
    λ_c = FT(4000)
    a_c = FT(5000)
    x_c = FT(30000)
    zₛ = @. h_c * exp(-((x - x_c) / a_c)^2) * (cospi((x - x_c) / λ_c))^2
    return zₛ
end
