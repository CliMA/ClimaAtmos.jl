"""
    topography_dcmip200(λ,ϕ)
λ = longitude (degrees)
ϕ = latitude (degrees)
Given horizontal coordinates in lon-lat coordinates,
compute and return the local elevation of the surface
consistent with the test problem DCMIP-2-0-0.
"""
function topography_dcmip200(_, coord)
    λ, ϕ = coord.long, coord.lat
    FT = eltype(λ)
    ϕₘ = FT(0) # degrees (equator)
    λₘ = 3 * FT(180) / 2  # degrees
    rₘ = acos(sind(ϕₘ) * sind(ϕ) + cosd(ϕₘ) * cosd(ϕ) * cosd(λ - λₘ)) # Great circle distance (rads)
    Rₘ = 3 * FT(π) / 4 # Moutain radius
    ζₘ = FT(π) / 16 # Mountain oscillation half-width
    h₀ = FT(2000)
    zₛ = ifelse(
        rₘ < Rₘ,
        h₀ / 2 * (1 + cospi(rₘ / Rₘ)) * (cospi(rₘ / ζₘ))^2,
        FT(0),
    )
    return zₛ
end

function generate_topography_warp(earth_spline)
    function topography_earth(_, coord)
        λ, Φ = coord.long, coord.lat
        FT = eltype(λ)
        elevation = FT(earth_spline(λ, Φ))
        zₛ = ifelse(elevation > 0, elevation, FT(0))
        return zₛ
    end
    return topography_earth
end

# Additional topography functions with analytic Fourier transforms can be found
# in src/analytic_solutions/analytic_solutions.jl.
