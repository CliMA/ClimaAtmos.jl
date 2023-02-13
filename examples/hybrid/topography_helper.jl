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
