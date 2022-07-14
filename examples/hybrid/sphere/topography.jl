using NCDatasets
using Dierckx

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
    if rₘ < Rₘ
        zₛ = FT(h₀ / 2) * (1 + cospi(rₘ / Rₘ)) * (cospi(rₘ / ζₘ))^2
    else
        zₛ = FT(0)
    end
    return zₛ
end
