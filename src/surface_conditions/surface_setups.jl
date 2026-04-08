# Surface types
"""
    PrescribedSurface()

Used to indicate that there is no surface parameterization and
that the surface conditions will be explicitly prescribed (e.g. by the coupler).
"""
struct PrescribedSurface end
(surface_setup::PrescribedSurface)(params) = nothing
"""
    DefaultMoninObukhov()

Monin-Obukhov surface, see the link below for more information
https://clima.github.io/SurfaceFluxes.jl/dev/SurfaceFluxes/#Monin-Obukhov-Similarity-Theory-(MOST)
"""
struct DefaultMoninObukhov end
function (::DefaultMoninObukhov)(params)
    FT = eltype(params)
    z0 = FT(1e-5)
    return SurfaceState(; parameterization = MoninObukhov(; z0))
end

"""
    DefaultExchangeCoefficients()

Bulk surface, parameterized only by a default exchange coefficient.
"""
struct DefaultExchangeCoefficients end
(::DefaultExchangeCoefficients)(params) =
    SurfaceState(; parameterization = ExchangeCoefficients(params.C_H))
