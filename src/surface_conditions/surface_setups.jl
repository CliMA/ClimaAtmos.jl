# Surface types
"""
    CouplerManagedSurface()

Sentinel indicating the coupler manages surface conditions externally.
ClimaAtmos will not compute surface fluxes when this is the active setup.
The coupler writes directly to `p.precomputed.sfc_conditions`.
"""
struct CouplerManagedSurface end
(::CouplerManagedSurface)(params) = CouplerManagedSurface()

"""
    PrescribedSurface

Deprecated: use [`CouplerManagedSurface`](@ref) instead.
"""
function PrescribedSurface()
    Base.depwarn(
        "`PrescribedSurface` is deprecated, use `CouplerManagedSurface` instead.",
        :PrescribedSurface,
    )
    return CouplerManagedSurface()
end
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
