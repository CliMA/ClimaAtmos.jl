"""
    DefaultMoninObukhov()

Monin-Obukhov surface with a default roughness length
(see https://clima.github.io/SurfaceFluxes.jl/dev/SurfaceFluxes/#Monin-Obukhov-Similarity-Theory-(MOST)).
"""
struct DefaultMoninObukhov end
function (::DefaultMoninObukhov)(params)
    FT = eltype(params)
    return MoninObukhov(; z0 = FT(1e-5))
end

"""
    DefaultExchangeCoefficients()

Bulk surface, parameterized only by a default exchange coefficient.
"""
struct DefaultExchangeCoefficients end
(::DefaultExchangeCoefficients)(params) = ExchangeCoefficients(params.C_H)
