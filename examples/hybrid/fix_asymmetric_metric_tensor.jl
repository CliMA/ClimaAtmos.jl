import ClimaCore: Geometry, DataLayouts
import LinearAlgebra: issymmetric

@inline function Geometry.LocalGeometry(
    coordinates,
    J,
    WJ,
    ∂x∂ξ::Geometry.Axis2Tensor{
        FT,
        Tuple{Geometry.LocalAxis{I}, Geometry.CovariantAxis{I}},
        S,
    },
) where {FT, I, S}
    ∂ξ∂x = inv(∂x∂ξ)
    C = typeof(coordinates)
    Jinv = inv(J)
    gⁱʲ = (∂ξ∂x * ∂ξ∂x' + (∂ξ∂x * ∂ξ∂x')') / 2
    gᵢⱼ = (∂x∂ξ' * ∂x∂ξ + (∂x∂ξ' * ∂x∂ξ)') / 2
    issymmetric(Geometry.components(gⁱʲ)) || error("gⁱʲ is not symmetric.")
    issymmetric(Geometry.components(gᵢⱼ)) || error("gᵢⱼ is not symmetric.")
    return DataLayouts.bypass_constructor(
        Geometry.LocalGeometry{I, C, FT, S},
        (coordinates, J, WJ, Jinv, ∂x∂ξ, ∂ξ∂x, gⁱʲ, gᵢⱼ),
    )
end
