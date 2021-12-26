struct BoundaryCondition{G, S}
    get_value!::G
    state::S
end
(bc::BoundaryCondition)(vars, Y, cache, consts, t) =
    bc.get_value!(bc.state, vars, Y, cache, consts, t)

# TODO: How to allow Dual numbers???
# Since BCs correspond to tendencies, they must be Neumann.
# Also, they only apply to finite difference operators. In the future, we may
# add support for other operators.

ZeroBoundary(::Type{T} = Float64) where {T} =
    BoundaryCondition((_, vars, Y, cache, consts, t) -> zero(T), nothing)
ConstantBoundary(value) =
    BoundaryCondition((value, vars, Y, cache, consts, t) -> value, value)

#= TODO: Implement flux boundary conditions.
function Surface2AirBoundary(C, surface_temp)
    bot_air_temp = similar(surface_temp)
    flux = similar(surface_temp, Geometry.WVector{eltype(surface_temp)})
    FluxBoundaryCondition(
        (surface_temp, vars, Y, cache, consts, t) -> begin
            # TODO: iterate over Spaces.eachslabindex to set bot_air_temp
            @. flux = Geometry.WVector(C * (surface_temp - bot_air_temp))
        end,
        surface_temp,
    )
end
=#

abstract type AbstractBoundaryConditions end
struct NoBoundaryConditions <: AbstractBoundaryConditions end
struct VerticalBoundaryConditions{
    B <: BoundaryCondition,
    T <: BoundaryCondition,
} <: AbstractBoundaryConditions
    bottom::B
    top::T
end
