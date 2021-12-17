struct BoundaryCondition{G, S}
    get_value!::G
    state::S
end
(bc::BoundaryCondition)(m, Y, cache, t) =
    bc.get_value!(bc.state, m, Y, cache, t)

# TODO: How to allow Dual numbers???

ZeroBoundary(::Type{T} = Float64) where {T} =
    BoundaryCondition((_, m, Y, cache, t) -> zero(T), nothing)
ConstantBoundary(value) =
    BoundaryCondition((_, m, Y, cache, t) -> value, nothing)
FixedBoundary(value) =
    BoundaryCondition((value, m, Y, cache, t) -> value, value)

#= TODO: Implement flux boundary conditions.
function Surface2AirBoundary(C, surface_temp)
    bot_air_temp = similar(surface_temp)
    flux = similar(surface_temp, Geometry.WVector{eltype(surface_temp)})
    FluxBoundaryCondition(
        (surface_temp, m, Y, cache, t) -> begin
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
