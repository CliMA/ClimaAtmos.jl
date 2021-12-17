abstract type AbstractDirection end
struct HorizontalDirection <: AbstractDirection end
struct VerticalDirection <: AbstractDirection end
struct AllDirections <: AbstractDirection end

abstract type AbstractTendencyType end
struct Source <: AbstractTendencyType end
struct Flux{D <: AbstractDirection} <: AbstractTendencyType
    direction::D
end

abstract type AbstractJacobianName end
struct DefaultFluidJacobian <: AbstractJacobianName end

abstract type AbstractTimesteppingMode end
struct Explicit <: AbstractTimesteppingMode end
Base.@kwdef struct Implicit{
    J <: Union{AbstractJacobianName, Nothing},
} <: AbstractTimesteppingMode
    jac_name::J = nothing
end

"""
    AbstractTendencyTerm{M <: AbstractTimesteppingMode}

Supertype for all tendency terms. All subtypes must contain the field `mode::M`.
"""
abstract type AbstractTendencyTerm{M <: AbstractTimesteppingMode} end

"""
    tendency_type(tendency_term)

Get the `AbstractTendencyType` of the given tendency term.

This is implemented as a function rather than as type information so that a
single tendency term struct can have different tendency types depending on which
variables it is initialized with.
"""
function tendency_type(::AbstractTendencyTerm) end

"""
    cache_reqs(tendency_term, vars)

Get the cache variables required by the given tendency term, based on the state
variables.
"""
function cache_reqs(::AbstractTendencyTerm, vars) end

"""
    (::AbstractTendencyTerm)(vars, Y, cache, consts, t)

Compute the value of the given tendency term.

If the returned value involves dotted operations, it is recommended to use
`@lazydots` to delay the evaluation of those operations until an optimal time.
"""
function (::AbstractTendencyTerm)(vars, Y, cache, consts, t) end

"""
    Tendency{
        V <: Var,
        B <: AbstractBoundaryConditions,
        T <: NTuple{N, AbstractTendencyTerm} where N,
    }

A representation of a "tendency" ``
    ∂ₜ\\text{var} =
    \\sum_{\\text{term} ∈ \\text{terms}}\\text{term}(vars, Y, cache, consts, t)
    \\bigr|_{\\text{bcs}}
``.

Provides the compiler with a map from a variable to its boundary conditions and
tendency terms.
"""
struct Tendency{
    V <: Var,
    B <: AbstractBoundaryConditions,
    T <: NTuple{N, AbstractTendencyTerm} where N,
}
    var::V
    bcs::B
    terms::T
end
Tendency(var, bcs::AbstractBoundaryConditions, terms...) =
    Tendency(var, bcs, terms)
Tendency(var, terms...) = Tendency(var, NoBoundaryConditions(), terms)