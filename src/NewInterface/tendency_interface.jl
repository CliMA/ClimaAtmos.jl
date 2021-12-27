abstract type AbstractDirection end
struct HorizontalDirection <: AbstractDirection end
struct VerticalDirection <: AbstractDirection end
struct AllDirections <: AbstractDirection end

abstract type AbstractJacobianName end
struct DefaultFluidJacobian <: AbstractJacobianName end

abstract type AbstractTimesteppingMode end
struct Explicit <: AbstractTimesteppingMode end
Base.@kwdef struct Implicit{J <: Union{AbstractJacobianName, Nothing}} <:
                   AbstractTimesteppingMode
    jac_name::J = nothing
end

"""
    AbstractTendencyTerm{M <: AbstractTimesteppingMode}

Supertype for all tendency terms. All subtypes must contain the field `mode::M`.
"""
abstract type AbstractTendencyTerm{M <: AbstractTimesteppingMode} end

"""
    cache_reqs(tendency_term, vars)

Get the cache variables required by the specified tendency term, given the
mdoel's independent variables.
"""
function cache_reqs(::AbstractTendencyTerm, vars) end

"""
    (::AbstractTendencyTerm)(vars, Y, cache, consts, t)

Compute the value of the specified tendency term.

If the returned value involves dotted operations, it is recommended to use
`@lazydots` to delay the evaluation of those operations until an optimal time.
"""
function (::AbstractTendencyTerm)(vars, Y, cache, consts, t) end

"""
    Tendency{
        V <: Var,
        B <: AbstractBoundaryConditions,
        T <: NTuple{N, AbstractTendencyTerm} where {N},
    }

A representation of a "tendency" ``
    ∂ₜ\\text{var} =
    \\sum_{\\text{term} ∈ \\text{terms}}\\text{term}(vars, Y, cache, consts, t)
    \\bigr|_{\\text{bcs}}
``.

Provides the compiler with a way to map a variable to its boundary conditions
and tendency terms.

If the tendency has boundary conditions, it is recommended to avoid using
boundary conditions in any of its terms, since only the tendency's boundary
conditions will be applied.
"""
struct Tendency{
    V <: Var,
    B <: AbstractBoundaryConditions,
    T <: NTuple{N, AbstractTendencyTerm} where {N},
}
    var::V
    bcs::B
    terms::T
end

"""
    Tendency(var, [bcs], [terms...])

Recommended constructor for a `Tendency`.
"""
Tendency(var, bcs::AbstractBoundaryConditions, terms...) =
    Tendency(var, bcs, terms)
Tendency(var, terms...) = Tendency(var, NoBoundaryConditions(), terms)

cache_reqs(tendency::Tendency, vars) =
    cache_reqs_for_terms(vars, tendency.terms...)
cache_reqs_for_terms(vars) = ()
cache_reqs_for_terms(vars, term, terms...) =
    (cache_reqs(term, vars)..., cache_reqs_for_terms(vars, terms...)...)
