# pros:
# easy tests
# easy inspection
# high flexibility

# cons:
# very functional

abstract type AbstractModel end
abstract type AbstractEquation end
abstract type AbstractBoundaryCondition end
abstract type Rate{N} end
abstract type SteppingStyle end
abstract type AbstractDirection end
abstract type AbstractTendency end
abstract type AbstractConstraint end

# example of a concrete model
struct MyModel{EQ <: AbstractEquation, CO <: AbstractConstraint} <:
       AbstractModel
    eq1::EQ
    eq2::EQ
    eq3::EQ
    co1::CO
    co2::CO
end

# example of a concrete equation
struct Density{
    B <: AbstractBoundaryCondition,
    T <: AbstractTendency,
    C <: AbstractConstraint,
} <: AbstractEquation
    top::B
    btm::B
    advection::T
    diffusion::T
    positivity::T
end

# example of a concrete boundary condition
struct LatentHeatFlux <: AbstractBoundaryCondition
    bc_state # e.g., prescribed T_sfc
end

# example of concrete stepping styles
struct Explicit{R <: Rate{N}} <: SteppingStyle end
struct Implicit{R <: Rate{N}} <: SteppingStyle end

# example of concrete concrete directions
struct HorizontalDirection <: AbstractDirection end
struct VerticalDirection <: AbstractDirection end

# examples of a concrete tendencies
struct Advection{S <: SteppingStyle, D <: AbstractDirection} <: AbstractTendency end
struct Diffusion{S <: SteppingStyle, D <: AbstractDirection} <: AbstractTendency end
struct Coriolis{FT, S <: SteppingStyle} <: AbstractTendency
    rotation_rate::FT
end

# examples of a concrete constraints types
struct Positivity{R <: Rate} <: AbstractConstraint end
struct ThermodynamicEquilibrium{R <: Rate} <: AbstractConstraint end

# returns all equations
return_equations(model::AbstractModel)

# returns all tendencies
return_tendencies(model::AbstractModel, eq::AbstractEquation)

# returns all constraints
return_constraints(model::AbstractModel) # global constraints
return_constraints(model::AbstractModel, eq::AbstractEquation)

# mutates tendency state 
@inline (tend::AbstractTendency)(dY, Y, Yd, t, eq::AbstractEquation)
# alternatively: applyto!(dY, Y, D, t, tend::AbstractTendency, eq::AbstractEquation)

# returns the flux on the boundary
@inline (bc::AbstractBoundaryCondition)(Y, Yd, t, eq::AbstractEquation)
# alternatively: applyto!(dY, Y, D, t, bc::AbstractBoundaryCondition, eq::AbstractEquation)

# mutates prognostic and diagnostic state
@inline (con::AbstractConstraint)(Y, Yd, t, m::AbstractModel)
@inline (con::AbstractConstraint)(Y, Yd, t, eq::AbstractEquation)
# alternatively: applyto!(Y, D, t, con::AbstractConstraint, m::AbstractModel)

# returns the wrapped rhs function components for a model and stepping style (e.g., rhs for rate1, rate2, etc.)
function PDEFunction(
    model::AbstractModel,
    ss::SteppingStyle{R},
) where {R <: Rate{N}} end

# returns the wrapped rhs function components for a model and all stepping styles/rates
function PDEFunction(model::AbstractModel) end
