"""
    NoFluxCondition <: AbstractFluxBoundaryCondition

    A boundary condition that implies no transport across the 
    boundary it is attached to.
"""
struct NoFluxCondition <: AbstractBoundaryCondition end

"""
    CustomFluxCondition <: AbstractFluxBoundaryCondition

    A boundary condition that applies a user-defined cross-boundary flux
    at the boundary it is attached to. The user is charged with making the 
    custom flux function consistent wth the numerics of the model that 
    invokes this boundary condition.
"""
struct CustomFluxCondition <: AbstractBoundaryCondition
    compute_flux::Function
end

"""
    DragLawCondition <: AbstractFluxBoundaryCondition
"""
struct DragLawCondition <: AbstractBoundaryCondition end
