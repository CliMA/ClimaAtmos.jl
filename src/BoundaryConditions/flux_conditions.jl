"""
    CustomFluxCondition{F} <: AbstractBoundaryCondition

Computes a user-defined boundary flux. The user is charged with making the
custom flux function consistent wth the numerics of the model that invokes this
boundary condition.
"""
struct CustomFluxCondition{F <: Function} <: AbstractBoundaryCondition
    compute_flux::F
end

"""
    NoFluxCondition <: AbstractBoundaryCondition

Computes a fixed boundary flux of 0.
"""
struct NoFluxCondition <: AbstractBoundaryCondition end

"""
    DragLawCondition{C} <: AbstractBoundaryCondition

Computes the boundary flux using the bulk formula and constant or
Mohnin-Obukhov-based drag coefficient `Cd`. Specific to momentum density.
"""
struct DragLawCondition{C} <: AbstractBoundaryCondition
    coefficients::C
end

"""
    BulkFormulaCondition{C, T} <: AbstractBoundaryCondition

Computes the boundary flux using the bulk formula and constant or
Mohnin-Obukhov-based heat transfer coefficient `Ch`. Specific to potential
temperature density.
"""
struct BulkFormulaCondition{C, T} <: AbstractBoundaryCondition
    coefficients::C
    θ_sfc::T
end
