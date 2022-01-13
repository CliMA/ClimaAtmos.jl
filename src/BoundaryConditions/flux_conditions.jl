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
Mohnin-Obukhov-based heat transfer coefficient `Ch`.
"""
struct BulkFormulaCondition{C, T} <: AbstractBoundaryCondition
    coefficients::C
    surface_field::T
end
