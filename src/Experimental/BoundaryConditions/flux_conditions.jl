"""
    NoFlux <: AbstractBoundary

Computes a fixed boundary flux of 0.
"""
struct NoFlux <: AbstractBoundary end

"""
    NoVectorFlux <: AbstractBoundary

Computes a fixed boundary vector flux of 0
"""
struct NoVectorFlux <: AbstractBoundary end

"""
    DragLaw{C} <: AbstractBoundary

Computes the boundary flux using the bulk formula and constant or
Mohnin-Obukhov-based drag coefficient `Cd`. Specific to momentum density.
"""
struct DragLaw{C} <: AbstractBoundary
    coefficient::C
end

"""
    BulkFormula{C, T} <: AbstractBoundary

Computes the boundary flux using the bulk formula and constant or
Mohnin-Obukhov-based heat transfer coefficient `Ch`.
"""
struct BulkFormula{C, T} <: AbstractBoundary
    coefficients::C
    surface_field::T
end
