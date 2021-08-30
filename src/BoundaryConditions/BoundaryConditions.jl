module BoundaryConditions

"""
    AbstractFluxBoundaryCondition

    Abstract supertype for boundary conditions that imply fluxes
    at the boundaries they are attached to.
"""
abstract type AbstractBoundaryCondition end

include("flux_conditions.jl")

export AbstractBoundaryCondition
export CustomFluxCondition
export DragLawCondition
export NoFluxCondition

end # module
