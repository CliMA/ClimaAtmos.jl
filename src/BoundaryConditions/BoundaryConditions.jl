module BoundaryConditions

"""
    AbstractFluxBoundaryCondition

    Abstract supertype for boundary conditions that imply fluxes
    at the boundaries they are attached to.
"""
abstract type AbstracBoundaryCondition end

include("flux_conditions.jl")

export AbstracBoundaryCondition
export CustomFluxCondition
export DragLawCondition
export NoFluxCondition

end # module