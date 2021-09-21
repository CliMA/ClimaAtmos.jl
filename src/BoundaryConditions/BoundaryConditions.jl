module BoundaryConditions

using ClimaCore: Geometry, Fields, Operators, Spaces
using ClimaCore.Geometry: ⊗
using LinearAlgebra
using LinearAlgebra: norm, ×

"""
    AbstractFluxBoundaryCondition
    
    Abstract supertype for boundary conditions that imply fluxes
    at the boundaries they are attached to.
"""
abstract type AbstractBoundaryCondition end

include("flux_conditions.jl")
include("flux_calculations.jl")

export AbstractBoundaryCondition
export CustomFluxCondition
export DragLawCondition
export NoFluxCondition
export BulkFormulaCondition
export get_boundary_flux

end # module
