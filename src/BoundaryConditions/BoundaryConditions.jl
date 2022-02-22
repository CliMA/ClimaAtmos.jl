module BoundaryConditions

using LinearAlgebra: norm
using ClimaCore: Geometry, Fields, Operators
using ClimaCore.Geometry: ⊗
using SurfaceFluxes

export get_boundary_flux,
    AbstractBoundaryCondition,
    CustomFluxCondition,
    NoFluxCondition,
    DragLawCondition,
    BulkFormulaCondition

"""
    get_boundary_flux(model, bc, var, Ym, Ya)

Get the flux of variable `var` across the boundary of `model` using boundary
condition `bc`.
"""
function get_boundary_flux end

"""
Supertype for all boundary conditions.
"""
abstract type AbstractBoundaryCondition end

include("flux_conditions.jl")
include("flux_calculations.jl")

end # module
