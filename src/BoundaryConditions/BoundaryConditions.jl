module BoundaryConditions

using LinearAlgebra: norm
using ClimaCore: Geometry, Fields, Operators
using ClimaCore.Geometry: ⊗
using ClimaAtmos

import SurfaceFluxes
const SF = SurfaceFluxes
const UF = SurfaceFluxes.UniversalFunctions
import Thermodynamics
const TD = Thermodynamics

export get_boundary_flux,
    AbstractBoundary, NoFlux, NoVectorFlux, DragLaw, BulkFormulaDryTotalEnergy

"""
    get_boundary_flux(model, bc, var, Ym, Ya)

Get the flux of variable `var` across the boundary of `model` using boundary
condition `bc`.
"""
function get_boundary_flux end

"""
Supertype for all boundary conditions.
"""
abstract type AbstractBoundary end

include("flux_conditions.jl")
include("flux_calculations.jl")

end # module
