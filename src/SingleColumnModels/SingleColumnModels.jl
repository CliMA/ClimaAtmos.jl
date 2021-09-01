module SingleColumnModels

using LinearAlgebra
using RecursiveArrayTools: ArrayPartition
using UnPack: @unpack

# clima ecosystem
using ClimaAtmos.BoundaryConditions:
    NoFluxCondition, CustomFluxCondition, DragLawCondition
using ClimaAtmos.Domains: AbstractVerticalDomain, make_function_space
using ClimaAtmos.Models: AbstractModel, get_boundary_flux
using ClimaCore: Fields, Geometry, Operators, Spaces

# we are extending here the Models interface by providing concrete implementions of models
import ClimaAtmos: Models

include("single_column_model.jl")

export SingleColumnModel

end # module
