module Models

using LinearAlgebra: norm, ×
using RecursiveArrayTools: ArrayPartition
using UnPack: @unpack

import ClimaAtmos.BoundaryConditions: NoFluxCondition, CustomFluxCondition, DragLawCondition
import ClimaAtmos.Domains: AbstractHorizontalDomain, AbstractVerticalDomain
import ClimaAtmos.Domains: make_function_space
import ClimaCore:
    Fields, 
    Geometry,
    Operators,
    Spaces
import ClimaCore.Spaces:
    FiniteDifferenceSpace,
    SpectralElementSpace2D

"""
    AbstractEquationSplitting
"""
abstract type AbstractEquationSplitting end

"""
    AbstractModel
"""
abstract type AbstractModel end

include("splitting.jl")
include("shallow_water_models/shallow_water_model.jl")
include("single_column_models/single_column_model.jl")

export AbstractEquationSplitting
export AbstractModel
export NoEquationSplitting
export ShallowWaterModel
export SingleColumnModel

export default_initial_conditions
export make_ode_function

end # module