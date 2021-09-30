module ShallowWaterModels

using LinearAlgebra: norm, ×
using UnPack: @unpack

# clima ecosystem
using ClimaAtmos.Domains: AbstractHorizontalDomain, make_function_space
using ClimaAtmos.Models: AbstractModel
using ClimaCore: Fields, Geometry, Operators, Spaces

# we are extending here the Models interface by providing concrete implementions of models
import ClimaAtmos: Models

include("shallow_water_model.jl")

export ShallowWaterModel
export get_velocities

end # module
