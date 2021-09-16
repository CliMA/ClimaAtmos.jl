module Nonhydrostatic2DModels

using LinearAlgebra: norm, ×
using StaticArrays
using UnPack: @unpack

# clima ecosystem
using ClimaAtmos.Domains: AbstractHybridDomain, make_function_space
using ClimaAtmos.Models: AbstractModel
using ClimaCore: Fields, Geometry, Operators, Spaces
using ClimaCore.Geometry: ⊗

# we are extending here the Models interface by providing concrete implementions of models
import ClimaAtmos: Models

include("nonhydrostatic_2d_model.jl")

export Nonhydrostatic2DModel

end # module
