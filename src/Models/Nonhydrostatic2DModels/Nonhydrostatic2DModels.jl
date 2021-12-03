module Nonhydrostatic2DModels

using StaticArrays
using UnPack
using ClimaCore: Geometry, Spaces, Fields, Operators
using ClimaCore.Geometry: ⊗
using ...Domains, ...Models

export Nonhydrostatic2DModel

include("nonhydrostatic_2d_model.jl")

end # module
