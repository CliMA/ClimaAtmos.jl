module ShallowWaterModels

using LinearAlgebra: norm, ×
using UnPack
using ClimaCore: Geometry, Spaces, Fields, Operators
using ...Domains, ...Models

export ShallowWaterModel

include("shallow_water_model.jl")

end # module
