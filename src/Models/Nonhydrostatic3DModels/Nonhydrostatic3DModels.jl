module Nonhydrostatic3DModels

using LinearAlgebra
using StaticArrays
using UnPack
using CLIMAParameters
using Thermodynamics
using ClimaCore: Geometry, Spaces, Fields, Operators
using ClimaCore.Geometry: ⊗
using ...Domains, ...Models

export Nonhydrostatic3DModel

include("nonhydrostatic_3d_model.jl")

end # module
