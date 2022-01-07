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
include("equations_gravitational_potential.jl")
include("equations_pressure.jl")
include("equations_base_model.jl")
include("equations_thermodynamics.jl")
include("equations_moisture.jl")

end # module
