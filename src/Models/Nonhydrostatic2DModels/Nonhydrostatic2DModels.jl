module Nonhydrostatic2DModels

using StaticArrays
using UnPack
using CLIMAParameters
using Thermodynamics
using CloudMicrophysics
using ClimaCore: Geometry, Spaces, Fields, Operators
using ClimaCore.Geometry: ⊗
using ...Domains, ...Models

using LinearAlgebra: norm_sqr

export Nonhydrostatic2DModel

include("nonhydrostatic_2d_model.jl")
include("equations_pressure.jl")
include("equations_base_model.jl")
include("equations_thermodynamics.jl")
include("equations_moisture.jl")
include("equations_gravitational_potential.jl")

end # module
