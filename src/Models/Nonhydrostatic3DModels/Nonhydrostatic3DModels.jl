module Nonhydrostatic3DModels

using LinearAlgebra
using LinearAlgebra: norm_sqr
using StaticArrays
using UnPack
using OrdinaryDiffEq: ODEFunction, SplitFunction
using CLIMAParameters
using Thermodynamics
using ClimaCore: Geometry, Spaces, Fields, Operators
using ClimaCore.Geometry: ⊗
using ...Domains, ...Models

export Nonhydrostatic3DModel

include("nonhydrostatic_3d_model.jl")
include("jacobian.jl")
include("equations_gravitational_potential.jl")
include("equations_pressure.jl")
include("equations_base_model.jl")
include("equations_thermodynamics.jl")
include("equations_moisture.jl")
include("equations_vertical_diffusion.jl")

end # module
