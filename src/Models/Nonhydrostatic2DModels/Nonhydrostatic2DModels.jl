module Nonhydrostatic2DModels

using StaticArrays
using UnPack
using CLIMAParameters
using Thermodynamics
using ClimaCore: Geometry, Spaces, Fields, Operators
using ClimaCore.Geometry: ⊗
using ...Domains, ...Models, ...BoundaryConditions

export Nonhydrostatic2DModel

include("nonhydrostatic_2d_model.jl")
include("equations_pressure.jl")
include("equations_base_model.jl")
include("equations_thermodynamics.jl")
include("equations_moisture.jl")

end # module
