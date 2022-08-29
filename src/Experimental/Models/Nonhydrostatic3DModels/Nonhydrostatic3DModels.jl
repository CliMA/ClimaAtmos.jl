module Nonhydrostatic3DModels

import ....Parameters as CAP
using LinearAlgebra
using StaticArrays
using UnPack
import Thermodynamics as TD
using ClimaCore: Geometry, Spaces, Fields, Operators
using ClimaCore.Geometry: ⊗
using ...Domains, ...Models
using ClimaAtmos.Experimental.BoundaryConditions

export Nonhydrostatic3DModel

include("nonhydrostatic_3d_model.jl")
include("equations_gravitational_potential.jl")
include("equations_pressure.jl")
include("equations_base_model.jl")
include("equations_thermodynamics.jl")
include("equations_moisture.jl")
include("equations_vertical_diffusion.jl")

end # module
