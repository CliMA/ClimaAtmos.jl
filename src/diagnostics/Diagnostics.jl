module Diagnostics

import ClimaCore: Fields, Geometry, InputOutput, Meshes, Spaces

import ..AtmosModel
import ..call_every_n_steps

# moisture_model
import ..DryModel
import ..EquilMoistModel
import ..NonEquilMoistModel

# turbconv_model
import ..EDMFX
import ..DiagnosticEDMFX

# Abbreviations (following utils/abbreviations.jl)
const curlₕ = Operators.Curl()
const CT3 = Geometry.Contravariant3Vector
const ᶜinterp = Operators.InterpolateF2C()
# TODO: Implement proper extrapolation instead of simply reusing the first
# interior value at the surface.
const ᶠinterp = Operators.InterpolateC2F(
    bottom = Operators.Extrapolate(),
    top = Operators.Extrapolate(),
)

include("diagnostic.jl")
include("writers.jl")
end
