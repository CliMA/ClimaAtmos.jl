module Diagnostics

import LinearAlgebra: dot

import ClimaCore: Fields, Geometry, InputOutput, Meshes, Spaces, Operators
import ClimaCore.Utilities: half
import Thermodynamics as TD

import ..AtmosModel
import ..call_every_n_steps
import ..Parameters as CAP

import ..unit_basis_vector_data

# moisture_model
import ..DryModel
import ..EquilMoistModel
import ..NonEquilMoistModel

# energy_form
import ..TotalEnergy

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
