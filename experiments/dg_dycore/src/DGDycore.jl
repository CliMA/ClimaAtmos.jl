#=
DGDycore — sandboxed DG-FD sphere dynamical cores (flux-form FDDG and
vector-invariant) in the ClimaAtmos repository.

Reuses ClimaAtmos as a library (parameters, ICs, Held–Suarez forcing,
diagnostics conventions) while owning the prognostic state, tendencies,
Jacobians, and integrator assembly. NOTHING under ClimaAtmos/src is touched. All
configuration lives in a concretely-typed `DGModel` built once per problem and
passed to the tendency functions as the integrator parameter `p`.

DG discretization invariant: `Spaces.weighted_dss!` is NEVER called — the DG face
operators (`add_numerical_flux_internal!`, lifting corrections) replace DSS
entirely (enforced by a grep test in test/).
=#
module DGDycore

using LinearAlgebra: ×, norm, norm_sqr, dot
import LinearAlgebra: ldiv!

import ClimaComms
ClimaComms.@import_required_backends

import ClimaCore
import ClimaCore:
    Domains,
    Fields,
    Geometry,
    Grids,
    Hypsography,
    Meshes,
    Operators,
    Quadratures,
    Spaces,
    Topologies
using ClimaUtilities: SpaceVaryingInputs.SpaceVaryingInput
import ClimaCore.Geometry: ⊗
using ClimaCore.MatrixFields
using ClimaCore.MatrixFields: @name

import SciMLBase
import ClimaTimeSteppers as CTS
import Printf

import ClimaParams as CP
import ClimaAtmos as CA
import ClimaAtmos.Parameters as CAP
import Thermodynamics as TD

const C3 = Geometry.Covariant3Vector
const C12 = Geometry.Covariant12Vector
const C123 = Geometry.Covariant123Vector
const CT3 = Geometry.Contravariant3Vector
const CT12 = Geometry.Contravariant12Vector
const CT123 = Geometry.Contravariant123Vector

export BaroclinicWaveFDDG, BaroclinicWaveDG, DGSimulation, run!

include("parameters.jl")
include("problems.jl")
include("model.jl")
include("initial_conditions.jl")
include("held_suarez.jl")
include("entropy_correction.jl")
include("microphysics.jl")
include("flux_form.jl")
include("vector_invariant.jl")
include("diagnostics.jl")
include("jacobians.jl")
include("simulation.jl")

end # module
