using Dates
using GaussQuadrature
using JLD2
using LinearAlgebra
using Logging
using MPI
using Printf
using StaticArrays
using Test

using ClimateMachine
using ClimateMachine.Atmos: NoReferenceState
using ClimateMachine.MPIStateArrays
using ClimateMachine.VariableTemplates
using ClimateMachine.Mesh.Geometry
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.BalanceLaws
using ClimateMachine.ODESolvers
using ClimateMachine.SystemSolvers
using ClimateMachine.Orientations
using ClimateMachine.VTK

# to be removed
using ClimateMachine.GenericCallbacks:
    EveryXWallTimeSeconds, EveryXSimulationSteps
using ClimateMachine.Thermodynamics: soundspeed_air
using ClimateMachine.VariableTemplates: flattenednames

# to be removed: needed for updating ref state
import ClimateMachine.ODESolvers: update_backward_Euler_solver!
import ClimateMachine.DGMethods: update_auxiliary_state!

# interface includes
dirs = ["dg_model_backend/three_dimensional_dry_compressible_euler_with_total_energy/"]
for dir in dirs
    include(dir * "balance_law_interface.jl")
    include(dir * "boundary_conditions_interface.jl")
    include(dir * "numerical_fluxes_interface.jl")
end