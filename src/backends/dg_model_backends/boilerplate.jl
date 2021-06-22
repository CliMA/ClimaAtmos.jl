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

# for grids
import ClimateMachine.Mesh.Grids: DiscontinuousSpectralElementGrid

# for balance laws and bcs
import ClimateMachine.BalanceLaws:
    # declaration
    vars_state,
    # initialization
    nodal_init_state_auxiliary!,
    init_state_prognostic!,
    init_state_auxiliary!,
    # rhs computation
    compute_gradient_argument!,
    compute_gradient_flux!,
    flux_first_order!,
    flux_second_order!,
    source!,
    # boundary conditions
    boundary_conditions,
    boundary_state!
import ClimateMachine.NumericalFluxes:
    numerical_boundary_flux_first_order!,
    numerical_boundary_flux_second_order!
# to be removed: needed for updating ref state
import ClimateMachine.ODESolvers: update_backward_Euler_solver!
import ClimateMachine.DGMethods: update_auxiliary_state!

# for numerical fluxes
using ClimateMachine.NumericalFluxes
using ClimateMachine.DGMethods.NumericalFluxes: NumericalFluxFirstOrder
import ClimateMachine.DGMethods.NumericalFluxes:
    numerical_volume_conservative_flux_first_order!,
    numerical_volume_fluctuation_flux_first_order!,
    ave,
    numerical_flux_first_order!,
    numerical_flux_second_order!,
    numerical_boundary_flux_second_order!
import ClimateMachine.BalanceLaws:
    wavespeed

# for callbacks
using ClimateMachine.GenericCallbacks:
    EveryXWallTimeSeconds, EveryXSimulationSteps
using ClimateMachine.Thermodynamics: soundspeed_air
using ClimateMachine.VariableTemplates: flattenednames

# interface includes
include("grids.jl")
include("balance_laws.jl")
include("callbacks.jl")

# TODO! Remove this.
dirs = ["three_dimensional_dry_compressible_euler_with_total_energy/",
    "three_dimensional_moist_compressible_euler_with_total_energy/",
    "barotropic_fluid/"]
for dir in dirs
    include(dir * "balance_law_interface.jl")
    include(dir * "boundary_conditions_interface.jl")
    include(dir * "numerical_fluxes_interface.jl")
end
include("./three_dimensional_dry_compressible_euler_with_total_energy/physics/physics.jl")
include("./three_dimensional_dry_compressible_euler_with_total_energy/physics/advection.jl")
include("./three_dimensional_dry_compressible_euler_with_total_energy/physics/coriolis.jl")
include("./three_dimensional_dry_compressible_euler_with_total_energy/physics/pressure_force.jl")
include("./three_dimensional_dry_compressible_euler_with_total_energy/physics/gravity.jl")
include("./three_dimensional_dry_compressible_euler_with_total_energy/physics/thermodynamics.jl")
include("./three_dimensional_dry_compressible_euler_with_total_energy/physics/temperature_profiles.jl")

include("./three_dimensional_moist_compressible_euler_with_total_energy/physics/physics.jl")
include("./three_dimensional_moist_compressible_euler_with_total_energy/physics/advection.jl")
include("./three_dimensional_moist_compressible_euler_with_total_energy/physics/coriolis.jl")
include("./three_dimensional_moist_compressible_euler_with_total_energy/physics/pressure_force.jl")
include("./three_dimensional_moist_compressible_euler_with_total_energy/physics/gravity.jl")
include("./three_dimensional_moist_compressible_euler_with_total_energy/physics/thermodynamics.jl")
include("./three_dimensional_moist_compressible_euler_with_total_energy/physics/temperature_profiles.jl")
include("./three_dimensional_moist_compressible_euler_with_total_energy/physics/microphysics.jl")

# TODO! Remove this.
include("./barotropic_fluid/physics/physics.jl")
include("./barotropic_fluid/physics/advection.jl")
include("./barotropic_fluid/physics/coriolis.jl")
include("./barotropic_fluid/physics/thermodynamics.jl")

ClimateMachine.init()