module Backends

export rad, lon, lat, r̂ⁿᵒʳᵐ, ϕ̂ⁿᵒʳᵐ, λ̂ⁿᵒʳᵐ, r̂, ϕ̂, λ̂, rfunc, ϕfunc, λfunc, ρu⃗

import ..Interface: AbstractModel, AbstractDomain, Rectangle, PeriodicRectangle, SphericalShell, ClimaCoreBackend, DiscontinuousGalerkinBackend, SingleColumn
import ClimaAtmos.Interface: TimeStepper, AbstractTimestepper, NoSplitting, DefaultBC, Simulation

import ClimaAtmos.Interface: BarotropicFluidModel, HydrostaticModel, AbstractBackend
import ClimaAtmos.Interface: BarotropicFluid, DeepShellCoriolis

@info "error / warning comes from ClimaCore"
using UnPack
using ClimaCore
using RecursiveArrayTools

using OrdinaryDiffEq: ODEProblem, solve, SSPRK33

using ClimaCore.Spaces
using ClimaCore.Geometry
import ClimaCore: Fields, Domains, Topologies, Meshes, Spaces
import ClimaCore.Operators
import ClimaCore.Geometry
import ClimaCore:
    Fields,
    Domains,
    Topologies,
    Meshes,
    DataLayouts,
    Operators,
    Geometry,
    Spaces

using ClimaCore.Spaces: SpectralElementSpace2D, CenterFiniteDifferenceSpace, FaceFiniteDifferenceSpace
import LinearAlgebra: norm, ×

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

# include
include("climacore/function_spaces.jl")
include("climacore/initial_conditions.jl")
include("climacore/ode_problems.jl")
include("climacore/tendencies.jl")

include("discontinuous_galerkin/WIP_ode_problem.jl")
include("discontinuous_galerkin/WIP_grids.jl")
include("discontinuous_galerkin/WIP_balance_laws.jl")
include("discontinuous_galerkin/WIP_backend_hook.jl")
include("discontinuous_galerkin/sphere_utils.jl")

include("discontinuous_galerkin/barotropic_fluid/balance_law_interface.jl")
include("discontinuous_galerkin/barotropic_fluid/boundary_conditions_interface.jl")
include("discontinuous_galerkin/barotropic_fluid/numerical_fluxes_interface.jl")
include("discontinuous_galerkin/barotropic_fluid/sources_interface.jl")
include("discontinuous_galerkin/barotropic_fluid/thermodynamics.jl")

# Simulation 
function Simulation(
    backend::AbstractBackend;
    model,
    timestepper::AbstractTimestepper,
    callbacks,
)
    ode_problem = create_ode_problem(
        backend, 
        model,
        timestepper, 
    )
  
    return Simulation(
        backend,
        model,
        timestepper, 
        callbacks, 
        ode_problem,
    )
end

# time integration for ClimaCore
function evolve(simulation::Simulation{<:ClimaCoreBackend})
    return solve(
        simulation.ode_problem,
        simulation.timestepper.method,
        dt = simulation.timestepper.dt,
        saveat = simulation.timestepper.saveat,
        progress = simulation.timestepper.progress, 
        progress_message = simulation.timestepper.progress_message,
    )
end

# time integration for DG end in ClimateMachine
function evolve(simulation::Simulation{<:DiscontinuousGalerkinBackend})
    method        = simulation.timestepper.method
    start         = simulation.timestepper.tspan[1]
    finish        = simulation.timestepper.tspan[2]
    timestep      = simulation.timestepper.dt
    splitting     = simulation.timestepper.splitting
    rhs           = simulation.ode_problem.rhs
    state         = simulation.ode_problem.state

    ode_solver = construct_odesolver(splitting, simulation)

    # TODO: add back in callbacks
    cb_vector = () #create_callbacks(simulation, ode_solver)

    # Perform evolution of simulations
    if isempty(cb_vector)
        solve!(
            state, 
            ode_solver; 
            timeend = finish, 
            adjustfinalstep = false,
        )
    else
        solve!(
            state,
            ode_solver;
            timeend = finish,
            callbacks = cb_vector,
            adjustfinalstep = false,
        )
    end
end

end # end of module