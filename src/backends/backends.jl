module Backends

import ..Interface: AbstractModel, AbstractDomain, Rectangle, PeriodicRectangle, SphericalShell, ClimaCoreBackend, DiscontinuousGalerkinBackend, SingleColumn
import ClimaAtmos.Interface: TimeStepper, AbstractTimestepper, NoSplitting, Simulation

import ClimaAtmos.Interface: BarotropicFluidModel, HydrostaticModel, AbstractBackend

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

# include
include("climacore/function_spaces.jl")
include("climacore/initial_conditions.jl")
include("climacore/ode_problems.jl")
include("climacore/tendencies.jl")

include("discontinuous_galerkin/WIP_ode_problem.jl")
include("discontinuous_galerkin/WIP_grids.jl")
include("discontinuous_galerkin/WIP_balance_laws.jl")
include("discontinuous_galerkin/WIP_backend_hook.jl")

include("discontinuous_galerkin/barotropic_fluid/numerical_fluxes_interface.jl")

# Simulation for ClimaCore
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

end # end of module