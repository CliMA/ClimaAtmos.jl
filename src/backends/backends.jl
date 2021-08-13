module Backends

import ..Interface: AbstractModel, AbstractDomain, Rectangle, PeriodicRectangle, ClimaCoreBackend, SingleColumn
import ClimaAtmos.Interface: TimeStepper, AbstractTimestepper, Simulation

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
# include
include("climacore/function_spaces.jl")
include("climacore/initial_conditions.jl")
include("climacore/ode_problems.jl")
include("climacore/tendencies.jl")

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