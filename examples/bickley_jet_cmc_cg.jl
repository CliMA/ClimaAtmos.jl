push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))

using LinearAlgebra, IntervalSets
using OrdinaryDiffEq: ODEProblem, solve, SSPRK33
using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

using ClimaCore.Geometry, LinearAlgebra, UnPack
import ClimaCore: Fields, Domains, Topologies, Meshes, Spaces
import ClimaCore.Operators
import ClimaCore.Geometry

#=
# set up boilerplate
include("../src/interface/WIP_domains.jl")
include("../src/interface/WIP_models.jl")
include("../src/interface/WIP_timesteppers.jl")
#include("../src/interface/timestepper_abstractions.jl")
include("../src/backends/backends.jl")
include("../src/backends/climacore/function_spaces.jl")
include("../src/backends/climacore/initial_conditions.jl")
include("../src/backends/climacore/ode_problems.jl")
include("../src/backends/climacore/tendencies.jl")
include("../src/interface/WIP_boundary_conditions.jl")
include("../src/interface/WIP_simulations.jl")
include("../src/interface/WIP_physics.jl")
=#
using ClimaAtmos
using ClimaAtmos.Utils
using ClimaAtmos.Interface
using ClimaAtmos.Backends

# explicit imports from Interface
import ClimaAtmos.Interface: PeriodicRectangle
import ClimaAtmos.Interface: BarotropicFluidModel
import ClimaAtmos.Interface: TimeStepper
import ClimaAtmos.Interface: DirichletBC, Simulation

# explicit imports from Backends
import ClimaAtmos.Backends: create_ode_problem, evolve

# External Stuff
using IntervalSets
using OrdinaryDiffEq: ODEProblem, solve, SSPRK33


# set up parameters
parameters = (
    ϵ  = 0.1,  # perturbation size for initial condition
    l  = 0.5,  # Gaussian width
    k  = 0.5,  # Sinusoidal wavenumber
    ρ₀ = 1.0,  # reference density
    c  = 2,
    g  = 10,   # gravitation constant
    D₄ = 1e-4, # hyperdiffusion coefficient
)

# set up domain
domain = PeriodicRectangle(
    xlim = -2π..2π, 
    ylim = -2π..2π, 
    nelements = (16, 16), 
    npolynomial = 4, 
)

# set up initial conditions
function init_state(x, y, parameters)
    UnPack.@unpack ρ₀, l, k, ϵ = parameters
    
    # set initial density field
    ρ = ρ₀

    # set initial velocity field
    U₁ = cosh(y)^(-2)
    gaussian = exp(-(y + l / 10)^2 / 2l^2)
    u₁′ = gaussian * (y + l / 10) / l^2 * cos(k * x) * cos(k * y)
    u₁′ += k * gaussian * cos(k * x) * sin(k * y)
    u₂′ = -k * gaussian * sin(k * x) * cos(k * y)
    u = Cartesian12Vector(U₁ + ϵ * u₁′, ϵ * u₂′)
    
    # set initial tracer field
    θ = sin(k * y)

    return (ρ = ρ, u = u, ρθ = ρ * θ)
end

# set up model
model = BarotropicFluidModel( 
    domain = domain, 
    boundary_conditions = nothing, 
    initial_conditions = init_state, 
    parameters = parameters
)

# set up timestepper
timestepper = TimeStepper(
    method = SSPRK33(),
    dt = 0.04,
    tspan = (0.0, 10.0),
    saveat = 1.0,
    progress = true,
    progress_message = (dt, u, p, t) -> t,
)

# set up simulation
simulation = Simulation(
    ClimaCoreBackend(),
    model = model, 
    timestepper = timestepper,
    callbacks = (),
)

# run simulation
sol = evolve(simulation)

# post-processing
ENV["GKSwstype"] = "nul"
import Plots
Plots.GRBackend()

dirname = "cg_invariant_hypervisc"
path = joinpath(@__DIR__, "output", dirname)
mkpath(path)

anim = Plots.@animate for u in sol.u
    Plots.plot(u.ρθ, clim = (-1, 1))
end
Plots.mp4(anim, joinpath(path, "tracer.mp4"), fps = 10)