push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))

using ClimaCore.Geometry, LinearAlgebra, UnPack
import ClimaCore: Fields, Domains, Topologies, Meshes, Spaces
import ClimaCore.Operators
import ClimaCore.Geometry
using LinearAlgebra, IntervalSets
using OrdinaryDiffEq: ODEProblem, solve, SSPRK33

using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

# set up boilerplate
include("../src/interface/WIP_domains.jl")
include("../src/interface/WIP_models.jl")
include("../src/interface/WIP_timesteppers.jl")
include("../src/backends/backends.jl")
include("../src/backends/climacore/function_spaces.jl")
include("../src/backends/climacore/ode_problems.jl")
include("../src/backends/climacore/tendencies.jl")
include("../src/interface/WIP_boundary_conditions.jl")
include("../src/interface/WIP_simulations.jl")

# set up parameters
const parameters = (
    ϵ  = 0.1,   # perturbation size for initial condition
    l  = 0.5,   # Gaussian width
    k  = 0.5,   # Sinusoidal wavenumber
    ρ₀ = 1.0,  # reference density
    c  = 2,
    g  = 10,
    D₄ = 1e-4, # hyperdiffusion coefficient
)

# set up domain
domain = Rectangle(
    xlim = -2π..2π, 
    ylim = -2π..2π, 
    nelements = (16, 16), 
    npolynomial = 4, 
    periodic = (true, true)
)

# set up initial condition
function init_state(x, p)
    @unpack x1, x2 = x

    # set initial density field
    ρ = p.ρ₀

    # set initial velocity field
    U₁ = cosh(x2)^(-2)
    gaussian = exp(-(x2 + p.l / 10)^2 / 2p.l^2)
    u₁′ = gaussian * (x2 + p.l / 10) / p.l^2 * cos(p.k * x1) * cos(p.k * x2)
    u₁′ += p.k * gaussian * cos(p.k * x1) * sin(p.k * x2)
    u₂′ = -p.k * gaussian * sin(p.k * x1) * cos(p.k * x2)
    u = Cartesian12Vector(U₁ + p.ϵ * u₁′, p.ϵ * u₂′)
    
    # set initial tracer field
    θ = sin(p.k * x2)

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
    callbacks = nothing,
)

# run simulation
evolution = evolve(simulation)

# post-processing
ENV["GKSwstype"] = "nul"
import Plots
Plots.GRBackend()

dirname = "cg_invariant_hypervisc"
path = joinpath(@__DIR__, "output", dirname)
mkpath(path)

anim = Plots.@animate for u in evolution.u
    Plots.plot(u.ρθ, clim = (-1, 1))
end
Plots.mp4(anim, joinpath(path, "tracer.mp4"), fps = 10)