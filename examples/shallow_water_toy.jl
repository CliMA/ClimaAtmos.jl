# Julia ecosystem
using UnPack: @unpack
using OrdinaryDiffEq: SSPRK33

# Clima ecosystem
using ClimaAtmos.Domains: PeriodicPlane
using ClimaAtmos.Models: ShallowWaterModel
using ClimaAtmos.Timesteppers: Timestepper
using ClimaAtmos.Simulations: Simulation, run
using ClimaCore.Geometry: Cartesian12Vector

parameters = (
    ϵ  = 0.1,  # perturbation size for initial condition
    l  = 0.5,  # Gaussian width
    k  = 0.5,  # sinusoidal wavenumber
    h₀ = 1.0,  # reference height
    c  = 2.0,  
    g  = 9.8,  # gravitational constant
    D₄ = 1e-4, # hyperdiffusion coefficient
)

# this function initializes the prognostic state
function initialize_state(x, y, parameters)
    @unpack h₀, l, k, ϵ = parameters
    
    # density
    h = h₀

    # velocity
    U₁ = cosh(y)^(-2)
    gaussian = exp(-(y + l / 10)^2 / 2l^2)
    u₁′ = gaussian * (y + l / 10) / l^2 * cos(k * x) * cos(k * y)
    u₁′ += k * gaussian * cos(k * x) * sin(k * y)
    u₂′ = -k * gaussian * sin(k * x) * cos(k * y)
    u = Cartesian12Vector(U₁ + ϵ * u₁′, ϵ * u₂′)
    
    # passive tracer
    c = sin(k * y)

    return (h = h, u = u, c = c,)
end

# set up domain
domain = PeriodicPlane(xlim = (-2π, 2π), ylim = (-2π, 2π), nelements = (16,16), npolynomial = 4)

# set up model
model = ShallowWaterModel(
    domain = domain, 
    boundary_conditions = nothing, 
    initial_conditions = initialize_state, 
    parameters = parameters
)

# set up time stepper
#stepper = Timestepper(method = SSPRK33(), dt = 0.04, tspan = (0.0,10.0), saveat = 1.0)
stepper = Timestepper(method = SSPRK33(), dt = 0.04, tspan = (0.0, 0.04), saveat = 1.0)

# set up & run simulation
simulation = Simulation(model = model, stepper = stepper)
solution = run(simulation)

# post-processing
ENV["GKSwstype"] = "nul"
import Plots
Plots.GRBackend()

# make output directory
dirname = "shallow_water_toy"
path = joinpath(@__DIR__, "output", dirname)
mkpath(path)

# make a video
anim = Plots.@animate for u in solution.u
    Plots.plot(u.c, clim = (-1, 1))
end
Plots.mp4(anim, joinpath(path, "tracer.mp4"), fps = 10)