# Julia ecosystem
using UnPack: @unpack
using OrdinaryDiffEq: SSPRK33, step!

# Clima ecosystem
using ClimaAtmos.Domains: PeriodicPlane
using ClimaAtmos.Models: ShallowWaterModel
using ClimaAtmos.Timesteppers: Timestepper
using ClimaAtmos.Simulations: Simulation, run, init
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

# set up domain
domain = PeriodicPlane(xlim = (-2π, 2π), ylim = (-2π, 2π), nelements = (16,16), npolynomial = 4)

# set up model
model = ShallowWaterModel(domain = domain, parameters = parameters)

# set up time stepper
#stepper = Timestepper(method = SSPRK33(), dt = 0.04, tspan = (0.0,10.0), saveat = 1.0)
stepper = Timestepper(method = SSPRK33(), dt = 0.04, tspan = (0.0, 0.04), saveat = 1.0)

# set up & run simulation
simulation = Simulation(model = model, stepper = stepper)
integrator = init(simulation)
step!(integrator)

# # post-processing
# ENV["GKSwstype"] = "nul"
# import Plots
# Plots.GRBackend()

# # make output directory
# dirname = "shallow_water_toy"
# path = joinpath(@__DIR__, "output", dirname)
# mkpath(path)

# # make a video
# anim = Plots.@animate for u in solution.u
#     Plots.plot(u.c, clim = (-1, 1))
# end
# Plots.mp4(anim, joinpath(path, "tracer.mp4"), fps = 10)