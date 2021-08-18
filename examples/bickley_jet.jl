using ClimaAtmos

# imports from Backends 
import ClimaAtmos.Backends: evolve, ClimaCoreBackend

# imports from Interface
import ClimaAtmos.Interface: TimeStepper, Simulation
import ClimaAtmos.Interface: PeriodicRectangle, BarotropicFluidModel

# Julia Ecosystem
using IntervalSets, UnPack
using OrdinaryDiffEq: SSPRK33

# import from Clima Core
import ClimaCore.Geometry: Cartesian12Vector

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