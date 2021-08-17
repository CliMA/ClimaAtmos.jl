using ClimaAtmos
using ClimateMachine
ClimateMachine.init()

@boilerplate

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

# set up grid
domain = SphericalShell(
    radius = 1.0,
    height = 0.2,
    nelements = (8, 1),
    npolynomial = 3,
    vertical_discretization = VerticalDiscontinousGalerkin(vpolynomial=1),
)

# set up inital condition
uᵐ(p, λ, ϕ, r) =  p.ℓᵐ * sech(p.ℓᵐ * ϕ)^2 
vᵐ(p, λ, ϕ, r) =  0.0
hᵐ(p, λ, ϕ, r) =  0.0

u1(p, λ, ϕ, r) =  p.ℓ * 2 * (ϕ - p.ϕᵖ)* exp(-p.ℓ * (ϕ - p.ϕᵖ)^2) * cos(ϕ) * cos(2 * (ϕ - p.ϕᵖ)) * sin(p.m * λ)
u2(p, λ, ϕ, r) =  exp(-p.ℓ * (ϕ - p.ϕᵖ)^2) * sin(ϕ) * cos(2 * (ϕ - p.ϕᵖ)) * sin(p.m * λ)
u3(p, λ, ϕ, r) =  2*exp(-p.ℓ * (ϕ - p.ϕᵖ)^2) * cos(ϕ) * sin(2 * (ϕ - p.ϕᵖ)) * sin(p.m * λ)
uᵖ(p, λ, ϕ, r) =  u1(p, λ, ϕ, r) + u2(p, λ, ϕ, r) + u3(p, λ, ϕ, r)
vᵖ(p, λ, ϕ, r) =  p.m * exp(-p.ℓ * (ϕ - p.ϕᵖ)^2) * cos(2 * (ϕ - p.ϕᵖ)) * cos(p.m * λ)
hᵖ(p, λ, ϕ, r) =  0.0

ρ₀(p, λ, ϕ, r)    = p.ρₒ 
ρuʳᵃᵈ(p, λ, ϕ, r) = 0
ρuˡᵃᵗ(p, λ, ϕ, r) = p.vˢ * ρ₀(p, λ, ϕ, r) * (p.ϵ * vᵖ(p, λ, ϕ, r))
ρuˡᵒⁿ(p, λ, ϕ, r) = p.vˢ * ρ₀(p, λ, ϕ, r) * (uᵐ(p, λ, ϕ, r) + p.ϵ * uᵖ(p, λ, ϕ, r))
ρθ₀(p, λ, ϕ, r) = ρ₀(p, λ, ϕ, r) * tanh(p.ℓᵐ * ϕ)

# Cartesian Representation for initialization (boiler plate really)
ρ₀ᶜᵃʳᵗ(p, x...)  = ρ₀(p, lon(x...), lat(x...), rad(x...))
ρu⃗₀ᶜᵃʳᵗ(p, x...) = (   ρuʳᵃᵈ(p, lon(x...), lat(x...), rad(x...)) * r̂(x...) 
                     + ρuˡᵃᵗ(p, lon(x...), lat(x...), rad(x...)) * ϕ̂(x...)
                     + ρuˡᵒⁿ(p, lon(x...), lat(x...), rad(x...)) * λ̂(x...) ) 
ρθ₀ᶜᵃʳᵗ(p, x...) = ρθ₀(p, lon(x...), lat(x...), rad(x...))

# set up model
model = BarotropicFluidModel(
    domain = domain,
    physics = ModelPhysics(
        equation_of_state = BarotropicFluid(),
        ref_state = NoReferenceState(),
        sources = ( 
            coriolis = DeepShellCoriolis(), 
        ),
    ),
    boundary_conditions = (DefaultBC(), DefaultBC()),
    initial_conditions = (
        ρ = ρ₀ᶜᵃʳᵗ, ρu = ρu⃗₀ᶜᵃʳᵗ, ρθ = ρθ₀ᶜᵃʳᵗ,
    ),
    parameters = parameters,
)

# set up timestepper
timestepper = TimeStepper(
    method = SSPRK22Heuns,
    dt = 1.0,
    tspan = (0.0, 8*1600.0),
    splitting = NoSplitting(),
    saveat = 1.0,
    progress = true,
    progress_message = (dt, u, p, t) -> t,
)

simulation = Simulation(
    DiscontinuousGalerkinBackend(numerics = (flux = :lmars,),),
    model = model,
    timestepper = timestepper,
    callbacks = (
        # Info(),
        # VTKState(iteration = Int(floor(100.0/1.0)), filepath = "./out/"),
        # CFL(), 
    ), 
)

#=
# set up simulation
# run the simulation
evolve(simulation)

nothing
=#