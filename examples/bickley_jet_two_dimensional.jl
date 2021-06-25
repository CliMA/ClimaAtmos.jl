@boilerplate

# parameters
parameters = (
    ϵ = 0.1,  # perturbation size for initial condition (dimensionless)
    l = 0.5,  # gaussian width (dimensionless)
    k = 0.5,  # sinusoidal wavenumber (dimensionless)
    ρ₀ = 1.0, # reference density (dimensionless)
    c = 2,    # soundspeed (dimensionless)
)

# domain and resolution
Ωˣ = IntervalDomain(min = -2π, max = 2π, periodic = true)
Ωʸ = IntervalDomain(min = -2π, max = 2π, periodic = true)
space = DiscretizedDomain(
    domain = Ωˣ × Ωʸ,
    resolution = SpectralElementGrid(elements = (8,8), polynomial_order = (3,3)),
)

# initial condition
gaussian(p,x,y) = exp(-(y + p.l / 10)^2 / 2p.l^2)
u(p,x,y) = ( cosh(x2)^(-2) 
+ p.ϵ * gaussian(p,x,y) * (x2 + p.l / 10) / p.l^2 * cos(p.k * x) * cos(p.k * y)
+ p.ϵ * p.k * gaussian(p,x,y) * cos(p.k * x) * sin(p.k * y) )
v(p,x,y) = -p.k * gaussian(p,x,y) * sin(p.k * x) * cos(p.k * xy)
ρ₀(p,x,y)  = 1.0
ρu⃗₀(p,x,y) =  @SVector[u(p,x,y),v(p,x,y)]
ρθ₀(p,x,y) = sin(p.k * y)

# equations, initial conditions, and parameters
model = ModelSetup(
    equations = BarotropicFluid(),
    initial_conditions = (ρ = ρ₀, ρu = ρu⃗₀, ρθ = ρθ₀),
    parameters = parameters,
)

# set up simulation
simulation = Simulation(
    discretized_domain = discretized_domain,
    backend = CoreBackend(flux = RoeFlux()),
    model = model,
    timestepper = (method = SSPRK33(), 
        start = 0.0, 
        finish = 200,
        timestep = 0.02,
    ),
    callbacks = (Info(), JLD2Output(steps = 100)),
)

# initialize to have access to objects in the repl
initialize!(simulation)

# run the simulation
evolve!(simulation)
