# @boilerplate

# Parameters
parameters = (
    ϵ = 0.1,  # perturbation size for initial condition
    l = 0.5, # Gaussian width
    k = 0.5, # Sinusoidal wavenumber
    ρ₀ = 1.0, # reference density
    c = 2,
    g = 10,
)

# Set up grid
Ωˣ = IntervalDomain(min = -2π, max = 2π, periodic = true)
Ωʸ = IntervalDomain(min = -2π, max = 2π, periodic = true)
discretized_domain = DiscretizedDomain(
    domain = Ωˣ × Ωʸ,
    discretization = (
	    horizontal = SpectralElementGrid(elements = 8, polynomial_order = 3), 
	),
)

########
# Set up inital condition
########
gaussian(p,x,y) = exp(-(y + p.l / 10)^2 / 2p.l^2)
u(p,x,y) =  ( cosh(x2)^(-2) 
+ p.ϵ * gaussian(p,x,y) * (x2 + p.l / 10) / p.l^2 * cos(p.k * x) * cos(p.k * y)
+ p.ϵ * p.k * gaussian(p,x,y) * cos(p.k * x) * sin(p.k * y)
)
v(p,x,y) = -p.k * gaussian(p,x,y) * sin(p.k * x) * cos(p.k * xy)
ρ₀(p,x,y)  = 1.0
ρu⃗₀(p,x,y) =  @SVector[u(p,x,y),v(p,x,y)]
ρθ₀(p,x,y) = sin(p.k * y)

# set up model
model = ModelSetup(
    equations = TwoDimensionalEuler(
        thermodynamic_variable = Density(),
        equation_of_state = BarotropicFluid(),
        pressure_convention = Compressible(),
    ),
    initial_conditions = (
        ρ = ρ₀, ρu = ρu⃗₀, ρθ = ρθ₀,
    ),
    parameters = parameters,
)

# set up simulation
simulation = Simulation(
    discretized_domain = discretized_domain,
    backend = CoreBackend(
        numerics = (
            flux = :roe,
        ),
    ),
    model = model,
    timestepper = (
        method = SSPRK33(), 
        start = 0.0, 
        finish = 200,
        timestep = 0.02,
    ),
    callbacks = (
        Info(),
    ),
)

# run the simulation
initialize!(simulation)
evolve!(simulation)

nothing