using StaticArrays
include("../src/interface/domains.jl")
include("../src/interface/models.jl")
include("../src/interface/physics.jl")
include("../src/interface/boundary_conditions.jl")
include("../src/interface/grids.jl")
include("../src/backends/backends.jl")
include("../src/interface/timestepper_abstractions.jl")
include("../src/backends/dg_model_backends/backend_hook.jl")
include("../src/interface/simulations.jl")
include("../src/interface/callbacks.jl")
include("../src/backends/dg_model_backends/boilerplate.jl")
include("../src/utils/sphere_utils.jl")

# set up backend
backend = DiscontinuousGalerkinBackend(numerics = (flux = :lmars,),)

# set up parameters
const parameters = (
    g  = 0.0,
    ρₒ = 1.0,  # reference density
    cₛ = 1e-2, # sound speed
    ℓᵐ = 10,   # jet thickness, (larger is thinner)
    ℓ  = 20,   # perturbation thickness, (larger is thinner)
    m  = 2,    # number of sign changes on equator for perturbation
    ϕᵖ = π/2 * 0.05, # of centerdness of perturbation
    ϵ  = 0.3,  # perturbation amplitude
    vˢ = 5e-4, # velocity scale
    α  = 2e-4,
    Ω  = 1e-3,
)

# set up grid
discretized_domain = DiscretizedDomain(
    domain = SphericalShell(
        radius = 1.0,
        height = 0.2,
    ),
    discretization = (
	    horizontal = SpectralElementGrid(elements = 8, polynomial_order = 3), 
	    vertical = SpectralElementGrid(elements = 1, polynomial_order = 1)
	),
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
model = ModelSetup(
    equations = ThreeDimensionalEuler(
        thermodynamic_variable = Density(),
        equation_of_state = BarotropicFluid(),
        pressure_convention = Compressible(),
        sources = (
           DeepShellCoriolis(),
        ),
        ref_state = NoReferenceState(),
    ),
    domain = discretized_domain,
    boundary_conditions = (DefaultBC(), DefaultBC()),
    initial_conditions = (
        ρ = ρ₀ᶜᵃʳᵗ, ρu = ρu⃗₀ᶜᵃʳᵗ, ρθ = ρθ₀ᶜᵃʳᵗ,
    ),
    parameters = parameters,
)

# # set up simulation
# simulation = Simulation(
#     backend = backend,
#     discretized_domain = discretized_domain,
#     model = model,
#     timestepper = (
#         method = SSPRK22Heuns, 
#         start = 0.0, 
#         finish = 8*1600.0,
#         timestep = 1.0,
#     ),
#     callbacks = (
#         Info(),
#         VTKState(iteration = Int(floor(100.0/1.0)), filepath = "./out/"),
#         CFL(), 
#     ),
# )

# # run the simulation
# initialize!(simulation)
# evolve!(simulation)

nothing