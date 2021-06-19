# #!/usr/bin/env julia --project
# include("../interface/utilities/boilerplate.jl")
# set up parameters
# backend = DiscontinuousGalerkinBackend
# boilerplate(backend)

using StaticArrays
include("../src/interface/domains.jl")
include("../src/interface/grids.jl")
include("../src/interface/models.jl")
include("../src/interface/physics.jl")
include("../src/interface/boundary_conditions.jl")
include("../src/interface/callbacks.jl")
include("../src/backends/backends.jl")
include("../src/interface/simulations.jl")
include("../src/interface/callbacks.jl")
include("../src/backends/dg_model_backends/boilerplate.jl")
include("../src/interface/timestepping.jl")

# to be removed
using CLIMAParameters#: AbstractEarthParameterSet
struct PlanetParameterSet <: AbstractEarthParameterSet end
get_planet_parameter(p::Symbol) = getproperty(CLIMAParameters.Planet, p)(PlanetParameterSet())

parameters = (
    R_d  = get_planet_parameter(:R_d),
    pₒ   = get_planet_parameter(:MSLP),
    κ    = get_planet_parameter(:kappa_d),
    g    = get_planet_parameter(:grav),
    cp_d = get_planet_parameter(:cp_d),
    cv_d = get_planet_parameter(:cv_d),
    γ    = get_planet_parameter(:cp_d)/get_planet_parameter(:cv_d),
    T_0  = get_planet_parameter(:T_0),
    xc   = 5000,
    yc   = 1000,
    zc   = 2000,
    rc   = 2000,
    xmax = 10000,
    ymax = 500,
    zmax = 10000,
    θₐ   = 2.0,
    cₛ   = 340,
)

# set up grid
x_domain = IntervalDomain(min = 0, max = parameters.xmax, periodic = true)
y_domain = IntervalDomain(min = 0, max = parameters.ymax, periodic = true)
z_domain = IntervalDomain(min = 0, max = parameters.zmax, periodic = false)
grid = DiscretizedDomain(
    domain = x_domain × y_domain × z_domain,
    discretization = SpectralElementGrid(elements = (10,1,10), polynomial_order = (4,4,4),),
)

# set up inital condition
r(p,x,z)       = sqrt((x - p.xc)^2 + (z - p.zc)^2)
Δθ(p,x,y,z)    = (r(p,x,z) < p.rc) ? ( p.θₐ * (1.0 - r(p,x,z) / p.rc) ) : 0
θ₀(p,x,y,z)    = 300.0 + Δθ(p,x,y,z)
π_exn(p,x,y,z) = 1.0 - p.g / (p.cp_d * θ₀(p,x,y,z) ) * z  

e_pot(p,x,y,z) = p.g * z
e_int(p,x,y,z) = p.cv_d * (θ₀(p,x,y,z) * π_exn(p,x,y,z) - p.T_0 )
e_kin(p,x,y,z) = 0.0

ρ₀(p,x,y,z)    = p.pₒ / (p.R_d * θ₀(p,x,y,z)) * (π_exn(p,x,y,z))^(p.cv_d / p.R_d)
ρu₀(p,x,y,z)   = @SVector [0.0, 0.0, 0.0]
ρe₀(p,x,y,z)   = ρ₀(p,x,y,z) * (e_kin(p,x,y,z) + e_int(p,x,y,z) + e_pot(p,x,y,z))

# set up model
model = ModelSetup(
    equations = ThreeDimensionalEuler(
        thermodynamic_variable = TotalEnergy(),
        equation_of_state = DryIdealGas(),
        pressure_convention = Compressible(),
        physics = (
            orientation = FlatOrientation(),
            gravity = ShallowGravity(),
            ref_state = NoReferenceState(),
            parameters = parameters,
        )
    ),
    boundary_conditions = (
        ρ  = (top = NoFlux(), bottom = NoFlux(),),
        ρu = (top = FreeSlip(), bottom = FreeSlip(),),
        ρe = (top = NoFlux(), bottom = NoFlux(),),
    ),
    initial_conditions = (
        ρ = ρ₀, ρu = ρu₀, ρe = ρe₀,
    ),
    parameters = parameters,
)

# set up simulation
simulation = Simulation(
    backend = DiscontinuousGalerkinBackend(
        grid = grid,
        numerics = (
            flux = :lmars,
        ),
    ),
    model = model,
    timestepper = (
        method = SSPRK22Heuns, 
        start = 0.0, 
        finish = 4000.0,
        timestep = 10.0,
    ),
    callbacks = (
        Info(),
    ),
)

# run the simulation
initialize!(simulation)
evolve!(simulation)