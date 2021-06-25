# #!/usr/bin/env julia --project
# include("../interface/utilities/boilerplate.jl")
# set up parameters
# backend = DiscontinuousGalerkinBackend
# boilerplate(backend)

using StaticArrays
include("../src/interface/domains.jl")
include("../src/interface/models.jl")
include("../src/interface/physics.jl")
include("../src/interface/boundary_conditions.jl")
include("../src/interface/grids.jl")
include("../src/backends/backends.jl")
include("../src/backends/dg_model_backends/backend_hook.jl")
include("../src/interface/simulations.jl")
include("../src/interface/callbacks.jl")
include("../src/backends/dg_model_backends/boilerplate.jl")

# to be removed
using CLIMAParameters#: AbstractEarthParameterSet
struct PlanetParameterSet <: AbstractEarthParameterSet end
get_planet_parameter(p::Symbol) = getproperty(CLIMAParameters.Planet, p)(PlanetParameterSet())

# set up backend
backend = DiscontinuousGalerkinBackend(numerics = (flux = :lmars,),)

parameters = (
    R_d  = get_planet_parameter(:R_d),
    R_v  = get_planet_parameter(:R_v),
    pₒ   = get_planet_parameter(:MSLP),
    κ    = get_planet_parameter(:kappa_d),
    g    = get_planet_parameter(:grav),
    cp_d = get_planet_parameter(:cp_d),
    cp_v = get_planet_parameter(:cp_v),
    cp_l = get_planet_parameter(:cp_l),
    cp_i = get_planet_parameter(:cp_i),
    cv_d = get_planet_parameter(:cv_d),
    cv_v = get_planet_parameter(:cv_v),
    cv_l = get_planet_parameter(:cv_l),
    cv_i = get_planet_parameter(:cv_i),
    LH_v0    = get_planet_parameter(:LH_v0),
    e_int_v0 = get_planet_parameter(:e_int_v0),
    e_int_i0 = get_planet_parameter(:e_int_i0),
    molmass_ratio = get_planet_parameter(:molmass_dryair)/get_planet_parameter(:molmass_water),
    pₜᵣ  = get_planet_parameter(:press_triple),
    Tₜᵣ  = get_planet_parameter(:T_triple),
    T_0  = get_planet_parameter(:T_0),
    xc   = 5000,
    yc   = 1000,
    zc   = 2000,
    rc   = 2000,
    xmax = 10000,
    ymax = 5000,
    zmax = 10000,
    θₐ   = 2.0,
    cₛ   = 340,
    q₀   = 1e-3,
    τ_precip = 0.1, # might need longer timescale
)

# set up grid
x_domain = IntervalDomain(min = 0, max = parameters.xmax, periodic = true)
y_domain = IntervalDomain(min = 0, max = parameters.ymax, periodic = true)
z_domain = IntervalDomain(min = 0, max = parameters.zmax, periodic = false)
discretized_domain = DiscretizedDomain(
    domain = x_domain × y_domain × z_domain,
    discretization = SpectralElementGrid(elements = (10,1,10), polynomial_order = (4,4,4),),
)

# set up inital condition
r(p, x, z)          = sqrt((x - p.xc)^2 + (z - p.zc)^2)
Δθ(p, x, y, z)      = (r(p, x, z) < p.rc) ? ( p.θₐ * (1.0 - r(p, x, z) / p.rc) ) : 0
θ₀(p, x, y, z)      = 300.0 + Δθ(p, x, y, z)
Δq(p, x, y, z)      = (r(p, x, z) < p.rc) ? ( p.q₀ * (1.0 - r(p, x, z) / p.rc) ) : 0
q(p, x, y, z)       = 0.0 + Δq(p, x, y, z)
π_exner(p, x, y, z) = 1.0 - p.g / (p.cp_d * θ₀(p, x, y, z) ) * z  

ρ₀(p, x, y, z)  = p.pₒ / (p.R_d * θ₀(p, x, y, z)) * (π_exner(p, x, y, z))^(p.cv_d / p.R_d)
ρu⃗₀(p, x, y, z) = @SVector [0, 0, 0]

cv_m(p, x, y, z)  = p.cv_d + (p.cv_v - p.cv_d) * q(p, x, y, z)

e_pot(p, x, y, z) = p.g * z
e_int(p, x, y, z) = cv_m(p, x, y, z) * (θ₀(p, x, y, z) * π_exner(p, x, y, z) - p.T_0) + q(p, x, y, z) * p.e_int_v0
e_kin(p, x, y, z) = 0.0

ρe(p, x, y, z) = ρ₀(p, x, y, z) * (e_kin(p, x, y, z) + e_int(p, x, y, z) + e_pot(p, x, y, z))
ρq(p, x, y, z) = ρ₀(p, x, y, z) * q(p, x, y, z)

# set up model
model = ModelSetup(
    equations = ThreeDimensionalEuler(
        thermodynamic_variable = TotalEnergy(),
        equation_of_state = MoistIdealGas(),
        pressure_convention = Compressible(),
        sources = (
            gravity = Gravity(),
            # microphysics = ZeroMomentMicrophysics(),
        ),
        ref_state = NoReferenceState(),
    ),
    boundary_conditions = (0,0,1,1,DefaultBC(),DefaultBC()),
    # boundary_conditions = (
    #     ρ  = (top = NoFlux(), bottom = NoFlux(),),
    #     ρu = (top = FreeSlip(), bottom = FreeSlip(),),
    #     ρe = (top = NoFlux(), bottom = NoFlux(),),
    # ),
    initial_conditions = (
        ρ = ρ₀, ρu = ρu⃗₀, ρe = ρe, ρq = ρq,
    ),
    parameters = parameters,
)

# set up simulation
simulation = Simulation(
    backend = backend,
    discretized_domain = discretized_domain,
    model = model,
    timestepper = (
        method = SSPRK22Heuns, 
        start = 0.0, 
        finish = 4000.0,
        timestep = 0.063,
    ),
    callbacks = (
        Info(),
        VTKState(iteration = Int(floor(100.0/0.063)), filepath = "./out/"),
        CFL(), 
    ),
)


# run the simulation
initialize!(simulation)
evolve!(simulation)

nothing