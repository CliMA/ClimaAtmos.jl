using ClimaAtmos
using ClimaAtmos.Utils
using ClimaAtmos.Interface
using ClimaAtmos.Backends

# explicit imports from Interface
import ClimaAtmos.Interface: SphericalShell, VerticalDiscontinousGalerkin
import ClimaAtmos.Interface: BarotropicFluidModel
import ClimaAtmos.Interface: ModelPhysics, BarotropicFluid, DeepShellCoriolis
import ClimaAtmos.Interface: TimeStepper, NoSplitting
import ClimaAtmos.Interface: DefaultBC #, Simulation

# from ClimateMachine
using ClimateMachine
using ClimateMachine.Atmos: NoReferenceState
using ClimateMachine.ODESolvers

# # explicit imports from Backends
# import ClimaAtmos.Backends: create_ode_problem

# # External Stuff
# using IntervalSets
# using OrdinaryDiffEq: ODEProblem, solve, SSPRK33

# Domains
domain = SphericalShell(
    radius = 1.0,
    height = 0.2,
    nelements = (8, 1),
    npolynomial = 3,
    vertical_discretization = VerticalDiscontinousGalerkin(vpolynomial=1),
)

# Model
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
    initial_conditions = nothing,
    parameters = nothing,
)

# Timestepper
timestepper = TimeStepper(
    method = SSPRK22Heuns,
    dt = 1.0,
    tspan = (0.0, 8*1600.0),
    splitting = NoSplitting(),
    saveat = 1.0,
    progress = true,
    progress_message = (dt, u, p, t) -> t,
)
