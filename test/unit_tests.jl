using ClimaAtmos
using ClimaAtmos.Utils
using ClimaAtmos.Interface
using ClimaAtmos.Backends

# explicit imports from Interface
import ClimaAtmos.Interface: PeriodicRectangle
import ClimaAtmos.Interface: BarotropicFluidModel
import ClimaAtmos.Interface: TimeStepper
import ClimaAtmos.Interface: DirichletBC, Simulation

# explicit imports from Backends
import ClimaAtmos.Backends: create_ode_problem


# External Stuff
using IntervalSets
using OrdinaryDiffEq: ODEProblem, solve, SSPRK33

# Utils
ϕ̂(1,1,1)

# Domains
domain = PeriodicRectangle(
    xlim = -2π..2π, 
    ylim = -2π..2π, 
    nelements = (16, 16), 
    npolynomial = 4, 
)

# Model
model = BarotropicFluidModel( 
    domain = domain, 
    boundary_conditions = nothing, 
    initial_conditions = nothing, 
    parameters = nothing,
)

# Timestepper
timestepper = TimeStepper(
    method = SSPRK33(),
    dt = 0.04,
    tspan = (0.0, 10.0),
    saveat = 1.0,
    progress = true,
    progress_message = (dt, u, p, t) -> t,
)

# Boundary conditions
DirichletBC()
