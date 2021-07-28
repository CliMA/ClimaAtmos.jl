abstract type AbstractSimulation end

struct Simulation{A,B,C,D,E} <: AbstractSimulation
    backend::A
    model::B
    timestepper::C
    callbacks::D
    ode_problem::E
end

function Simulation(;
    backend::AbstractBackend,
    model::ModelSetup,
    timestepper::AbstractTimestepper,
    callbacks,
)
    # make the rhs! evaluation functions
    ode_problem = create_ode_problem(
        backend, 
        domain, 
        model,
        timestepper, 
        callbacks
    )
  
    return Simulation(
        backend,
        model,
        timestepper, 
        callbacks, 
        ode_problem,
    )
end

