abstract type AbstractSimulation end

struct Simulation{A,B,C,D,E} <: AbstractSimulation
    backend::A
    model::B
    timestepper::C
    callbacks::D
    ode_problem::E
end

function Simulation(
    backend::AbstractBackend;
    model,
    timestepper::AbstractTimestepper,
    callbacks,
)
    ode_problem = create_ode_problem(
        backend, 
        model,
        timestepper, 
    )
  
    return Simulation(
        backend,
        model,
        timestepper, 
        callbacks, 
        ode_problem,
    )
end

function evolve(simulation::Simulation)
    return solve(
        simulation.ode_problem,
        simulation.timestepper.method,
        dt = simulation.timestepper.dt,
        saveat = simulation.timestepper.saveat,
        progress = simulation.timestepper.progress, 
        progress_message = simulation.timestepper.progress_message,
    )
en