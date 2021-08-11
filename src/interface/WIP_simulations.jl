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

function evolve(simulation::Simulation{<:ClimaCoreBackend})
    return solve(
        simulation.ode_problem,
        simulation.timestepper.method,
        dt = simulation.timestepper.dt,
        saveat = simulation.timestepper.saveat,
        progress = simulation.timestepper.progress, 
        progress_message = simulation.timestepper.progress_message,
    )
end

function evolve(simulation::Simulation{<:DiscontinuousGalerkinBackend})
    method        = simulation.timestepper.method
    start         = simulation.timestepper.tspan[1]
    finish        = simulation.timestepper.tspan[2]
    timestep      = simulation.timestepper.dt
    splitting     = simulation.timestepper.splitting
    rhs           = simulation.ode_problem.rhs
    state         = simulation.ode_problem.state

    ode_solver = construct_odesolver(splitting, simulation)

    cb_vector = create_callbacks(simulation, ode_solver)

    # Perform evolution of simulations
    if isempty(cb_vector)
        solve!(
            state, 
            ode_solver; 
            timeend = finish, 
            adjustfinalstep = false,
        )
    else
        solve!(
            state,
            ode_solver;
            timeend = finish,
            callbacks = cb_vector,
            adjustfinalstep = false,
        )
    end
end