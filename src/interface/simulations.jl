abstract type AbstractSimulation end

Base.@kwdef struct Simulation{𝒜,ℬ,𝒞,𝒟,ℰ,ℱ,𝒢} <: AbstractSimulation
    backend::𝒜
    grid::ℬ 
    model::𝒞
    timestepper::𝒟
    callbacks::ℰ
    rhs::ℱ
    state::𝒢
end

function Simulation(;
    backend::AbstractBackend,
    grid, 
    model::ModelSetup, 
    timestepper,
    callbacks,
)
    rhs = create_rhs(model, grid, backend)
    state = create_init_state(model, backend, rhs = rhs)

    return Simulation(
        backend,
        grid,
        model, 
        timestepper, 
        callbacks, 
        rhs, 
        state
    )
end

function initialize!(simulation::Simulation; overwrite = false)
    if overwrite
        simulation = Simulation(
            backend = simulation.backend,
            grid = simulation.grid,
            model = simulation.model, 
            timestepper = simulation.timestepper, 
            callbacks = simulation.callbacks,
        )
    end
end

function evolve!(simulation::Simulation{<:DiscontinuousGalerkinBackend})
    method        = simulation.timestepper.method
    start         = simulation.timestepper.start
    finish        = simulation.timestepper.finish
    timestep      = simulation.timestepper.timestep
    rhs           = simulation.rhs
    state         = simulation.state

    # Instantiate time stepping method & create callbacks
    ode_solver = method(
        rhs,
        state;
        dt = timestep,
        t0 = start,
    )

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