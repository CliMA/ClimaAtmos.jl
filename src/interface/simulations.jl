abstract type AbstractSimulation end

Base.@kwdef struct Simulation{𝒜,ℬ,𝒞,𝒟,ℰ,ℱ,𝒢} <: AbstractSimulation
    backend::𝒜
    model::ℬ
    timestepper::𝒞
    callbacks::𝒟
    rhs::ℰ
    grid::ℱ 
    state::𝒢
end

function Simulation(
    backend::AbstractBackend, 
    model::ModelSetup, 
    timestepper,
    callbacks,
)
    grid = create_grid(model, backend)
    rhs = create_rhs(model, backend)
    state = initialize_state(model, backend)

    return Simulation(
        backend, 
        model, 
        timestepper, 
        callbacks, 
        rhs, 
        grid, 
        state
    )
end

function initialize!(simulation::Simulation; overwrite = false)
    if overwrite
        simulation = Simulation(
            backend = simulation.backend,
            model = simulation.model, 
            timestepper = simulation.timestepper, 
            callbacks = simulation.callbacks,
        )
    end
end

function evolve!(simulation::Simulation{DiscontinuousGalerkinBackend})
    method        = simulation.timestepper.method
    start         = simulation.timestepper.start
    finish        = simulation.timestepper.finish
    timestep      = simulation.timestepper.timestep
    rhs           = simulation.rhs
    state         = simulation.state

    # Instantiate time stepping method & create callbacks
    ode_solver = construct_odesolver(method, rhs, state, timestep, t0 = start) 
    cb_vector = create_callbacks(simulation, odesolver)

    # Perform evolution of simulations
    if isempty(cbvector)
        solve!(
            state, 
            odesolver; 
            timeend = finish, 
            adjustfinalstep = false,
        )
    else
        solve!(
            state,
            odesolver;
            timeend = finish,
            callbacks = cbvector,
            adjustfinalstep = false,
        )
    end

    return nothing
end

# TODO!: Awaits implementation
# function evolve!(simulation::Simulation{ClimateMachineCoreBackend})
# end