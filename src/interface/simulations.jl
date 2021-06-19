abstract type AbstractSimulation end

Base.@kwdef struct Simulation{𝒜,ℬ,𝒞,𝒟,ℰ,ℱ,𝒢} <: AbstractSimulation
    backend::𝒜
    model::ℬ
    timestepper::𝒞
    callbacks::𝒟
    grid::ℰ 
    rhs::ℱ
    state::𝒢
end

function Simulation(;
    backend::AbstractBackend, 
    model::ModelSetup, 
    timestepper,
    callbacks,
)
    grid = create_grid(backend)
    rhs = create_rhs(model, backend, grid = grid)
    state = create_init_state(model, backend, rhs = rhs)

    return Simulation(
        backend, 
        model, 
        timestepper, 
        callbacks, 
        grid, 
        rhs, 
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

    # # Perform evolution of simulations
    # if isempty(cbvector)
    #     solve!(
    #         state, 
    #         odesolver; 
    #         timeend = finish, 
    #         adjustfinalstep = false,
    #     )
    # else
    #     solve!(
    #         state,
    #         odesolver;
    #         timeend = finish,
    #         callbacks = cbvector,
    #         adjustfinalstep = false,
    #     )
    # end
end

# TODO!: Awaits implementation
# function evolve!(simulation::Simulation{ClimateMachineCoreBackend})
# end