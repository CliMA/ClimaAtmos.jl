abstract type AbstractSimulation end

Base.@kwdef struct Simulation{𝒜,ℬ,𝒞,𝒞𝒞,𝒟,ℰ,ℱ,𝒢} <: AbstractSimulation
    backend::𝒜
    discretized_domain::ℬ 
    model::𝒞
    splitting::𝒞𝒞
    timestepper::𝒟
    callbacks::ℰ
    rhs::ℱ
    state::𝒢
end

function Simulation(;
    backend::AbstractBackend,
    discretized_domain::DiscretizedDomain, 
    model::ModelSetup,
    splitting::AbstractSplitting = NoSplitting(), 
    timestepper,
    callbacks,
)
    grid = create_grid(backend, discretized_domain)
    rhs = create_rhs(splitting, model, backend, domain = discretized_domain.domain, grid = grid)
    if rhs isa SpaceDiscretization
        state = create_init_state(model, backend, rhs = rhs)
    elseif rhs isa Tuple 
        state = create_init_state(model, backend, rhs = rhs[1]) # what if rhs is not array??
    else
        println("rhs error => fail to initialize state")
    end

    return Simulation(
        backend,
        discretized_domain,
        model,
        splitting,
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
            discretized_domain = simulation.discretized_domain,
            model = simulation.model,
            splitting = simulation.splitting, 
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
    # ode_solver = method(
    #     rhs,
    #     state;
    #     dt = timestep,
    #     t0 = start,
    # )

    ode_solver = construct_odesolver(simulation.splitting, simulation)

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