abstract type DGProblem end
Base.@kwdef struct DGOEDProblem{𝒜,ℬ} <: DGProblem
    rhs::𝒜
    state::ℬ
end

function create_ode_problem(backend::DiscontinuousGalerkinBackend, model, timestepper)
    grid = create_grid(backend, model.domain)
    rhs = create_rhs(timestepper.splitting, model, backend, grid = grid)
    if rhs isa SpaceDiscretization
        state = create_init_state(model, backend, rhs = rhs)
    elseif rhs isa Tuple 
        state = create_init_state(model, backend, rhs = rhs[1]) 
    else
        println("rhs error => fail to initialize state")
    end

    return DGOEDProblem(
        rhs = rhs,
        state = state,
    )
end

function construct_odesolver(::NoSplitting, simulation)
    method        = simulation.timestepper.method
    start         = simulation.timestepper.tspan[1]
    timestep      = simulation.timestepper.dt
    rhs           = simulation.ode_problem.rhs
    state         = simulation.ode_problem.state

    ode_solver = method(
        rhs,
        state;
        dt = timestep,
        t0 = start,
    )

    return ode_solver
end
