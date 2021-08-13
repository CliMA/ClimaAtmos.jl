abstract type DGProblem end
Base.@kwdef struct DGOEDProblem{𝒜,ℬ} <: DGProblem
    rhs::𝒜
    state::ℬ
end

function create_ode_problem(backend::DiscontinuousGalerkinBackend, model, timestepper)
    grid = create_grid(backend, model.domain)
    rhs = create_rhs(timestepper.splitting, model, backend, domain = model.domain.domain, grid = grid)
    if rhs isa SpaceDiscretization
        state = create_init_state(model, backend, rhs = rhs)
    elseif rhs isa Tuple 
        state = create_init_state(model, backend, rhs = rhs[1]) # what if rhs is not array??
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

function construct_odesolver(splitting::IMEXSplitting, simulation; t0 = 0, split_explicit_implicit = false)
    method       = simulation.timestepper.method.method
    start        = simulation.timestepper.tspan[1]
    timestep     = simulation.timestepper.dt
    state        = simulation.ode_problem.state 

    explicit_rhs    = simulation.ode_problem.rhs[1]
    implicit_rhs    = simulation.ode_problem.rhs[2]

    implicit_method         = splitting.implicit_method
    split_explicit_implicit = splitting.split_explicit_implicit

    odesolver = method(
        explicit_rhs,
        implicit_rhs,
        implicit_method,
        state;
        dt = timestep,
        t0 = start,
        split_explicit_implicit = split_explicit_implicit,
    )

    return odesolver
end