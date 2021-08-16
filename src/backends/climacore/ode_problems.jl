# Default behavior for this backend
function create_ode_problem(backend::ClimaCoreBackend, model::AbstractModel, timestepper::AbstractTimestepper)
    function_space = create_function_space(backend, model.domain)
    rhs! = create_rhs(backend, model, function_space)
    y0 = create_initial_conditions(backend, model, function_space)

    return ODEProblem(
        rhs!, 
        y0, 
        timestepper.tspan,
    )
end