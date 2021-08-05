
function create_ode_problem(backend::ClimaCoreBackend, ::AbstractModel, ::AbstractTimestepper)
    error("Model or time stepper not supported by ClimaCoreBackend.")
end

function create_ode_problem(backend::ClimaCoreBackend, model, timestepper)
    function_space = create_function_space(backend, model.domain)
    rhs! = create_rhs(backend, model.equation_set, function_space)
    y0 = model.initial_conditions.(
        Fields.coordinate_field(function_space), 
        Ref(model.parameters)
    )

    return ODEProblem(
        rhs!, 
        y0, 
        timestepper.tspan,
    )
end