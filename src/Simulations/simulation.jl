"""
    Simulation <: AbstractSimulation
"""
struct Simulation <: AbstractSimulation
    model::AbstractModel
    integrator::DiffEqBase.DEIntegrator
end

"""
    Simulation(model::AbstractModel, method::AbstractTimestepper; dt, tspan, init_state, kwargs...)
"""
function Simulation(
    model::AbstractModel,
    method;
    Y_init = nothing,
    dt,
    tspan,
    kwargs...,
)
    # inital state is either default or set externally 
    Y = Y_init isa Nothing ? make_initial_conditions(model) : Y_init

    # contains all information about the 
    # pde systems jacobians and right-hand sides
    # to hook into the DiffEqBase.jl interface
    ode_function = make_ode_function(model)

    # we use the DiffEqBase.jl interface
    # to set up and an ODE integrator that handles
    # integration in time and callbacks
    ode_problem = DiffEqBase.ODEProblem(ode_function, Y, tspan)
    integrator = DiffEqBase.init(ode_problem, method, dt = dt, kwargs...)

    return Simulation(model, integrator)
end

step!(sim::AbstractSimulation, args...; kwargs...) =
    DiffEqBase.step!(sim.integrator, args...; kwargs...)

run!(sim::AbstractSimulation, args...; kwargs...) =
    DiffEqBase.solve!(sim.integrator, args...; kwargs...)

function set!(sim::AbstractSimulation; kwargs...)
    for (fldname, value) in kwargs
        if fldname ∈ propertynames(sim.integrator.u.prognostic)
            ϕ = getproperty(sim.integrator.u.prognostic, fldname)
        else
            throw(ArgumentError("name $fldname not found in prognostic state."))
        end
        if value isa Function
            space = axes(sim.integrator.u.prognostic) # function space for prog vars
            @unpack x1, x2 = coordinates = Fields.coordinate_field(space) # x, y, z
            ϕ .= value.(x1, x2)
        else
            value isa AbstractArray
            ϕ .= value
        end
    end
    # TODO! make diagnostic variables from prognostic variables
    # Models.diagnostic_from_prognostic!(
    #     sim.integrator.u.diagnostic, 
    #     sim.integrator.u.prognostic,
    #     sim.model,
    # )

    return nothing
end
