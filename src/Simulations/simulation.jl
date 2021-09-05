"""
    struct Simulation <: AbstractSimulation
        # a ClimaAtmos model
        model::AbstractModel
        # a DiffEqBase.jl integrator used for time
        # stepping the simulation
        integrator::DiffEqBase.DEIntegrator
    end

    A simulation wraps an abstract ClimaAtmos model containing 
    equation specifications and an instance of an integrator used for
    time integration of the discretized model PDE.
"""
struct Simulation <: AbstractSimulation
    model::AbstractModel
    integrator::DiffEqBase.DEIntegrator
end

"""
    Simulation(model::AbstractModel, method::AbstractTimestepper; dt, tspan, init_state, kwargs...)
"""
function Simulation(model::AbstractModel, method; Y_init = nothing, dt, tspan)
    # inital state is either default or set externally 
    Y = Y_init isa Nothing ? default_initial_conditions(model) : Y_init

    # contains all information about the 
    # pde systems jacobians and right-hand sides
    # to hook into the DiffEqBase.jl interface
    ode_function = make_ode_function(model)

    # we use the DiffEqBase.jl interface
    # to set up and an ODE integrator that handles
    # integration in time and callbacks
    ode_problem = DiffEqBase.ODEProblem(ode_function, Y, tspan)
    integrator = DiffEqBase.init(ode_problem, method, dt = dt)

    return Simulation(model, integrator)
end

"""
    set!
"""
function set!(
    simulation::AbstractSimulation,
    submodel_name = nothing;
    kwargs...,
)
    for (varname, f) in kwargs
        if varname ∉ simulation.model.varnames
            throw(ArgumentError("$varname not in model variables."))
        end

        # we need to use the reinit function because we don't 
        # have direct state access using DiffEqBase. For this
        # we need to copy
        Y = copy(simulation.integrator.u)

        # get fields for this submodel from integrator state
        if submodel_name === nothing
            # default behavior if model has no submodels but itself
            submodel_field = getproperty(Y, simulation.model.name)
        else
            submodel_field = getproperty(Y, submodel_name)
        end

        # get pointer to target field 
        target_field = getproperty(submodel_field, varname)

        # sometimes we want to set with a function and sometimes with an
        # Field. This supports both behaviors.
        if f isa Function
            space = axes(target_field)
            local_geometry = Fields.local_geometry_field(space)
            target_field .= f.(local_geometry)
        elseif f isa Fields.Field # ClimaCore Field
            target_field .= f
        else
            throw(ArgumentError("$varname not a compatible type."))
        end

        # we need to use the reinit function because we don't 
        # have direct state access using DiffEqBase
        DiffEqBase.reinit!(
            simulation.integrator,
            Y;
            t0 = simulation.integrator.t,
            tf = simulation.integrator.sol.prob.tspan[2],
            erase_sol = false,
            reset_dt = false,
            reinit_callbacks = false,
            initialize_save = false,
            reinit_cache = false,
        )
    end

    nothing
end

step!(sim::AbstractSimulation, args...; kwargs...) =
    DiffEqBase.step!(sim.integrator, args...; kwargs...)

run!(sim::AbstractSimulation, args...; kwargs...) =
    DiffEqBase.solve!(sim.integrator, args...; kwargs...)
