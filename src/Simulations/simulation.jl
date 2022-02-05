struct Simulation{M <: AbstractModel, I, C, R <: AbstractRestart}
    model::M
    integrator::I
    callbacks::C
    restart::R
end

"""
    Simulation(
        model,
        method;
        Y_init = nothing,
        dt,
        tspan,
        callbacks = nothing,
        restart = NoRestart(),
    )

Construct a `Simulation` for a `model` with a time stepping `method`,
initial conditions `Y_init`, and time step `dt` for a time interval of `tspan`.
If `Y_init` is not provided, the model's default initial conditions are used.
```
"""
function Simulation(
    model::AbstractModel,
    method;
    dt,
    tspan,
    callbacks = nothing,
    restart = NoRestart(),
)
    Y = default_initial_conditions(model)

    # contains all information about the 
    # pde systems jacobians and right-hand sides
    # to hook into the OrdinaryDiffEq.jl interface
    ode_function = make_ode_function(model)

    # we use the OrdinaryDiffEq.jl interface
    # to set up and an ODE integrator that handles
    # integration in time and callbacks
    ode_problem = ODE.ODEProblem(ode_function, Y, tspan)
    integrator = ODE.init(ode_problem, method, dt = dt, callback = callbacks)

    restart = restart
    return Simulation(model, integrator, callbacks, restart)
end

"""
    set!(simulation, subcomponent = :base; kwargs...)

Set the simulation state to a new state, either through an array or a function. Defaults
to `:base` component for model.
"""
function set!(simulation::Simulation, subcomponent = :base; kwargs...)
    for (varname, f) in kwargs
        # let's make sure that the variable we are setting is actually in
        # the model's state vector to give a more informative error message
        if varname ∉
           getproperty(Models.variable_names(simulation.model), subcomponent)
            throw(ArgumentError("$varname not in state vector subcomponent $subcomponent."))
        end

        # for restart we need to use the reinit function because we don't 
        # have direct state access using OrdinaryDiffEq. For this
        # we need to copy
        if simulation.restart isa NoRestart
            Y = copy(simulation.integrator.u)
            time_final = simulation.integrator.sol.prob.tspan[2]
        else
            filename = simulation.restart.restartfile
            restart_data = load(filename)
            @assert "model" in keys(restart_data)
            if restart_data["model"] == simulation.model
                new_end_time = simulation.restart.end_time
                restart_integrator = restart_data["integrator"]
                Y = copy(restart_integrator.u)
                time_final = new_end_time
            else
                throw(ArgumentError("Restart file needs a compatible model."))
            end
        end

        # if we want to set a field in the state vector, we first need to extract
        # from the state, for this we need `getproperty` to be general.
        # get fields for this subcomponent from integrator state
        # Ex.: Y = (:base = base_fieldvector, :thermodynamics = thermo_field_vector,)
        # Y.thermodynamics = (:ρθ = pottempfield,)
        subcomponent_name = getproperty(Y, subcomponent)
        target_field = getproperty(subcomponent_name, varname)

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
        # have direct state access using OrdinaryDiffEq
        ODE.reinit!(
            simulation.integrator,
            Y;
            t0 = simulation.integrator.t,
            tf = time_final,
            erase_sol = false,
            reset_dt = false,
            reinit_callbacks = true,
            initialize_save = false,
            reinit_cache = false,
        )
    end

    nothing
end

"""
    get_spaces(simulation::Simulation, subcomponent = :base; args...)

Extract the function space for each field in `args` within model subcomponents `subcomponent`.
"""
function get_spaces(simulation::Simulation, subcomponent_name = :base, args...)
    spaces = []
    for varname in args
        if varname ∉
           getproperty(state_variable_names(simulation.model), subcomponent)
            throw(ArgumentError("$varname not in state vector subcomponent $subcomponent."))
        end
        subcomponent = getproperty(simulation.integrator.u, subcomponent_name)
        target_field = getproperty(subcomponent, varname)
        target_space = axes(target_field)
        push!(spaces, target_space)
    end

    return NamedTuple{args}(spaces)
end

"""
    step!(simulation, args...; kwargs...)

Advance the simulation by one time step.
"""
step!(simulation::Simulation, args...; kwargs...) =
    ODE.step!(simulation.integrator, args...; kwargs...)

"""
    run!(simulation, args...; kwargs...)

Run the simulation to completion.
"""
run!(simulation::Simulation, args...; kwargs...) =
    ODE.solve!(simulation.integrator, args...; kwargs...)

function Base.show(io::IO, s::Simulation)
    print(
        io,
        "Simulation set-up:\n\tmodel type:\t\t",
        typeof(s.model).name.name,
    )
    print(
        io,
        "\n\tthermodynamic tar:\t",
        typeof(s.model.thermodynamics),
        "\n\n",
    )
    show(io, s.model.domain)
    print(io, "\n\nTimestepper set-up:")
    print(io, "\n\tmethod:\t", typeof(s.integrator.alg).name.name)
    @printf(io, "\n\tdt:\t%f", s.integrator.dt)
    @printf(io, "\n\ttspan:\t(%f, %f)", s.integrator.sol.prob.tspan...)
end
