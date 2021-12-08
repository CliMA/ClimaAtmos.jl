struct Simulation{
    M <: AbstractModel,
    I <: DiffEqBase.DEIntegrator,
    C <: Union{Nothing, DiffEqBase.DiscreteCallback, DiffEqBase.CallbackSet},
    R <: AbstractRestart,
}
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

# Example
```jldoctest; setup = :(using ClimaAtmos.Simulations)
julia> using OrdinaryDiffEq: Euler

julia> using ClimaAtmos.Domains, ClimaAtmos.Models.ShallowWaterModels

julia> domain =
        Plane(xlim = (-2π, 2π), ylim = (-2π, 2π), nelements = (16, 16), npolynomial = 3);

julia> parameters = (g = 9.8, D₄ = 1e-4, ϵ = 0.1, l = 0.5, k = 0.5, h₀ = 1.0);

julia> model = ShallowWaterModel(; domain, parameters);

julia> Simulation(model, Euler(), dt = 0.04, tspan = (0.0, 80.0))
Simulation set-up:
\tmodel type:\tShallowWaterModel
\tmodel vars:\t(:h, :u, :c)

Domain set-up:
\txy-plane:\t[-6.3, 6.3) × [-6.3, 6.3)
\t# of elements:\t(16, 16)
\tpoly order:\t3

Timestepper set-up:
\tmethod:\tEuler
\tdt:\t0.040000
\ttspan:\t(0.000000, 80.000000)
```
"""
function Simulation(
    model::AbstractModel,
    method;
    Y_init = nothing,
    dt,
    tspan,
    callbacks = nothing,
    restart = NoRestart(),
)

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
    integrator =
        DiffEqBase.init(ode_problem, method, dt = dt, callback = callbacks)

    restart = restart
    return Simulation(model, integrator, callbacks, restart)
end

"""
    set!(simulation, submodel_name = nothing; kwargs...)

Set the simulation state to a new state, either through an array or a function.
"""
function set!(simulation::Simulation, submodel_name = nothing; kwargs...)
    for (varname, f) in kwargs
        if varname ∉ simulation.model.varnames
            throw(ArgumentError("$varname not in model variables."))
        end

        # we need to use the reinit function because we don't 
        # have direct state access using DiffEqBase. For this
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
using ClimaCore.DataLayouts: column

"""
    step!(simulation, args...; kwargs...)

Advance the simulation by one time step.
"""
step!(simulation::Simulation, args...; kwargs...) =
    DiffEqBase.step!(simulation.integrator, args...; kwargs...)

"""
    run!(simulation, args...; kwargs...)

Run the simulation to completion.
"""
run!(simulation::Simulation, args...; kwargs...) =
    DiffEqBase.solve!(simulation.integrator, args...; kwargs...)

function Base.show(io::IO, s::Simulation)
    print(io, "Simulation set-up:\n\tmodel type:\t", typeof(s.model).name.name)
    print(io, "\n\tmodel vars:\t", s.model.varnames, "\n\n")
    show(io, s.model.domain)
    print(io, "\n\nTimestepper set-up:")
    print(io, "\n\tmethod:\t", typeof(s.integrator.alg).name.name)
    @printf(io, "\n\tdt:\t%f", s.integrator.dt)
    @printf(io, "\n\ttspan:\t(%f, %f)", s.integrator.sol.prob.tspan...)
end
