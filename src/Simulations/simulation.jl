using OrderedCollections: OrderedDict

using ClimaSimulations: evaluate_callbacks!, evaluate_output_writers!, @stopwatch, run!
using ClimaSimulations: prettytime

import ClimaSimulations: Simulation, time_step!, initialize_simulation!, stop_time_exceeded

function Simulation(model::AbstractModel, time_stepping_method;
    Δt,
    tspan,
    initial_state = nothing,
    callbacks = nothing,
    stop_criteria = Any[stop_time_exceeded])

    # inital state is either default or set externally 
    Y = initial_state isa Nothing ? default_initial_conditions(model) : initial_state

    # default_initial_conditions(::Nothing, model) = default_initial_conditions(model)
    # Y = default_initial_conditions(initial_state, model)

    # contains all information about the 
    # pde systems jacobians and right-hand sides
    # to hook into the DiffEqBase.jl interface
    ode_function = make_ode_function(model)

    # Convert numbers to correct floating point type
    FT = eltype(model.parameters)

    tspan = FT.(tspan)
    Δt = FT(Δt)
    stop_time = FT(tspan[2])

    # we use the DiffEqBase.jl interface
    # to set up and an ODE timestepper that handles
    # integration in time and callbacks
    ode_problem = DiffEqBase.ODEProblem(ode_function, Y, tspan)
    timestepper =
        DiffEqBase.init(ode_problem, time_stepping_method, dt = Δt, callback = callbacks)

    output_writers = OrderedDict{Symbol, Any}()
    callbacks = OrderedDict{Symbol, Any}()

    return Simulation(model, timestepper, Δt, stop_criteria, Inf, stop_time,
                      Inf, nothing, output_writers, callbacks, 0.0, false, false)
end

"""
    set!(
        simulation::AbstractSimulation,
        submodel_name = nothing;
        kwargs...,
    )

Set the `simulation` state to a new state, either through 
an array or a function.
"""
function set!(
    simulation::Simulation,
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
        Y = copy(simulation.timestepper.u)

        # get fields for this submodel from timestepper state
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
            simulation.timestepper,
            Y;
            t0 = simulation.timestepper.t,
            tf = simulation.timestepper.sol.prob.tspan[2],
            erase_sol = false,
            reset_dt = false,
            reinit_callbacks = true,
            initialize_save = false,
            reinit_cache = false,
        )
    end

    nothing
end

const ClimaAtmosSimulation = Simulation{<:AbstractModel}

initialize_simulation!(sim::ClimaAtmosSimulation, pickup=nothing) = sim.initialized = true

function stop_time_exceeded(sim::ClimaAtmosSimulation)
    current_time = sim.timestepper.t
    if current_time >= sim.stop_time
          @info "Simulation is stopping. Model time $(prettytime(current_time)) " *
                "has hit or exceeded simulation stop time $(prettytime(sim.stop_time))."
          return true
    end
    return false
end

#=
"""
    @stopwatch sim expr

Increment sim.stopwatch with the execution time of expr.
"""
macro stopwatch(sim, expr)
    return esc(quote
       local time_before = time_ns() * 1e-9
       local output = $expr
       local time_after = time_ns() * 1e-9
       sim.wall_time += time_after - time_before
       output
   end)
end
=#

"""
    time_step!(simulation::AbstractSimulation, args...; kwargs...)

Step forward a `simulation` one time step.
"""
function time_step!(sim::ClimaAtmosSimulation, args...; kwargs...)

    initialization_step = !(sim.initialized)

    if initialization_step
        @stopwatch sim initialize_simulation!(sim)
        start_time = time_ns()
        @info "Executing first time step..."
    end

    @stopwatch sim DiffEqBase.step!(sim.timestepper, args...; kwargs...)
    
    if initialization_step
        elapsed_first_step_time = prettytime(1e-9 * (time_ns() - start_time))
        @info "    ... first time step complete ($elapsed_first_step_time)."
    end

    @stopwatch sim begin
        evaluate_callbacks!(sim)
        evaluate_output_writers!(sim)
    end

    return nothing
end

function Base.show(io::IO, s::ClimaAtmosSimulation)
    println(io, "Simulation set-up:")
    @printf(io, "\tmodel type:\t%s\n", typeof(s.model).name.name)
    @printf(io, "\tmodel vars:\t%s\n\n", s.model.varnames)
    show(io, s.model.domain)
    println(io, "\nTimestepper set-up:")
    @printf(io, "\tmethod:\t%s\n", typeof(s.timestepper.alg).name.name)
    @printf(io, "\tdt:\t%f\n", s.timestepper.dt)
    @printf(
        io,
        "\ttspan:\t(%f, %f)\n",
        s.timestepper.sol.prob.tspan[1],
        s.timestepper.sol.prob.tspan[2],
    )
end
