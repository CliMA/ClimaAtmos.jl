"""
    Simulation <: AbstractSimulation
"""
struct Simulation{MT,TT,OT} <: AbstractSimulation
    model::MT
    timestepper::TT
    ode_problem::OT
end

"""
    Simulation(model::AbstractModel, stepper::AbstractTimestepper)
"""
function Simulation(; model::AbstractModel, stepper::AbstractTimestepper)
    ode_function = make_ode_function(model)
    Y = default_initial_conditions(model)
    ode_problem = DiffEqBase.ODEProblem(ode_function, Y, stepper.tspan)

    return Simulation(model, stepper, ode_problem)
end

"""
    init(sim::AbstractSimulation)
"""
function init(sim::AbstractSimulation)
    return DiffEqBase.init(
        sim.ode_problem,
        sim.timestepper.method,
        dt = sim.timestepper.dt,
        saveat = sim.timestepper.saveat,
        progress = sim.timestepper.progress, 
        progress_message = sim.timestepper.progress_message,
    )
end

"""
    run(sim::AbstractSimulation)
"""
function run(sim::AbstractSimulation)
    return DiffEqBase.solve(
        sim.ode_problem,
        sim.timestepper.method,
        dt = sim.timestepper.dt,
        saveat = sim.timestepper.saveat,
        progress = sim.timestepper.progress, 
        progress_message = sim.timestepper.progress_message,
    )
end