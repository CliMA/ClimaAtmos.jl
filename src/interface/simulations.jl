abstract type AbstractSimulation end

struct Simulation{A,B,C,D,E} <: AbstractSimulation
    backend::A
    model::B
    timestepper::C
    callbacks::D
    ode_problem::E
end