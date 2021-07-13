abstract type AbstractRate end
abstract type AbstractSplitting end

struct NoSplitting <: AbstractSplitting end

Base.@kwdef struct IMEXSplitting{𝒜} <: AbstractSplitting
    ref_state::𝒜
end

# TODO: Add more methods here such as MultiRate, Explicit [can't reuse word]
Base.@kwdef struct IMEX{ℱ}
    method::ℱ
end

function IMEX()
    return IMEX(ARK2GiraldoKellyConstantinescu)
end

function construct_odesolver(::NoSplitting, simulation)
    method        = simulation.timestepper.method
    start         = simulation.timestepper.start
    timestep      = simulation.timestepper.timestep
    rhs           = simulation.rhs
    state         = simulation.state

    ode_solver = method(
        rhs,
        state;
        dt = timestep,
        t0 = start,
    )
    return ode_solver
end
