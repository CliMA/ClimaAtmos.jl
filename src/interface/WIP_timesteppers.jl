abstract type AbstractTimestepper end
abstract type AbstractRate end
abstract type AbstractSplitting end

Base.@kwdef struct TimeStepper{A,B,C,D,E,F,G} <: AbstractTimestepper
    method::A
    dt::B
    tspan::C
    splitting::D = NoSplitting()
    saveat::E
    progress::F
    progress_message::G
end

struct NoSplitting <: AbstractSplitting end

Base.@kwdef struct IMEXSplitting{A,B,C} <: AbstractSplitting
    linear_model::A = :linear
    implicit_method::B = LinearBackwardEulerSolver(ManyColumnLU(); isadjustable = false)
    split_explicit_implicit::C = false
end

# TODO: Add more methods here such as MultiRate, Explicit [can't reuse word]
Base.@kwdef struct IMEX{A}
    method::A
end

function IMEX()
    return IMEX(ARK2GiraldoKellyConstantinescu)
end

