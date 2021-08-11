abstract type AbstractTimestepper end
abstract type AbstractRate end
abstract type AbstractSplitting end

Base.@kwdef struct TimeStepper <: AbstractTimestepper
    method
    dt
    tspan
    splitting = NoSplitting()
    saveat 
    progress
    progress_message
end

struct NoSplitting <: AbstractSplitting end

Base.@kwdef struct IMEXSplitting{𝒜,ℬ,𝒞} <: AbstractSplitting
    linear_model::𝒞 = :linear
    implicit_method::𝒜 = LinearBackwardEulerSolver(ManyColumnLU(); isadjustable = false)
    split_explicit_implicit::ℬ = false
end

# TODO: Add more methods here such as MultiRate, Explicit [can't reuse word]
Base.@kwdef struct IMEX{ℱ}
    method::ℱ
end

function IMEX()
    return IMEX(ARK2GiraldoKellyConstantinescu)
end

