abstract type AbstractTimestepper end
abstract type AbstractSplitting end

"""
    TimeStepper <: AbstractTimestepper
"""
Base.@kwdef struct TimeStepper{A,B,C,D,E,F,G} <: AbstractTimestepper
    splitting::A = NoSplitting()
    method::B
    dt::C
    tspan::D
    saveat::E
    progress::F
    progress_message::G
end

"""
    NoSplitting  <: AbstractSplitting
"""
struct NoSplitting <: AbstractSplitting end