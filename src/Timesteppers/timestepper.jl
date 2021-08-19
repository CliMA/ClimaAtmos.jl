"""
    TimeStepper <: AbstractTimestepper
"""
Base.@kwdef struct Timestepper{MT,DT,TT,SAT,PT,PMT} <: AbstractTimestepper
    method::MT
    dt::DT
    tspan::TT
    saveat::SAT
    progress::PT = true
    progress_message::PMT = (dt, u, p, t) -> t
end