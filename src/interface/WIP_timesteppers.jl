abstract type AbstractTimestepper end

Base.@kwdef struct TimeStepper <: AbstractTimestepper
    method
    dt
    tspan
    saveat 
    progress
    progress_message
end