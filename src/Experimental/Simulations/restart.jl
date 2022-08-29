"""
    NoRestart <: AbstractRestart

Specifies that a simulation should use its original initial conditions and end
time.
"""
struct NoRestart <: AbstractRestart end

"""
    Restart <: AbstractRestart

Specifies that a simulation should begin from the state recorded in a restart
file and end at a specific time. The restart file must be a `.jld2` file
containing the `simulation.integrator` and `simulation.model` objects. Users
must `set!` the simulation prior to restart.
"""
Base.@kwdef struct Restart{T <: Real} <: AbstractRestart
    restartfile::String
    end_time::T
end
