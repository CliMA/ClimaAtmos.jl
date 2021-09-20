
"""
    NoRestart <: AbstractRestart
Empty container, default method for no restarts, begin
simulation from model initial conditions. 
"""
struct NoRestart <: AbstractRestart end

"""
    Restart <: AbstractRestart
Container for restart file parameters and updated
simulation end time. Restart file must be a .jld2
file containing the `simulation.integrator` and 
`simulation.model` objects. User must `set!` simulation
prior to restart.
"""
struct Restart <: AbstractRestart
    restartfile::String
    end_time::Real
end

function Restart(; restartfile = nothing, end_time = nothing)
    # Check valid filename input for restarts
    @assert restartfile isa String
    # Check new end time (tspan[2]) is positive
    @assert end_time > 0.0
    return Restart(restartfile, end_time)
end
