abstract type AbstractCallback end

struct Default <: AbstractCallback end
struct Info <: AbstractCallback end
struct CFL <: AbstractCallback end

struct StateCheck{T} <: AbstractCallback
    number_of_checks::T
end

Base.@kwdef struct JLD2State{T, V, B} <: AbstractCallback
    iteration::T
    filepath::V
    overwrite::B = true
end

Base.@kwdef struct VTKState{T, V, C, B} <: AbstractCallback
    iteration::T = 1
    filepath::V = "."
    counter::C = [0]
    overwrite::B = true
end

Base.@kwdef struct PositivityPreservingCallback{ℱ} <: AbstractCallback 
    filterstates::ℱ = 6:6
end

Base.@kwdef struct ReferenceStateUpdate{ℱ} <: AbstractCallback 
    recompute::ℱ = 20
end

function create_callbacks(simulation::Simulation, odesolver)
    callbacks = simulation.callbacks

    if isempty(callbacks)
        return ()
    else
        cbvector = [
            create_callback(callback, simulation, odesolver)
            for callback in callbacks
        ]
        return tuple(cbvector...)
    end
end

function create_callback(::Default, simulation::Simulation, odesolver)
    cb_info = create_callback(Info(), simulation, odesolver)
    cb_state_check = create_callback(StateCheck(10), simulation, odesolver)

    return (cb_info, cb_state_check)
end