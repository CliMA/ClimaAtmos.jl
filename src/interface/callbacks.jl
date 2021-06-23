abstract type AbstractCallback end

struct Info <: AbstractCallback end
struct CFL <: AbstractCallback end

Base.@kwdef struct StateCheck{𝒜} <: AbstractCallback
    number_of_checks::𝒜
end

Base.@kwdef struct VTKState{𝒜,ℬ,𝒞,𝒟} <: AbstractCallback
    iteration::𝒜 = 1
    filepath::ℬ = "."
    counter::𝒞 = [0]
    overwrite::𝒟 = true
end

Base.@kwdef struct JLD2State{𝒜,ℬ,𝒞} <: AbstractCallback
    iteration::𝒜
    filepath::ℬ
    overwrite::𝒞 = true
end

Base.@kwdef struct PositivityPreservingCallback{𝒜} <: AbstractCallback 
    filterstates::𝒜 = 6:6
end

Base.@kwdef struct ReferenceStateUpdate{𝒜} <: AbstractCallback 
    recompute::𝒜 = 20
end

function create_callbacks(simulation::Simulation, ode_solver)
    callbacks = simulation.callbacks

    if isempty(callbacks)
        return ()
    else
        cbvector = [
            create_callback(callback, simulation, ode_solver)
            for callback in callbacks
        ]
        return tuple(cbvector...)
    end
end