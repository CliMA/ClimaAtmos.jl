"""
    JLD2Output{M, I} <: AbstractCallback

Specifies that a `DiffEqCallbacks.PeriodicCallback` should be constructed that
extracts the model state from the integrator and stores it in a `.jld2` file. 
"""
struct JLD2Output{M <: AbstractModel, I <: Number} <: AbstractCallback
    model::M
    filedir::String
    filename::String
    interval::I
end

function (F::JLD2Output)(integrator)
    # Create directory
    mkpath(F.filedir)
    # Save data
    savefile = joinpath(F.filedir, F.filename * "_$(integrator.t)" * ".jld2")
    jldsave(savefile, integrator = integrator, model = F.model)
    return nothing
end

function generate_callback(F::JLD2Output; kwargs...)
    return PeriodicCallback(F, F.interval; initial_affect = true, kwargs...)
end
