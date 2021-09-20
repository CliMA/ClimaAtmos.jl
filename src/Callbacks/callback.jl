"""
    JLD2Output <: AbstractCallback
    Container for JLD2 output callback (writes to disk)
"""
struct JLD2Output <: AbstractCallback
    model::AbstractModel
    filedir::String
    filename::String
    interval::Number
end

function (F::JLD2Output)(integrator)
    # Create directory
    mkpath(F.filedir)
    # Save data
    savefile = joinpath(F.filedir, F.filename * "_$(integrator.t)" * ".jld2")
    jldsave(savefile, integrator = integrator, model = F.model)
    return nothing
end

"""
    generate_callback(F::JLD2Output; kwargs...)

    Creates a PeriodicCallback object that extracts solution
    variables from the integrator and stores them in a jld2 file. 
"""
function generate_callback(F::JLD2Output; kwargs...)
    return PeriodicCallback(F, F.interval; initial_affect = true, kwargs...)
end
