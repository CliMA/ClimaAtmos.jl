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
    jldsave(joinpath(F.filedir, F.filename * ".jld2"), integrator = integrator)
    return nothing
end

"""
    generate_callback(F::JLD2Output; kwargs...)

    Creates a PeriodicCallback object that extracts solution
    variables from the integrator and stores them in a jld2 file. 
"""
function generate_callback(F::JLD2Output; kwargs...)
    return PeriodicCallback(F, F.interval; initial_affect = false, kwargs...)
end

"""
    CFLInfo <: AbstractCallback
    Container for CFL information callback.
"""
struct CFLInfo <: AbstractCallback
    model::AbstractModel
    interval::Number
end

"""
    (F::CFLInfo)(integrator)

"""
function (F::CFLInfo)(integrator)
    # Get model components
    model = F.model
    # Get state variables
    Y = getproperty(integrator.u, model.name)
    # Unpack horizontal and vertical velocity components
    uₕ = getproperty(Y.u.:2,1).:1
    uᵥ = getproperty(Y.u.:2,1).:2
    # Unpack axes  : Assumes Spectral-2D System
    x₁ = ClimaCore.Fields.coordinate_field(uₕ).x
    x₂ = ClimaCore.Fields.coordinate_field(uᵥ).y
    # Update integrator timestep
    integrator.dt *= 2 
    return nothing
end

"""
    generate_callback(F::CFLInfo; kwargs...)

    Creates a PeriodicCallback object that computes the 
    maximum CFL number in the domain. 
"""
function generate_callback(F::CFLInfo; kwargs...)
    return PeriodicCallback(F, F.interval; initial_affect = false, kwargs...)
end
