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
    CFLAdaptive <: AbstractCallback
    Container for CFL information callback.
"""
mutable struct CFLAdaptive <: AbstractCallback
    model::AbstractModel
    cfl_current::Real
    cfl_target::Real
    update::Bool
end

"""
    get_nodal_distance(space::Space)
# Move to ClimaCore
"""
function get_nodal_distance(space::ClimaCore.Spaces.AbstractSpace)
    return nothing
end
function get_nodal_distance(space::ClimaCore.Spaces.ExtrudedFiniteDifferenceSpace)
    Δh_local = space.horizontal_space.local_geometry.WJ
    Δv_local = diff(space.vertical_mesh.faces)
    # TODO : Currently horizontal directions have npolynomial_x = npolynomial_y
    return (Δx₁ = Δh_local, Δx₂= Δh_local, Δx₃ = Δv_local)
end
function get_nodal_distance(space::ClimaCore.Spaces.SpectralElementSpace1D)
    Δh_local = space.local_geometry.WJ
    return (Δx₁ = Δh_local, Δx₂ = Inf, Δx₃ = Inf)
end
function get_nodal_distance(space::ClimaCore.Spaces.SpectralElementSpace2D)
    Δh_local = space.local_geometry.WJ
    return (Δx₁ = Δh_local, Δx₂ = Δh_local, Δx₃ = Inf)
end

function (F::CFLAdaptive)(u, t, integrator)
    # Get model components
    model = F.model
    # Get state variables
    Y = getproperty(integrator.u, model.name)
    # Unpack horizontal and vertical velocity components
    uₕ = getproperty(Y.u.:2, 1).:1
    uᵥ = getproperty(Y.u.:2, 1).:2
    # Get underlying space
    space = ClimaCore.Fields.axes(uₕ)
    # Get local nodal distances
    Δx, Δy, Δz = get_nodal_distance(space)
    # Compute local Courant number
    cfl_local = abs.(ClimaCore.Fields.field_values(uₕ)) ./ Δx .* integrator.dt
    cfl_domain_max = maximum(cfl_local)
    F.cfl_current = cfl_domain_max
    return cfl_domain_max > F.cfl_target
end

"""
    (F::CFLAdaptive)(integrator)
"""
function (F::CFLAdaptive)(integrator)
    if F.update == true
        dt_suggested = F.cfl_target / F.cfl_current * integrator.dt
        isinf(dt_suggested) ? nothing : integrator.dtcache = dt_suggested #dtcache if adaptive option is false // 
        @info ("New Δt = $(integrator.dt)")
    else
        nothing
    end
    return nothing
end

"""
    generate_callback(F::CFLAdaptive; kwargs...)

    Creates a PeriodicCallback object that computes the 
    maximum CFL number in the domain. 
"""
function generate_callback(F::CFLAdaptive; kwargs...)
    return DiffEqCallbacks.DiscreteCallback(F, F; kwargs...)
end
