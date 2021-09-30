CCF = ClimaCore.Fields
CCS = ClimaCore.Spaces

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
    interval::Real
    cfl_target::Real
    update::Bool
end

"""
    get_nodal_distance(space::Space)
    
Return a tuple of local spacing for a ClimaCore Space object.
Always returns a 3 component tuple, with `Inf` values for non-existent
dimensions (in the case of 1, 2 dimensional spaces). Δx₁, Δx₂ correspond
to horizontal spacing and Δx₃ corresponds to vertical spacing.
    
# Move to ClimaCore
"""
function get_nodal_distance(space::CCS.AbstractSpace)
    return nothing
end
function get_nodal_distance(space::CCS.ExtrudedFiniteDifferenceSpace)
    Δh_local = space.horizontal_space.local_geometry.WJ
    Δv_local = diff(space.vertical_mesh.faces)
    # TODO : Currently horizontal directions have npolynomial_x = npolynomial_y
    return (Δx₁ = Δh_local, Δx₂= Δh_local, Δx₃ = Δv_local)
end
function get_nodal_distance(space::CCS.SpectralElementSpace1D)
    Δh_local = space.local_geometry.WJ
    return (Δx₁ = Δh_local, Δx₂ = Inf, Δx₃ = Inf)
end
function get_nodal_distance(space::CCS.SpectralElementSpace2D)
    Δh_local = space.local_geometry.WJ
    return (Δx₁ = Δh_local, Δx₂ = Δh_local, Δx₃ = Inf)
end
function get_nodal_distance(space::CCS.FiniteDifferenceSpace)
    Δv_local = diff(space.vertical_mesh.faces)
    return (Δx₁ = Inf, Δx₂ = Inf, Δx₃ = Δv_local)
end
function get_local_courant(u, Δx, Δt, space::CCS.SpectralElementSpace2D)
    return @. abs(u) / Δx * Δt
end

"""
    (F::CFLAdaptive)(integrator)
"""
function (F::CFLAdaptive)(integrator)
    # Get model components
    model = F.model
    # Get state variables
    # Unpack horizontal and vertical velocity components: Generalise across models
    velocities = get_velocities(integrator.u, model)
    space = CCF.axes(velocities[1])
    if space isa CCS.Spaces.SpectralElementSpace2D
        Δx, Δy, Δz = get_nodal_distance(space)
        cfl_local_x = get_local_courant(CCF.field_values(velocities[1]), Δx, integrator.dt, space)
        cfl_local_y = get_local_courant(CCF.field_values(velocities[2]), Δy, integrator.dt, space)
        cfl_domain_x = maximum(cfl_local_x)
        cfl_domain_y = maximum(cfl_local_y)
        cfl_domain_max = max(cfl_domain_x, cfl_domain_y)
        cfl_current = cfl_domain_max
        rtol = abs((cfl_domain_max - F.cfl_target) / F.cfl_target)
    else
        @error "$space unsupported for adaptive timestepping."
    end
    if F.update == true
        dt_suggested = F.cfl_target / cfl_current * integrator.dt
        isinf(dt_suggested) ? nothing : integrator.dtcache = dt_suggested 
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
    return DiffEqCallbacks.PeriodicCallback(F, F.interval; kwargs...)
end
