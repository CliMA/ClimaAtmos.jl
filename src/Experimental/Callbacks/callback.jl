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
Dispatched over AbstractSpace subtypes.
"""
function get_nodal_distance(space::ClimaCore.Spaces.AbstractSpace)
    return nothing
end
function get_nodal_distance(
    space::ClimaCore.Spaces.ExtrudedFiniteDifferenceSpace,
)
    Δh_local = space.horizontal_space.local_geometry.WJ
    vertspace = Spaces.CenterFiniteDifferenceSpace(space.vertical_topology.mesh)
    Δv_local = ClimaCore.Spaces.coordinates_data(vertspace).z[1]
    return (Δh_local, Δv_local)
end
function get_nodal_distance(
    space::Union{
        ClimaCore.Spaces.SpectralElementSpace1D,
        ClimaCore.Spaces.SpectralElementSpace2D,
    },
)
    # Assumes uniform polynomial order in horizontal and vertical direction
    Δh_local = space.local_geometry.WJ
    return (Δh_local)
end
function get_nodal_distance(space::ClimaCore.Spaces.FiniteDifferenceSpace)
    Δv_local = ClimaCore.Spaces.coordinates_data(space).z[1]
    return (Δv_local)
end

abstract type CFLDirection end
struct HorizontalCFL <: CFLDirection end
struct VerticalCFL <: CFLDirection end

function get_local_courant(uh, Δx, Δt, ::HorizontalCFL)
    return abs.(ClimaCore.Fields.field_values(uh.components.data.:1)) ./ Δx .*
           eltype(Δx)(Δt)
end
function get_local_courant(w, Δz, Δt, ::VerticalCFL)
    return maximum(abs.(w.components.data.:1)) ./ Δz .* eltype(Δz)(Δt)
end

"""
    (F::CFLAdaptive)(integrator)
For the given model and function space, gets the nodal distances
(local=> horizontal), (minimum=> vertical) and model velocities
from which the CFL number is calculated at the user specified
time interval. If the update flag is true, sets new timestep size.
"""
function domain_max_cfl(
    space::ClimaCore.Spaces.AbstractSpace,
    integrator,
    model,
)
    return nothing
end
function domain_max_cfl(
    space::ClimaCore.Spaces.ExtrudedFiniteDifferenceSpace,
    integrator,
    model,
)
    Δx, Δz = get_nodal_distance(space)
    Δt = integrator.dt
    uh, w = Models.get_velocities(integrator.u, model)
    cfl_local_x = get_local_courant(uh, Δx, Δt, HorizontalCFL())
    cfl_local_z = get_local_courant(w, Δz, Δt, VerticalCFL())
    cfl_domain_x = maximum(cfl_local_x)
    cfl_domain_z = maximum(cfl_local_z)
    return max(cfl_domain_x, cfl_domain_z)
end
function domain_max_cfl(
    space::ClimaCore.Spaces.FiniteDifferenceSpace,
    integrator,
    model,
)
    w = Models.get_velocities(integrator.u, model)
    Δz = get_nodal_distance(space)
    cfl_local_z = get_local_courant(w, Δz, integrator.dt, VerticalCFL())
    return max(cfl_local_z)
end
function (F::CFLAdaptive)(integrator)
    model = F.model
    # Assumes mass conservation equation is included in model.
    space = ClimaCore.Fields.axes(integrator.u.base.ρ)
    cfl_current = domain_max_cfl(space, integrator, model)
    if F.update == true
        dt_updated = F.cfl_target / cfl_current * integrator.dt
        isinf(dt_updated) ? nothing : integrator.dtcache = dt_updated
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
