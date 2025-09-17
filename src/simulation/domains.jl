
# Default constants for domain parameters
const DEFAULT_Z_MAX = 30000.0
const DEFAULT_Z_ELEM = 10
const DEFAULT_DZ_BOTTOM = 500.0
const DEFAULT_Z_STRETCH = true
const DEFAULT_NH_POLY = 3
const DEFAULT_BUBBLE = false
const DEFAULT_DEEP_ATMOSPHERE = true
const DEFAULT_H_ELEM = 6
const DEFAULT_RADIUS = 6.371229e6
const DEFAULT_TOPO_DAMPING = 5.0
const DEFAULT_MESH_WARP_TYPE = "SLEVE"
const DEFAULT_SLEVE_ETA = 0.7
const DEFAULT_SLEVE_S = 10.0
const DEFAULT_TOPO_SMOOTHING = false
const DEFAULT_X_MIN = 0.0
const DEFAULT_X_MAX = 300000.0
const DEFAULT_X_ELEM = 6
const DEFAULT_Y_MIN = 0.0
const DEFAULT_Y_MAX = 300000.0
const DEFAULT_Y_ELEM = 6
const DEFAULT_PERIODIC_X = true
const DEFAULT_PERIODIC_Y = true

export AbstractDomain, SphereDomain, ColumnDomain, BoxDomain, PlaneDomain

abstract type AbstractDomain end

Base.@kwdef struct SphereDomain{FT} <: AbstractDomain
    z_max::FT = DEFAULT_Z_MAX
    z_elem::Int = DEFAULT_Z_ELEM
    z_stretch::Bool = DEFAULT_Z_STRETCH
    dz_bottom::FT = DEFAULT_DZ_BOTTOM
    radius::FT = DEFAULT_RADIUS
    h_elem::Int = DEFAULT_H_ELEM
    nh_poly::Int = DEFAULT_NH_POLY
    bubble::Bool = DEFAULT_BUBBLE
    deep_atmosphere::Bool = DEFAULT_DEEP_ATMOSPHERE
    topography_damping_factor::FT = DEFAULT_TOPO_DAMPING
    mesh_warp_type::String = DEFAULT_MESH_WARP_TYPE
    sleve_eta::FT = DEFAULT_SLEVE_ETA
    sleve_s::FT = DEFAULT_SLEVE_S
    topo_smoothing::Bool = DEFAULT_TOPO_SMOOTHING
end

Base.@kwdef struct ColumnDomain{FT} <: AbstractDomain
    z_max::FT = DEFAULT_Z_MAX
    z_elem::Int = DEFAULT_Z_ELEM
    z_stretch::Bool = DEFAULT_Z_STRETCH
    dz_bottom::FT = DEFAULT_DZ_BOTTOM
end

Base.@kwdef struct BoxDomain{FT} <: AbstractDomain
    x_min::FT = DEFAULT_X_MIN
    x_max::FT = DEFAULT_X_MAX
    x_elem::Int = DEFAULT_X_ELEM
    y_min::FT = DEFAULT_Y_MIN
    y_max::FT = DEFAULT_Y_MAX
    y_elem::Int = DEFAULT_Y_ELEM
    z_max::FT = DEFAULT_Z_MAX
    z_elem::Int = DEFAULT_Z_ELEM
    nh_poly::Int = DEFAULT_NH_POLY
    z_stretch::Bool = DEFAULT_Z_STRETCH
    dz_bottom::FT = DEFAULT_DZ_BOTTOM
    bubble::Bool = DEFAULT_BUBBLE
    deep_atmosphere::Bool = DEFAULT_DEEP_ATMOSPHERE
    periodic_x::Bool = DEFAULT_PERIODIC_X
    periodic_y::Bool = DEFAULT_PERIODIC_Y
end

Base.@kwdef struct PlaneDomain{FT} <: AbstractDomain
    x_min::FT = DEFAULT_X_MIN
    x_max::FT = DEFAULT_X_MAX
    x_elem::Int = DEFAULT_X_ELEM
    z_max::FT = DEFAULT_Z_MAX
    z_elem::Int = DEFAULT_Z_ELEM
    nh_poly::Int = DEFAULT_NH_POLY
    z_stretch::Bool = DEFAULT_Z_STRETCH
    dz_bottom::FT = DEFAULT_DZ_BOTTOM
    bubble::Bool = DEFAULT_BUBBLE
    deep_atmosphere::Bool = DEFAULT_DEEP_ATMOSPHERE
    periodic_x::Bool = DEFAULT_PERIODIC_X
end


function get_spaces(domain::PlaneDomain, params, comms_ctx)
    FT = eltype(params)
    quad = Quadratures.GLL{domain.nh_poly + 1}()
    horizontal_mesh = periodic_line_mesh(;
        x_max = domain.x_max,
        x_elem = domain.x_elem,
        periodic = domain.periodic_x,
    )
    h_space =
        make_horizontal_space(horizontal_mesh, quad, comms_ctx, domain.bubble)
    z_stretch = if domain.z_stretch
        Meshes.HyperbolicTangentStretching(domain.dz_bottom)
    else
        Meshes.Uniform()
    end
    center_space, face_space = make_hybrid_spaces(
        h_space,
        domain.z_max,
        domain.z_elem,
        z_stretch;
        deep = domain.deep_atmosphere,
    )
    return (; center_space, face_space)
end

function get_spaces(domain::BoxDomain, params, comms_ctx)
    FT = eltype(params)
    quad = Quadratures.GLL{domain.nh_poly + 1}()
    horizontal_mesh = periodic_rectangle_mesh(;
        x_max = domain.x_max,
        y_max = domain.y_max,
        x_elem = domain.x_elem,
        y_elem = domain.y_elem,
        periodic = (domain.periodic_x, domain.periodic_y),
    )
    h_space =
        make_horizontal_space(horizontal_mesh, quad, comms_ctx, domain.bubble)
    z_stretch = if domain.z_stretch
        Meshes.HyperbolicTangentStretching(domain.dz_bottom)
    else
        Meshes.Uniform()
    end
    center_space, face_space = make_hybrid_spaces(
        h_space,
        domain.z_max,
        domain.z_elem,
        z_stretch;
        deep = domain.deep_atmosphere,
    )
    return (; center_space, face_space)
end

function get_spaces(domain::ColumnDomain, params, comms_ctx)
    FT = eltype(params)
    @warn "perturb_initstate flag is ignored for single column configuration"
    Δx = FT(1) # Note: This value shouldn't matter, since we only have 1 column.
    quad = Quadratures.GL{1}()
    horizontal_mesh = periodic_rectangle_mesh(;
        x_max = Δx,
        y_max = Δx,
        x_elem = 1,
        y_elem = 1,
        periodic = (true, true),
    )
    bubble = false # bubble correction not compatible with single column configuration
    h_space = make_horizontal_space(horizontal_mesh, quad, comms_ctx, bubble)
    z_stretch = if domain.z_stretch
        Meshes.HyperbolicTangentStretching(domain.dz_bottom)
    else
        Meshes.Uniform()
    end
    center_space, face_space =
        make_hybrid_spaces(h_space, domain.z_max, domain.z_elem, z_stretch)
    return (; center_space, face_space)
end

function get_spaces(domain::SphereDomain, params, comms_ctx)
    FT = eltype(params)
    quad = Quadratures.GLL{domain.nh_poly + 1}()
    horizontal_mesh =
        cubed_sphere_mesh(; radius = domain.radius, h_elem = domain.h_elem)
    h_space =
        make_horizontal_space(horizontal_mesh, quad, comms_ctx, domain.bubble)
    z_stretch = if domain.z_stretch
        Meshes.HyperbolicTangentStretching(domain.dz_bottom)
    else
        Meshes.Uniform()
    end
    center_space, face_space = make_hybrid_spaces(
        h_space,
        domain.z_max,
        domain.z_elem,
        z_stretch;
        deep = domain.deep_atmosphere,
        topography_damping_factor = domain.topography_damping_factor,
        mesh_warp_type = domain.mesh_warp_type,
        sleve_eta = domain.sleve_eta,
        sleve_s = domain.sleve_s,
        topo_smoothing = domain.topo_smoothing,
    )
    return (; center_space, face_space)
end
