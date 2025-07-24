
export AbstractDomain, SphereDomain, ColumnDomain, BoxDomain, PlaneDomain

abstract type AbstractDomain end

Base.@kwdef struct SphereDomain{FT} <: AbstractDomain
    z_max::FT
    z_elem::Int
    z_stretch::Bool
    dz_bottom::FT
    radius::FT
    h_elem::Int
    nh_poly::Int
    bubble::Bool
    deep_atmosphere::Bool
    topography_damping_factor::FT
    mesh_warp_type::String
    sleve_eta::FT
    sleve_s::FT
    topo_smoothing::Bool
end

Base.@kwdef struct ColumnDomain{FT} <: AbstractDomain
    z_max::FT
    z_elem::Int
    z_stretch::Bool
    dz_bottom::FT
end

Base.@kwdef struct BoxDomain{FT} <: AbstractDomain
    x_min::FT
    x_max::FT
    x_elem::Int
    y_min::FT
    y_max::FT
    y_elem::Int
    z_max::FT
    z_elem::Int
    nh_poly::Int
    z_stretch::Bool
    dz_bottom::FT
    bubble::Bool
    deep_atmosphere::Bool
    periodic_x::Bool
    periodic_y::Bool
end

Base.@kwdef struct PlaneDomain{FT} <: AbstractDomain
    x_min::FT
    x_max::FT
    x_elem::Int
    z_max::FT
    z_elem::Int
    nh_poly::Int
    z_stretch::Bool
    dz_bottom::FT
    bubble::Bool
    deep_atmosphere::Bool
    periodic_x::Bool
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
    horizontal_mesh = cubed_sphere_mesh(; radius = domain.radius, h_elem = domain.h_elem)
    h_space = make_horizontal_space(horizontal_mesh, quad, comms_ctx, domain.bubble)
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
