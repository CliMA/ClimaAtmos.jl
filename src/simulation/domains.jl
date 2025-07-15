
export AbstractDomain, SphereDomain, ColumnDomain, BoxDomain, PlaneDomain

abstract type AbstractDomain end

struct SphereDomain{FT} <: AbstractDomain
    z_min::FT
    z_max::FT
    z_elem::Int
    z_stretch::Bool
    dz_bottom::FT
    radius::FT
    h_elem::Int
    nh_poly::Int
    bubble::Bool
    deep_atmosphere::Bool
end

struct ColumnDomain{FT} <: AbstractDomain
    z_min::FT
    z_max::FT
    z_elem::Int
    z_stretch::Bool
    dz_bottom::FT
end

struct BoxDomain{FT} <: AbstractDomain
    x_min::FT
    x_max::FT
    x_elem::Int
    y_min::FT
    y_max::FT
    y_elem::Int
    z_min::FT
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

struct PlaneDomain{FT} <: AbstractDomain
    x_min::FT
    x_max::FT
    x_elem::Int
    z_min::FT
    z_max::FT
    z_elem::Int
    nh_poly::Int
    z_stretch::Bool
    dz_bottom::FT
    bubble::Bool
    deep_atmosphere::Bool
    periodic_x::Bool
end 
