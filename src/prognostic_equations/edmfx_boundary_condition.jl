#####
##### EDMFX SGS boundary condition
#####

function sgs_scalar_first_interior_bc(
    ᶜz_int::FT,
    ᶜρ_int,
    ᶜscalar_int,
    sfc_buoyancy_flux,
    sfc_ρ_flux_scalar,
    ustar,
    obukhov_length,
    sfc_local_geometry,
) where {FT}
    sfc_buoyancy_flux > 0 || return ᶜscalar_int
    # a_total = edmf.surface_area
    # a_ = area_surface_bc(surface_conditions, edmf)
    scalar_var = get_first_interior_variance(
        sfc_ρ_flux_scalar / ᶜρ_int,
        ustar,
        ᶜz_int,
        obukhov_length,
        sfc_local_geometry,
    )
    # surface_scalar_coeff = percentile_bounds_mean_norm(
    #     1 - a_total + (i - 1) * a_,
    #     1 - a_total + i * a_,
    # )
    surface_scalar_coeff = FT(1.75)
    return ᶜscalar_int + surface_scalar_coeff * sqrt(scalar_var)
end

function get_first_interior_variance(
    flux,
    ustar,
    z::FT,
    oblength,
    local_geometry,
) where {FT}
    c_star = -flux / ustar
    if oblength < 0
        return 4 *
               Geometry._norm_sqr(c_star, local_geometry) *
               (1 - FT(8.3) * z / oblength)^(-FT(2 / 3))
    else
        return 4 * Geometry._norm_sqr(c_star, local_geometry)
    end
end
