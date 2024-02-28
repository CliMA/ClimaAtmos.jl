#####
##### EDMFX SGS boundary condition
#####

function sgs_scalar_first_interior_bc(
    ᶜz_int::FT,
    ᶜρ_int,
    ᶜaʲ_int,
    ᶜscalar_int,
    sfc_buoyancy_flux,
    sfc_ρ_flux_scalar,
    ustar,
    obukhov_length,
    sfc_local_geometry,
) where {FT}
    sfc_buoyancy_flux > 0 || return ᶜscalar_int
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
    # TODO: This assumes that there is only one updraft
    surface_scalar_coeff = percentile_bounds_mean_norm(1 - ᶜaʲ_int, FT(1))
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

function approximate_inverf(x::FT) where {FT <: Real}
    # From Wikipedia
    a = FT(0.147)
    term1 = (2 / (π * a) + log(1 - x^2) / 2)
    term2 = log(1 - x^2) / a
    term3 = sqrt(term1^2 - term2)

    return sign(x) * sqrt(term3 - term1)
end

function guass_quantile(p::FT) where {FT <: Real}
    return sqrt(2) * approximate_inverf(2p - 1)
end

function percentile_bounds_mean_norm(low_percentile, high_percentile)
    gauss_int(x) = -exp(-x * x / 2) / sqrt(2 * pi)
    xp_high = guass_quantile(high_percentile)
    xp_low = guass_quantile(low_percentile)
    return (gauss_int(xp_high) - gauss_int(xp_low)) /
           max(high_percentile - low_percentile, eps(typeof(high_percentile)))
end
