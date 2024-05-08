#####
##### EDMFX SGS boundary condition
#####

import Distributions

function sgs_scalar_first_interior_bc(
    ᶜz_int::FT,
    ᶜρ_int::FT,
    ᶜaʲ_int::FT,
    ᶜscalar_int::FT,
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
    ustar::FT,
    z::FT,
    oblength::FT,
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

function percentile_bounds_mean_norm(
    low_percentile::FT,
    high_percentile::FT,
) where {FT <: Real}
    gauss_int(x) = -exp(-x * x / 2) / sqrt(2 * pi)
    xp_low = Distributions.quantile(Distributions.Normal(), low_percentile)
    xp_high = Distributions.quantile(Distributions.Normal(), high_percentile)
    return (gauss_int(xp_high) - gauss_int(xp_low)) / max(
        Distributions.cdf(Distributions.Normal(), xp_high) -
        Distributions.cdf(Distributions.Normal(), xp_low),
        eps(FT),
    )
end
