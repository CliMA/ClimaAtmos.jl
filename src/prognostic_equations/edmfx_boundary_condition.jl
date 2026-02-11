#####
##### EDMFX SGS boundary condition
#####

"""
    sgs_scalar_first_interior_bc(
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

Calculates a boundary condition for a subgrid-scale (SGS) scalar within an
EDMFX updraft `j` at the first interior cell center (`ᶜz_int`).

The method sets the updraft scalar value as the sum of the grid-mean scalar at
that level (`ᶜscalar_int`) and an SGS fluctuation term. This fluctuation is
proportional to the square root of the estimated SGS scalar variance (σ), i.e.,
`SGS_scalar = Mean_scalar + C * σ`.

The SGS variance is computed using Monin-Obukhov Similarity Theory via
`get_first_interior_variance`. The coefficient `C` (here, `surface_scalar_coeff`)
is determined by `percentile_bounds_mean_norm`, assuming the updraft samples
from the upper tail (from percentile `1 - ᶜaʲ_int` to 1) of a Gaussian distribution
of SGS fluctuations.

This boundary condition is applied only when the surface buoyancy flux
is positive (unstable conditions), indicating surface-driven updrafts. When the surface buoyancy flux is non-positive, the updraft scalar value is set to the grid-mean scalar.

Arguments:
- `ᶜz_int`: Height of the first interior cell center [m].
- `ᶜρ_int`: Grid-mean air density at `ᶜz_int` [kg/m³].
- `ᶜaʲ_int`: Area fraction of the specific updraft `j` at `ᶜz_int` (dimensionless).
- `ᶜscalar_int`: Grid-mean value of the scalar at `ᶜz_int`.
- `sfc_buoyancy_flux`: Surface buoyancy flux (e.g., w'b'_sfc) [m²/s³ or K⋅m/s].
                       Positive for unstable conditions.
- `sfc_ρ_flux_scalar`: Density-weighted surface flux of the scalar (e.g., ρ⋅w'c'_sfc)
                       [e.g., (kg/m³)⋅K⋅m/s or (kg/m³)⋅(kg/kg)⋅m/s].
- `ustar`: Friction velocity [m/s].
- `obukhov_length`: Obukhov length [m].
- `sfc_local_geometry`: `ClimaCore.Geometry.LocalGeometry` at the surface, passed to
                        variance calculation.

Returns:
- The prescribed scalar value for the SGS updraft at the first interior level.
  Returns `ᶜscalar_int` if `sfc_buoyancy_flux <= 0`.
"""
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
    # Only apply adjustment if surface buoyancy flux is positive (unstable conditions)
    sfc_buoyancy_flux > 0 || return ᶜscalar_int

    kinematic_sfc_flux_scalar = sfc_ρ_flux_scalar / ᶜρ_int # Convert to kinematic flux [K m/s]

    scalar_var = get_first_interior_variance(
        kinematic_sfc_flux_scalar,
        ustar,
        ᶜz_int,
        obukhov_length,
        sfc_local_geometry,
    )
    # surface_scalar_coeff = percentile_bounds_mean_norm(
    #     1 - a_total + (i - 1) * a_,
    #     1 - a_total + i * a_,
    # )
    # TODO: This assumes that there is only one updraft, or that ᶜaʲ_int
    #       is the specific area fraction for the updraft being considered when
    #       sampling from the tail of the combined subgrid + grid-mean distribution.
    #       The percentile range [1 - ᶜaʲ_int, 1] samples the top ᶜaʲ_int fraction.
    surface_scalar_coeff = percentile_bounds_mean_norm(1 - ᶜaʲ_int, FT(1))
    return ᶜscalar_int + surface_scalar_coeff * sqrt(scalar_var)
end

"""
    get_first_interior_variance(
        kinematic_scalar_flux,
        ustar::FT,
        z,
        obukhov_length,
        local_geometry, 
    ) where {FT}

Calculates the variance of a scalar quantity (σ²) at height `z` within the
surface layer using Monin-Obukhov Similarity Theory (MOST).

The calculation depends on stability:
- For unstable conditions (Obukhov length `L < 0`):
  σ² = C₁ * c²∗ * (1 - C₂ * z / L)^(-2/3)
- For stable/neutral conditions (Obukhov length `L >= 0`):
  σ² = C₁ * c²∗
where `c∗ = -kinematic_scalar_flux / ustar` is the scalar flux scale.
The constants C₁ and C₂ are empirical (here, C₁=4, C₂=8.3).

Arguments:
- `kinematic_scalar_flux`: Kinematic surface flux of the scalar (e.g., w'c'_sfc) [K⋅m/s or (kg/kg)⋅m/s].
- `ustar`: Friction velocity [m/s].
- `z`: Height above the surface [m].
- `obukhov_length`: Obukhov length [m].
- `local_geometry`: `ClimaCore.Geometry.LocalGeometry` object, used for `_norm_sqr` 

Returns:
- The estimated variance of the scalar quantity [K² or (kg/kg)²].
"""
function get_first_interior_variance(
    kinematic_scalar_flux,
    ustar::FT,
    z,
    obukhov_length,
    local_geometry,
) where {FT}
    c_star = -kinematic_scalar_flux / ustar
    # TODO: Do we need geometry here? Or is c_star always scalar? Otherwise, replace c_star_sq by c_star * c_star
    c_star_sq = Geometry._norm_sqr(c_star, local_geometry)
    if obukhov_length < 0 # Unstable conditions
        # Matches empirical forms, e.g., Wyngaard et al. (1971), Garratt (1994)
        return 4 * c_star_sq * (1 - FT(8.3) * z / obukhov_length)^(-FT(2 / 3))
    else  # Stable or neutral conditions
        return 4 * c_star_sq
    end
end

"""
    approximate_inverf(x::FT) where {FT}

Approximates the inverse error function, erf⁻¹(x).

The approximation formula involves logarithmic and square root operations and is
based on a Pade approximation (Sergei Winitzki)
(https://drive.google.com/file/d/0B2Mt7luZYBrwZlctV3A3eF82VGM/view?resourcekey=0-UQpPhwZgzP0sF4LHBDlLtgi).
It is valid for `x` in the interval `(-1, 1)`.

Arguments:
- `x`: The value (scalar or array element) for which to compute erf⁻¹(x).
       Must be between -1 and 1, exclusive.

Returns:
- An approximation of erf⁻¹(x).

Note: Numerical precision issues or invalid results may occur if `x` is too
close to -1 or 1, due to `log(1 - x^2)`. The conditions for the terms under
square roots (`term1^2 - term2 >= 0` and `term3 - term1 >= 0`) must also hold
for the approximation to be valid.
"""
function approximate_inverf(x::FT) where {FT}
    # From Sergei Winitzki
    a = FT(0.147)
    term1 = (2 / (π * a) + log(1 - x^2) / 2)
    term2 = log(1 - x^2) / a
    term3 = sqrt(term1^2 - term2)

    return sign(x) * sqrt(term3 - term1)
end

"""
    gauss_quantile(p::FT) where {FT}

Computes the quantile (inverse of the cumulative distribution function, CDF)
for a standard normal distribution N(0,1) at probability `p`.

It uses the relationship Φ⁻¹(p) = √2 * erf⁻¹(2p - 1), where Φ⁻¹ is the
normal quantile function and erf⁻¹ is the inverse error function, approximated
by `approximate_inverf`.

Arguments:
- `p`: The probability (scalar or array element), ranging from 0 to 1.

Returns:
- The standard normal quantile corresponding to `p`.
"""
function gauss_quantile(p::FT) where {FT}
    return sqrt(FT(2)) * approximate_inverf(2p - 1)
end

"""
    percentile_bounds_mean_norm(low_percentile::FT, high_percentile::FT) where {FT}

Calculates the mean value of a standard normal variable X ~ N(0,1) that is
truncated to the interval `[xp_low, xp_high]`, where `xp_low` and `xp_high`
are the quantiles corresponding to `low_percentile` and `high_percentile`
respectively.

The formula used is: E[X | xp_low ≤ X ≤ xp_high] = (ϕ(xp_low) - ϕ(xp_high)) / (P_high - P_low),
where ϕ is the PDF of N(0,1), and P_high/P_low are the high/low percentiles.

This gives a coefficient representing the expected value of a fluctuation
drawn from a specific segment of a Gaussian distribution.

Arguments:
- `low_percentile`: The lower percentile bound (e.g., 0.8 for the 80th percentile).
- `high_percentile`: The upper percentile bound (e.g., 0.9 for the 90th percentile).

Returns:
- The mean of the standard normal distribution truncated between the quantiles
  of `low_percentile` and `high_percentile`.
"""
function percentile_bounds_mean_norm(
    low_percentile,
    high_percentile::FT,
) where {FT}
    std_normal_pdf(x) = -exp(-x * x / 2) / sqrt(2 * pi)
    xp_high = gauss_quantile(high_percentile)
    xp_low = gauss_quantile(low_percentile)

    return (std_normal_pdf(xp_high) - std_normal_pdf(xp_low)) /
           max(high_percentile - low_percentile, eps(FT))
end
