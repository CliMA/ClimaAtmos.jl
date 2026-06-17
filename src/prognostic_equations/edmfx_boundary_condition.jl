#####
##### EDMFX SGS boundary condition
#####


"""
    set_edmfx_surface_conditions!(Y, p)

Populate the per-updraft level-1 caches `sfc_mse_buoyantʲs`,
`sfc_q_tot_buoyantʲs`, and `sfc_mass_flux_sourceʲs` for each EDMF
updraft. Three separate scalar fields rather than a single NamedTuple-
typed field, because broadcasting a NamedTuple into a DataLayout at a
single level hits a GPU-incompatible `convert` path inside
`knl_copyto!`. No-op unless `turbconv_model isa PrognosticEDMFX`.
"""
function set_edmfx_surface_conditions!(Y, p)
    p.atmos.turbconv_model isa PrognosticEDMFX || return nothing
    (; params) = p
    turbconv_params = CAP.turbconv_params(params)
    n = n_mass_flux_subdomains(p.atmos.turbconv_model)
    (;
        sfc_mass_flux_sourceʲs,
        sfc_mse_buoyantʲs,
        sfc_q_tot_buoyantʲs,
        ᶜρʲs,
        ᶜK,
        ᶜh_tot,
    ) = p.precomputed
    (; ustar, obukhov_length, buoyancy_flux, ρ_flux_h_tot, ρ_flux_q_tot) =
        p.precomputed.sfc_conditions

    ᶜq_tot = @. lazy(specific(Y.c.ρq_tot, Y.c.ρ))
    ᶜmse⁰ = ᶜspecific_env_mse(Y, p)
    ᶜq_tot⁰ = ᶜspecific_env_value(@name(q_tot), Y, p)

    lg_val = Fields.field_values(
        Fields.local_geometry_field(Fields.level(Y.f, Fields.half)),
    )
    bf_val = Fields.field_values(buoyancy_flux)
    ustar_val = Fields.field_values(ustar)
    obukhov_val = Fields.field_values(obukhov_length)
    z_int_val =
        Fields.field_values(Fields.level(Fields.coordinate_field(Y.c).z, 1))
    z_sfc_val = Fields.field_values(
        Fields.level(Fields.coordinate_field(Y.f).z, Fields.half),
    )
    ρ_int_val = Fields.field_values(Fields.level(Y.c.ρ, 1))
    h_tot_int_val = Fields.field_values(Fields.level(ᶜh_tot, 1))
    K_int_val = Fields.field_values(Fields.level(ᶜK, 1))
    q_tot_int_val = Fields.field_values(Fields.level(ᶜq_tot, 1))
    mse_env_val = Fields.field_values(Fields.level(ᶜmse⁰, 1))
    q_tot_env_val = Fields.field_values(Fields.level(ᶜq_tot⁰, 1))
    ᶜdz = Fields.Δz_field(axes(Y.c))
    dz_int_val = Fields.field_values(Fields.level(ᶜdz, 1))

    # Project C3 surface flux vectors onto the surface normal.
    ρ_flux_h_tot_face_val = Fields.field_values(ρ_flux_h_tot)
    ρ_flux_q_tot_face_val = Fields.field_values(ρ_flux_q_tot)
    ρ_flux_h_tot_val =
        Fields.field_values(Fields.level(p.scratch.ᶜtemp_scalar, 1))
    ρ_flux_q_tot_val =
        Fields.field_values(Fields.level(p.scratch.ᶜtemp_scalar_2, 1))
    @. ρ_flux_h_tot_val =
        projected_vector_data(C3, ρ_flux_h_tot_face_val, lg_val)
    @. ρ_flux_q_tot_val =
        projected_vector_data(C3, ρ_flux_q_tot_face_val, lg_val)

    for j in 1:n
        ρʲ_val = Fields.field_values(Fields.level(ᶜρʲs.:($j), 1))
        ρaʲ_val = Fields.field_values(Fields.level(Y.c.sgsʲs.:($j).ρa, 1))
        mse_buoyant_val = Fields.field_values(
            Fields.level(sfc_mse_buoyantʲs.:($j), 1),
        )
        q_tot_buoyant_val = Fields.field_values(
            Fields.level(sfc_q_tot_buoyantʲs.:($j), 1),
        )
        mass_flux_source_val = Fields.field_values(
            Fields.level(sfc_mass_flux_sourceʲs.:($j), 1),
        )

        # Buoyant-air surface values first, so the mass-flux cap can
        # consume them. Each broadcast writes a plain scalar DataF.
        @. mse_buoyant_val = edmfx_sfc_buoyant(
            bf_val,
            ρ_int_val,
            ustar_val,
            obukhov_val,
            lg_val,
            z_int_val - z_sfc_val,
            h_tot_int_val - K_int_val,
            ρ_flux_h_tot_val,
            turbconv_params,
        )
        @. q_tot_buoyant_val = edmfx_sfc_buoyant(
            bf_val,
            ρ_int_val,
            ustar_val,
            obukhov_val,
            lg_val,
            z_int_val - z_sfc_val,
            q_tot_int_val,
            ρ_flux_q_tot_val,
            turbconv_params,
        )
        @. mass_flux_source_val = edmfx_sfc_mass_flux_source(
            bf_val,
            ρʲ_val,
            ρaʲ_val,
            ustar_val,
            dz_int_val,
            mse_buoyant_val,
            q_tot_buoyant_val,
            mse_env_val,
            q_tot_env_val,
            ρ_flux_h_tot_val,
            ρ_flux_q_tot_val,
            turbconv_params,
        )
    end
    return nothing
end

"""
    edmfx_sfc_buoyant(
        sfc_buoyancy_flux, ρ_int, ustar, obukhov_length, sfc_local_geometry,
        z_int, scalar_grid, sfc_ρ_flux_scalar, turbconv_params,
    )

High-tail buoyant-air surface value of a scalar (mse or q_tot) for an
EDMF updraft, evaluated from [`sgs_scalar_first_interior_bc`](@ref) with
the percentile fraction set to
`a_s = surface_mass_flux_coefficient(...)` — the same `a_s` that sets
the surface mass flux magnitude.
"""
@inline function edmfx_sfc_buoyant(
    sfc_buoyancy_flux,
    ρ_int,
    ustar,
    obukhov_length,
    sfc_local_geometry,
    z_int,
    scalar_grid,
    sfc_ρ_flux_scalar,
    turbconv_params,
)
    FT = typeof(ρ_int)
    z_i = edmf_convective_zi(FT)
    a_s_max = turbconv_params.surface_area
    a_s = surface_mass_flux_coefficient(sfc_buoyancy_flux, z_i, ustar, a_s_max)
    return sgs_scalar_first_interior_bc(
        z_int,
        ρ_int,
        a_s,
        scalar_grid,
        sfc_buoyancy_flux,
        sfc_ρ_flux_scalar,
        ustar,
        obukhov_length,
        sfc_local_geometry,
    )
end

"""
    edmfx_sfc_mass_flux_source(
        sfc_buoyancy_flux, ρʲ_int, ρaʲ_int, ustar, dz_int,
        mse_buoyant, q_tot_buoyant, mse_env, q_tot_env,
        sfc_ρ_flux_h_tot, sfc_ρ_flux_q_tot, turbconv_params,
    )

Volumetric mass source rate `F_sfc / dz_int` [kg/m³/s] at the first
cell for one EDMF updraft, equivalent to `div(F·ẑ)` at level 1.
`F_sfc` is the capped surface mass flux with the
`entr_upper_area_limiter_factor(a)` baked in:

    F_pre   = surface_mass_flux(...) · entr_upper_area_limiter_factor(a),
    F_max_χ = α · sfc_ρ_flux_χ / max(ϵ, χ_buoyant − χ_env)
                                              for χ ∈ {mse, q_tot},
    F_sfc   = max(0, min(F_pre, F_max_mse, F_max_q_tot)).

`α < 1` guarantees the env retains at least `(1−α)` of every surface
scalar flux. The denominator floor `ϵ = ϵ_numerics(FT)` keeps `F_max`
finite when the eddy contrast `χ_buoyant − χ_env` vanishes or goes
negative — in that case `F_max` is huge and effectively non-binding
through the subsequent `min`.

The buoyant values are passed in (precomputed by
[`edmfx_sfc_buoyant`](@ref)) so they can be cached separately and
consumed elsewhere by the mse/q_tot tendency.
"""
@inline function edmfx_sfc_mass_flux_source(
    sfc_buoyancy_flux,
    ρʲ_int,
    ρaʲ_int,
    ustar,
    dz_int,
    mse_buoyant,
    q_tot_buoyant,
    mse_env,
    q_tot_env,
    sfc_ρ_flux_h_tot,
    sfc_ρ_flux_q_tot,
    turbconv_params,
)
    FT = typeof(ρʲ_int)
    # TODO: promote `α_sfc_flux_cap` to a calibrated TOML parameter.
    α_sfc_flux_cap = FT(0.5)
    z_i = edmf_convective_zi(FT)
    a_s_max = turbconv_params.surface_area

    F_pre =
        surface_mass_flux(sfc_buoyancy_flux, ρʲ_int, z_i, ustar, a_s_max) *
        entr_upper_area_limiter_factor(
            draft_area(ρaʲ_int, ρʲ_int),
            turbconv_params,
        )
    F_max_mse =
        α_sfc_flux_cap * sfc_ρ_flux_h_tot /
        max(ϵ_numerics(FT), mse_buoyant - mse_env)
    F_max_q =
        α_sfc_flux_cap * sfc_ρ_flux_q_tot /
        max(ϵ_numerics(FT), q_tot_buoyant - q_tot_env)
    return max(zero(FT), min(F_pre, F_max_mse, F_max_q)) / dz_int
end

"""
    edmfx_boundary_condition_tendency!(Yₜ, Y, p, t, turbconv_model)

Apply the surface mass-flux boundary condition to the EDMFX updraft
scalar prognostic variables (`mse`, `q_tot`) in the first model cell.

The cached `mass_flux_source` (see [`edmfx_sfc_mass_flux_source`](@ref))
is the volumetric mass source rate `F_sfc / dz` at the first cell,
equivalent to `div(F·ẑ)` evaluated at level 1. That mass carries the
high-tail (buoyant) values `mse_buoyant`, `q_tot_buoyant` from
[`sgs_scalar_first_interior_bc`](@ref). For the specific (intensive)
updraft variables this gives a flux-form tendency at the first cell:

    d(val)/dt += mass_flux_source · (val_buoyant − val) / max(ρa, ρ·a_min),

where `mass_flux_source` already includes the env-positivity cap and
the `entr_upper_area_limiter_factor(a)` that smoothly shuts the source
off as the plume area approaches `a_max`. The `max(ρa, ρ·a_min)` floor
keeps the divisor finite when the updraft is small.

The corresponding `ρa` source is injected in the implicit ρa solve
(`solve_sgs_ρa_implicit_stage_analytic!`).

Note: at the first cell the updraft scalar tendencies receive *two*
contributions — this surface mass-flux BC and the standard lateral
entrainment from [`edmfx_entr_detr_tendency!`](@ref). These represent
distinct physical processes (surface mass injection from the buoyant
sub-cell tail vs. lateral entrainment from the environment at level 1)
and are intentionally both retained. The two relaxation targets differ
(grid-mean + SGS fluctuation vs. environment value), and the manual
Jacobian carries diagonal entries for both.
"""
edmfx_boundary_condition_tendency!(Yₜ, Y, p, t, turbconv_model) = nothing
function edmfx_boundary_condition_tendency!(
    Yₜ,
    Y,
    p,
    t,
    turbconv_model::PrognosticEDMFX,
)
    (; params) = p
    (;
        ᶜρʲs,
        sfc_mass_flux_sourceʲs,
        sfc_mse_buoyantʲs,
        sfc_q_tot_buoyantʲs,
    ) = p.precomputed
    FT = eltype(params)
    n = n_mass_flux_subdomains(p.atmos.turbconv_model)
    a_min = CAP.min_area(CAP.turbconv_params(params))

    for j in 1:n
        ρ_val = Fields.field_values(Fields.level(ᶜρʲs.:($j), 1))
        ρa_val = Fields.field_values(Fields.level(Y.c.sgsʲs.:($j).ρa, 1))
        mse_val = Fields.field_values(Fields.level(Y.c.sgsʲs.:($j).mse, 1))
        q_tot_val = Fields.field_values(Fields.level(Y.c.sgsʲs.:($j).q_tot, 1))
        mseₜ_val = Fields.field_values(Fields.level(Yₜ.c.sgsʲs.:($j).mse, 1))
        q_totₜ_val = Fields.field_values(Fields.level(Yₜ.c.sgsʲs.:($j).q_tot, 1))
        mass_flux_source_val = Fields.field_values(
            Fields.level(sfc_mass_flux_sourceʲs.:($j), 1),
        )
        mse_buoyant_val = Fields.field_values(
            Fields.level(sfc_mse_buoyantʲs.:($j), 1),
        )
        q_tot_buoyant_val = Fields.field_values(
            Fields.level(sfc_q_tot_buoyantʲs.:($j), 1),
        )

        # `mass_flux_source · (val_buoyant − val) / max(ρa, ρ·a_min)`
        # The `max(ρa, ρ·a_min)` floor keeps the divisor finite while
        # the updraft is just starting to grow.
        @. mseₜ_val +=
            mass_flux_source_val * (mse_buoyant_val - mse_val) /
            max(ρa_val, ρ_val * FT(a_min))
        @. q_totₜ_val +=
            mass_flux_source_val * (q_tot_buoyant_val - q_tot_val) /
            max(ρa_val, ρ_val * FT(a_min))
    end
    return nothing
end

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
is positive (unstable conditions), indicating surface-driven updrafts.
When the surface buoyancy flux is non-positive, the updraft scalar
value is set to the grid-mean scalar.

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
    c_star = -kinematic_scalar_flux / max(ustar, eps(FT))
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
