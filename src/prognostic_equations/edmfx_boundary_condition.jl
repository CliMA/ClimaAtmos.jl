#####
##### EDMFX SGS boundary condition
#####


"""
    set_edmfx_surface_conditions!(Y, p)

Populate the per-updraft surface boundary-condition payload at model level 1
for the PROPHET scheme (`EDMFX` in code).

For each updraft `j`, writes the level-1 caches

  - `p.precomputed.sfc_mse_buoyantʲs`, `sfc_q_tot_buoyantʲs`: the high-tail
    (buoyant-air) surface values of `mse` and `q_tot`
    (`edmfx_sfc_buoyant`) [J/kg] and [kg/kg];
  - `p.precomputed.sfc_mass_flux_sourceʲs`: the capped volumetric mass source
    rate (`edmfx_sfc_mass_flux_source`) [kg/m³/s].

The buoyant values are computed first so the mass-flux cap can consume them.
Surface `C3` flux vectors are projected onto the surface normal through
`p.scratch.ᶜtemp_scalar` and `ᶜtemp_scalar_2` at level 1.

These are three separate scalar fields rather than one `NamedTuple`-valued
field because broadcasting a `NamedTuple` into a `DataLayout` at a single level
hits a GPU-incompatible `convert` path inside `knl_copyto!`.

No-op unless `p.atmos.turbconv_model isa PrognosticEDMFX`. Mutates
`p.precomputed` and `p.scratch`; returns `nothing`. Called from
`set_prognostic_edmf_precomputed_quantities_explicit_closures!`; the payload is
consumed by `edmfx_boundary_condition_tendency!` and by the implicit
`ρa` solve.
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

Return the high-tail (buoyant-air) surface value of a scalar (`mse` or
`q_tot`) for a PROPHET updraft.

Evaluates `sgs_scalar_first_interior_bc` at the first interior cell
center with the sampled percentile fraction set to
`a_s = surface_mass_flux_coefficient(...)` — the same `a_s` that sets the
surface mass-flux magnitude in `surface_mass_flux`.

# Arguments

  - `sfc_buoyancy_flux`: Surface buoyancy flux [m²/s³]; the buoyant value equals
    the grid mean when it is non-positive.
  - `ρ_int`: Grid-mean density at the first interior cell center [kg/m³].
  - `ustar`: Friction velocity [m/s].
  - `obukhov_length`: Obukhov length [m].
  - `sfc_local_geometry`: `ClimaCore.Geometry.LocalGeometry` at the surface.
  - `z_int`: Height of the first interior cell center above the surface [m].
  - `scalar_grid`: Grid-mean value of the scalar at that level, `mse` [J/kg] or
    `q_tot` [kg/kg].
  - `sfc_ρ_flux_scalar`: Surface density-weighted flux of the same scalar,
    projected onto the surface normal.
  - `turbconv_params`: Turbulence-convection parameters (supply `convective_zi`,
    `max_surface_area`, and `sfc_mass_flux_ustar_coeff`).

# Returns

The buoyant-air surface value of the scalar, in the units of `scalar_grid`.
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
    z_i = CAP.convective_zi(turbconv_params)
    a_s_max = CAP.max_surface_area(turbconv_params)
    c_u = CAP.sfc_mass_flux_ustar_coeff(turbconv_params)
    a_s = surface_mass_flux_coefficient(
        sfc_buoyancy_flux,
        z_i,
        ustar,
        a_s_max,
        c_u,
    )
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

Return the volumetric mass source rate `F_sfc / dz_int` [kg/m³/s] at the first
cell for one PROPHET updraft, equivalent to `div(F·ẑ)` at level 1.

`F_sfc` is the capped surface mass flux (`surface_mass_flux`) with the
`upper_area_limiter_factor` baked in:

    F_pre   = surface_mass_flux(...) · upper_area_limiter_factor(a),
    F_max_χ = α · sfc_ρ_flux_χ / max(ϵ, χ_buoyant − χ_env)
                                              for χ ∈ {mse, q_tot},
    F_sfc   = max(0, min(F_pre, F_max_mse, F_max_q_tot)).

`α = sfc_mass_flux_cap_fraction < 1` guarantees the environment retains at
least `(1−α)` of every surface scalar flux. The denominator floor
`ϵ = ϵ_numerics(FT)` keeps `F_max` finite when the eddy contrast
`χ_buoyant − χ_env` vanishes or goes negative — in that case `F_max` is huge
and effectively non-binding through the subsequent `min`.

The buoyant values are passed in (precomputed by
`edmfx_sfc_buoyant`) so they can be cached separately and consumed
elsewhere by the `mse`/`q_tot` tendency.

# Arguments

  - `sfc_buoyancy_flux`: Surface buoyancy flux [m²/s³].
  - `ρʲ_int`, `ρaʲ_int`: Updraft density [kg/m³] and area-weighted density
    [kg/m³] at the first interior cell center.
  - `ustar`: Friction velocity [m/s].
  - `dz_int`: Depth of the first model cell [m].
  - `mse_buoyant`, `mse_env`: Buoyant-air and environment moist static energy
    [J/kg].
  - `q_tot_buoyant`, `q_tot_env`: Buoyant-air and environment total specific
    humidity [kg/kg].
  - `sfc_ρ_flux_h_tot`, `sfc_ρ_flux_q_tot`: Surface density-weighted total
    enthalpy [W/m²] and total water [kg/m²/s] fluxes, projected onto the
    surface normal.
  - `turbconv_params`: Turbulence-convection parameters.

# Returns

The non-negative volumetric mass source rate [kg/m³/s].
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
    α_sfc_flux_cap = CAP.sfc_mass_flux_cap_fraction(turbconv_params)
    z_i = CAP.convective_zi(turbconv_params)
    a_s_max = CAP.max_surface_area(turbconv_params)
    c_u = CAP.sfc_mass_flux_ustar_coeff(turbconv_params)

    F_pre =
        surface_mass_flux(
            sfc_buoyancy_flux,
            ρʲ_int,
            z_i,
            ustar,
            a_s_max,
            c_u,
        ) * upper_area_limiter_factor(
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

Apply the surface mass-flux boundary condition to the PROPHET (`EDMFX` in
code) updraft scalar prognostic variables (`mse`, `q_tot`) in the first model
cell.

The generic method is a no-op; the `turbconv_model::PrognosticEDMFX` method
increments `Yₜ.c.sgsʲs.:(j).mse` and `.q_tot` at level 1 for every updraft and
returns `nothing`.

The cached `mass_flux_source` (see `edmfx_sfc_mass_flux_source`)
is the volumetric mass source rate `F_sfc / dz` at the first cell,
equivalent to `div(F·ẑ)` evaluated at level 1. That mass carries the
high-tail (buoyant) values `mse_buoyant`, `q_tot_buoyant` from
`sgs_scalar_first_interior_bc`. For the specific (intensive)
updraft variables this gives a flux-form tendency at the first cell:

    d(val)/dt += mass_flux_source · (val_buoyant − val) / max(ρa, ρ·a_min),

where `mass_flux_source` already includes the env-positivity cap and
the `upper_area_limiter_factor(a)` that smoothly shuts the source
off as the plume area approaches `a_max`. The `max(ρa, ρ·a_min)` floor
keeps the divisor finite when the updraft is small.

The corresponding `ρa` source is injected in the implicit ρa solve
(`solve_sgs_ρa_implicit_stage_analytic!`).

# Notes

At the first cell the updraft scalar tendencies receive *two* contributions —
this surface mass-flux BC and the standard lateral entrainment from
`edmfx_entr_detr_tendency!`. These represent
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

Return the boundary value of a subgrid-scale (SGS) scalar for an updraft
sampling the top `ᶜaʲ_int` fraction of the distribution at the first interior
cell center.

The value is the grid-mean scalar plus an SGS fluctuation proportional to the
standard deviation of the SGS scalar distribution,

    scalar_sgs = ᶜscalar_int + C √σ²,

with `σ²` from `get_first_interior_variance` (Monin-Obukhov similarity
theory) and `C = percentile_bounds_mean_norm(1 - ᶜaʲ_int, 1)` the mean of a
standard normal truncated to its upper `ᶜaʲ_int` tail.

The adjustment is applied only when the surface buoyancy flux is positive
(unstable, surface-driven updrafts); otherwise the grid-mean value is
returned unchanged.

# Arguments

  - `ᶜz_int`: Height of the first interior cell center above the surface [m].
  - `ᶜρ_int`: Grid-mean air density at `ᶜz_int` [kg/m³].
  - `ᶜaʲ_int`: Fraction of the distribution sampled by the updraft [-]; the
    surface mass-flux area coefficient `a_s` in `edmfx_sfc_buoyant`.
  - `ᶜscalar_int`: Grid-mean value of the scalar at `ᶜz_int`.
  - `sfc_buoyancy_flux`: Surface buoyancy flux `⟨w'b'⟩_s` [m²/s³]. Positive for
    unstable conditions.
  - `sfc_ρ_flux_scalar`: Density-weighted surface flux of the scalar
    `⟨ρ w'c'⟩_s`, in the units of `ᶜscalar_int` times [kg/m²/s].
  - `ustar`: Friction velocity [m/s].
  - `obukhov_length`: Obukhov length [m].
  - `sfc_local_geometry`: `ClimaCore.Geometry.LocalGeometry` at the surface,
    passed to the variance calculation.

# Returns

The SGS updraft value of the scalar at the first interior level, in the units
of `ᶜscalar_int`; `ᶜscalar_int` itself when `sfc_buoyancy_flux ≤ 0`.
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

Return the surface-layer variance `σ²` of a scalar at height `z` from
Monin-Obukhov similarity theory.

With the scalar flux scale `c∗ = -kinematic_scalar_flux / max(ustar, eps)` and
empirical constants `C₁ = 4`, `C₂ = 8.3`:

  - unstable conditions (Obukhov length `L < 0`):
    `σ² = C₁ c∗² (1 - C₂ z / L)^(-2/3)`;
  - stable or neutral conditions (`L ≥ 0`): `σ² = C₁ c∗²`.

Empirical forms follow, e.g., Wyngaard et al. (1971) and Garratt (1994).

# Arguments

  - `kinematic_scalar_flux`: Kinematic surface flux of the scalar `⟨w'c'⟩_s`,
    in the units of the scalar times [m/s].
  - `ustar`: Friction velocity [m/s].
  - `z`: Height above the surface [m].
  - `obukhov_length`: Obukhov length [m].
  - `local_geometry`: `ClimaCore.Geometry.LocalGeometry`, used by `_norm_sqr`.

# Returns

The scalar variance, in the units of the scalar squared. Called from
`sgs_scalar_first_interior_bc`.
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

Approximate the inverse error function `erf⁻¹(x)` for `x ∈ (-1, 1)`.

Uses Winitzki's closed-form approximation with shape parameter `a = 0.147`.
Called from `gauss_quantile`.

# Arguments

  - `x`: Argument of `erf⁻¹`, strictly between -1 and 1 [-].

# Returns

An approximation of `erf⁻¹(x)` [-].

# Notes

Accuracy degrades and the result may become invalid as `|x| → 1`, because of
`log(1 - x²)`; the terms under the square roots (`term1² - term2 ≥ 0` and
`term3 - term1 ≥ 0`) must remain non-negative.    # From Sergei Winitzki
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

Compute the standard-normal quantile `Φ⁻¹(p) = √2 erf⁻¹(2p - 1)`, with `erf⁻¹`
approximated by `approximate_inverf`.

# Arguments

  - `p`: Probability, in `(0, 1)` [-].

# Returns

The standard normal quantile corresponding to `p` [-]. Called from
`percentile_bounds_mean_norm`.
"""
function gauss_quantile(p::FT) where {FT}
    return sqrt(FT(2)) * approximate_inverf(2p - 1)
end

"""
    percentile_bounds_mean_norm(low_percentile, high_percentile::FT) where {FT}

Compute the mean of a standard normal variable `X ~ N(0,1)` truncated to the
quantile interval of `[low_percentile, high_percentile]`:

    E[X | x_low ≤ X ≤ x_high] = (ϕ(x_low) - ϕ(x_high)) / (P_high - P_low),

where `ϕ` is the standard normal PDF and `x_low`, `x_high` are the quantiles
(`gauss_quantile`) of the two percentiles. The denominator is floored
at `eps(FT)`.

The result is the coefficient multiplying the SGS standard deviation for a
subdomain that samples that segment of the distribution.

# Arguments

  - `low_percentile`: Lower percentile bound, e.g. `0.8` for the 80th
    percentile [-].
  - `high_percentile`: Upper percentile bound [-].

# Returns

The truncated-normal mean [-]. Called from
`sgs_scalar_first_interior_bc`.
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
