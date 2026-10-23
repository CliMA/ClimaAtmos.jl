#####
##### EDMF entrainment and detrainment parameterizations
#####

import Thermodynamics.Parameters as TDP

# Entrainment models

# Helper function for Pi-Group based models
"""
    calculate_pi_groups(
        elev_above_sfc, ref_H, ᶜaʲ, ᶜwʲ, ᶜRHʲ, ᶜbuoyʲ,
        ᶜw⁰, ᶜRH⁰, ᶜbuoy⁰, ᶜtke
    )

Compute the non-dimensional Π-groups of the PROPHET (`EDMFX` in code)
entrainment closure.

# Arguments

  - `elev_above_sfc`: Cell-center height above the surface, `ᶜz - z_sfc` [m].
  - `ref_H`: Reference pressure scale height [m].
  - `ᶜaʲ`: Updraft area fraction [-].
  - `ᶜwʲ`: Updraft physical vertical velocity [m/s].
  - `ᶜRHʲ`: Updraft relative humidity [-].
  - `ᶜbuoyʲ`: Updraft buoyancy [m/s²].
  - `ᶜw⁰`: Environment physical vertical velocity [m/s].
  - `ᶜRH⁰`: Environment relative humidity [-].
  - `ᶜbuoy⁰`: Environment buoyancy [m/s²].
  - `ᶜtke`: Turbulent kinetic energy [m²/s²].

# Returns

The tuple `(Π₁, Π₂, Π₃, Π₄, Π₅)`, all dimensionless:

  - `Π₁ = z Δbuoy / Δw²`, buoyancy contrast against the velocity contrast.
  - `Π₂ = tke / Δw²`, environment turbulence against the velocity contrast.
  - `Π₃ = √a`, updraft area.
  - `Π₄ = ΔRH`, relative-humidity contrast.
  - `Π₅ = z / ref_H`, elevation normalized by the scale height.

`Π₁` and `Π₂` are divided by the empirical factors 100 and 2 to make them
`O(1)` and are then clipped to `[-1, 1]`; `Δw²` carries an `eps` floor.
"""
function calculate_pi_groups(
    elev_above_sfc,
    ref_H,
    ᶜaʲ,
    ᶜwʲ,
    ᶜRHʲ,
    ᶜbuoyʲ,
    ᶜw⁰,
    ᶜRH⁰,
    ᶜbuoy⁰,
    ᶜtke,
)
    FT = eltype(elev_above_sfc)
    vel_diff_sq = (ᶜwʲ - ᶜw⁰)^2 + eps(FT)
    Π₁_raw = elev_above_sfc * (ᶜbuoyʲ - ᶜbuoy⁰) / vel_diff_sq
    Π₂_raw = max(ᶜtke, 0) / vel_diff_sq
    Π₃ = sqrt(max(ᶜaʲ, 0))
    Π₄ = ᶜRHʲ - ᶜRH⁰
    Π₅ = elev_above_sfc / max(ref_H, eps(FT))

    Π₁ = min(max(Π₁_raw / FT(100), FT(-1)), FT(1))
    Π₂ = min(max(Π₂_raw / FT(2), FT(-1)), FT(1))
    return (Π₁, Π₂, Π₃, Π₄, Π₅)
end

"""
    upper_area_limiter_factor(a, turbconv_params)

Return a multiplicative limiter for the entrainment rate based on the
current subdraft area `a`.

The factor smoothly damps entrainment as the area approaches the upper
bound `a_max` (preventing the area from being driven above `a_max`):

    (max(0, 1 - a / a_max))^max_area_limiter_power

The upper bound `a_max` is obtained from `turbconv_params` through
`CAP.max_area`, and `max_area_limiter_power` from `CAP.max_area_limiter_power`.

This is the entrainment counterpart of `lower_area_limiter_factor`,
which damps detrainment near the lower bound `a_min`. Together they keep
`a ∈ [a_min, a_max]` without requiring a comparison between `entr` and `detr`.
"""
@inline function upper_area_limiter_factor(a::FT, turbconv_params) where {FT}
    a_max = CAP.max_area(turbconv_params)
    max_area_limiter_power = CAP.max_area_limiter_power(turbconv_params)
    a_safe = max(eps(FT), a)
    return (max(0, (1 - a_safe / a_max)))^max_area_limiter_power
end

"""
    lower_area_limiter_factor(a, turbconv_params)

Return a multiplicative limiter for the detrainment rate based on the current
subdraft area `a`.

The factor smoothly damps detrainment as the area approaches the lower bound
`a_min` (preventing the area from being driven below `a_min`):

    (max(0, 1 - a_min / a))^min_area_limiter_power

The lower bound `a_min` is obtained from `turbconv_params` through
`CAP.min_area`, and `min_area_limiter_power` from `CAP.min_area_limiter_power`.

This is the detrainment counterpart of `upper_area_limiter_factor`,
which damps entrainment near the upper bound `a_max`. Together they keep
`a ∈ [a_min, a_max]` without requiring a comparison between `entr` and `detr`.
"""
@inline function lower_area_limiter_factor(a::FT, turbconv_params) where {FT}
    a_min = CAP.min_area(turbconv_params)
    min_area_limiter_power = CAP.min_area_limiter_power(turbconv_params)
    a_safe = max(eps(FT), a)
    return (max(0, (1 - a_min / a_safe)))^min_area_limiter_power
end

"""
    entr_buoy_inv_time_scale(Δwʲ, Δbuoyʲ, entr_detr_buoy_inv_tau_max)

Clipped inverse buoyancy time-scale [1/s] used by the buoyancy-driven
entrainment branch:

    τ⁻¹_buoy = min(entr_detr_buoy_inv_tau_max,
                   max(Δbuoyʲ, 0) / max(eps, |Δwʲ|))

Only positive buoyancy contributes (a negatively buoyant plume detrains rather
than entrains), and the rate is capped at `entr_detr_buoy_inv_tau_max` so a
vanishing `Δwʲ` doesn't produce an unbounded rate.
"""
@inline function entr_buoy_inv_time_scale(Δwʲ, Δbuoyʲ, entr_detr_buoy_inv_tau_max)
    FT = typeof(Δwʲ)
    return min(
        entr_detr_buoy_inv_tau_max,
        max(Δbuoyʲ, FT(0)) / max(eps(FT), abs(Δwʲ)),
    )
end

"""
    detr_buoy_inv_time_scale(Δwʲ, Δbuoyʲ, entr_detr_buoy_inv_tau_max)

Clipped inverse buoyancy time-scale [1/s] used by the buoyancy-driven
detrainment branch:

    τ⁻¹_buoy = min(entr_detr_buoy_inv_tau_max,
                   |min(Δbuoyʲ, 0)| / max(eps, |Δwʲ|))

Only negative buoyancy contributes (positive buoyancy doesn't detrain), and
the rate is capped at `entr_detr_buoy_inv_tau_max` so a vanishing `Δwʲ` doesn't
produce an unbounded rate.

Extracted as a helper so it can be reused by `detrainment_rate` (the
explicit detrainment closure) and the implicit ρa solve (where the same
buoyancy-detrainment piece appears in the `(ε − δ)` denominator).
"""
@inline function detr_buoy_inv_time_scale(
    Δwʲ,
    Δbuoyʲ,
    entr_detr_buoy_inv_tau_max,
    w_min = zero(Δwʲ),
)
    FT = typeof(Δwʲ)
    return min(
        entr_detr_buoy_inv_tau_max,
        abs(min(Δbuoyʲ, FT(0))) / max(eps(FT), w_min, abs(Δwʲ)),
    )
end

"""
    effective_plume_height(elev_above_sfc, turbconv_params)

Effective plume height `z_eff` above the surface, floored at `entr_z_min` and
optionally capped at `entr_z_max` when `entr_z_max > 0`.
"""
@inline function effective_plume_height(elev_above_sfc, turbconv_params)
    FT = typeof(elev_above_sfc)
    z_min = CAP.entr_z_min(turbconv_params)
    z_max = CAP.entr_z_max(turbconv_params)
    z_eff = max(elev_above_sfc, z_min)
    return ifelse(z_max > FT(0), min(z_eff, z_max), z_eff)
end

"""
    buoyancy_and_shell_detrainment_rate(
        thermo_params, turbconv_params, ᶜz, z_sfc, ᶜp,
        ᶜT⁰, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰,
        ᶜTʲ, ᶜq_tot_nonnegʲ, ᶜq_liqʲ, ᶜq_iceʲ,
        ᶜwʲ, ᶜbuoyʲ, ᶜbuoy_inv_time_scale, ᶜlower_limiter_factor,
    )

Buoyancy and evaporative-shell detrainment rate `[1/s]`. Combines negative-buoyancy
detrainment amplified by environmental subsaturation `1 - RH⁰` (`detr_rh_coeff`)
with lateral shell detrainment (`detr_shell_coeff`) driven by the draft–environment
relative-humidity contrast `max(0, RHʲ - RH⁰)` and environmental subsaturation,
damped by the local nondimensional buoyancy-to-kinetic-energy ratio `Π₁_local`.
"""
@inline function buoyancy_and_shell_detrainment_rate(
    thermo_params,
    turbconv_params,
    ᶜz,
    z_sfc,
    ᶜp,
    ᶜT⁰,
    ᶜq_tot_nonneg⁰,
    ᶜq_liq⁰,
    ᶜq_ice⁰,
    ᶜTʲ,
    ᶜq_tot_nonnegʲ,
    ᶜq_liqʲ,
    ᶜq_iceʲ,
    ᶜwʲ,
    ᶜbuoyʲ,
    ᶜbuoy_inv_time_scale,
    ᶜlower_limiter_factor,
)
    FT = eltype(thermo_params)
    detr_buoy_coeff = CAP.detr_buoy_coeff(turbconv_params)
    detr_rh_coeff = CAP.detr_rh_coeff(turbconv_params)
    detr_shell_coeff = CAP.detr_shell_coeff(turbconv_params)

    RH⁰ = TD.relative_humidity(
        thermo_params,
        ᶜT⁰,
        ᶜp,
        ᶜq_tot_nonneg⁰,
        ᶜq_liq⁰,
        ᶜq_ice⁰,
    )
    subsat⁰ = clamp(FT(1) - RH⁰, FT(0), FT(1))
    buoy_detr =
        detr_buoy_coeff * (FT(1) + detr_rh_coeff * subsat⁰) * ᶜbuoy_inv_time_scale

    elev_above_sfc = ᶜz - z_sfc
    if detr_shell_coeff <= FT(0) || elev_above_sfc <= eps(FT)
        return ᶜlower_limiter_factor * buoy_detr
    end

    RHʲ = TD.relative_humidity(
        thermo_params,
        ᶜTʲ,
        ᶜp,
        ᶜq_tot_nonnegʲ,
        ᶜq_liqʲ,
        ᶜq_iceʲ,
    )
    ΔRH_pos = clamp(RHʲ - RH⁰, FT(0), FT(1))
    z_eff = effective_plume_height(elev_above_sfc, turbconv_params)
    w_min = max(eps(FT), CAP.entr_buoy_w_min(turbconv_params))
    w_pos = max(ᶜwʲ, FT(0))
    Π₁_local = ᶜbuoyʲ * z_eff / (w_pos^2 + w_min^2)

    Φ_shell = ΔRH_pos * subsat⁰ / (FT(1) + max(FT(0), Π₁_local))
    shell_detr = detr_shell_coeff * (w_pos / z_eff) * Φ_shell
    return ᶜlower_limiter_factor * (buoy_detr + shell_detr)
end

function compute_buoyancy_and_shell_detrainment_rate!(
    ᶜbuoy_shell_detr,
    Y,
    p,
    ::Val{j},
    ᶜw,
    ᶜbuoy_inv_time_scale,
    ᶜlower_limiter_factor,
) where {j}
    FT = Spaces.undertype(axes(Y.c))
    turbconv_params = CAP.turbconv_params(p.params)
    detr_buoy_coeff = CAP.detr_buoy_coeff(turbconv_params)
    detr_rh_coeff = CAP.detr_rh_coeff(turbconv_params)
    detr_shell_coeff = CAP.detr_shell_coeff(turbconv_params)
    if (detr_rh_coeff > FT(0) || detr_shell_coeff > FT(0)) &&
       !(p.atmos.microphysics_model isa DryModel)
        thermo_params = CAP.thermodynamics_params(p.params)
        (; ᶜgradᵥ_ᶠΦ) = p.core
        (;
            ᶜp,
            ᶜρ_diffʲs,
            ᶜT⁰,
            ᶜq_tot_nonneg⁰,
            ᶜq_liq⁰,
            ᶜq_ice⁰,
            ᶜTʲs,
            ᶜq_tot_nonnegʲs,
            ᶜq_liqʲs,
            ᶜq_iceʲs,
        ) = p.precomputed
        ᶜlg = Fields.local_geometry_field(Y.c)
        ᶜz = Fields.coordinate_field(Y.c).z
        z_sfc = Fields.level(Fields.coordinate_field(Y.f).z, half)
        @. ᶜbuoy_shell_detr = buoyancy_and_shell_detrainment_rate(
            thermo_params,
            turbconv_params,
            ᶜz,
            z_sfc,
            ᶜp,
            ᶜT⁰,
            ᶜq_tot_nonneg⁰,
            ᶜq_liq⁰,
            ᶜq_ice⁰,
            ᶜTʲs.:($$j),
            ᶜq_tot_nonnegʲs.:($$j),
            ᶜq_liqʲs.:($$j),
            ᶜq_iceʲs.:($$j),
            ᶜw,
            vertical_buoyancy_acceleration(ᶜρ_diffʲs.:($$j), ᶜgradᵥ_ᶠΦ, ᶜlg),
            ᶜbuoy_inv_time_scale,
            ᶜlower_limiter_factor,
        )
    else
        @. ᶜbuoy_shell_detr =
            ᶜlower_limiter_factor * detr_buoy_coeff * ᶜbuoy_inv_time_scale
    end
    return ᶜbuoy_shell_detr
end

"""
    compute_entrainment(ᶜentr_vel_scale, ᶜentr_nonvel_rate,
                        ᶜarea_bounding_entr_detr, ᶜwʲ)

Total entrainment rate [1/s], assembled from the three precomputed pieces:

    entr = entr_vel_scale * |wʲ|           # velocity-proportional  [1/m] * [m/s]
         + entr_nonvel_rate                # buoyancy-driven + constant [1/s]
         + max(0, area_bounding_entr_detr)  # area-bounding          [1/s]

`entr_vel_scale` [1/m] is precomputed by `entrainment_velocity_scale`,
`entr_nonvel_rate` [1/s] by `entrainment_nonvel_rate`, and
`area_bounding_entr_detr` [1/s] is the signed rate produced by
`area_bounding_entr_detr` (positive ⇒ this entrainment branch,
negative ⇒ the detrainment branch in `compute_detrainment`).
`ᶜwʲ` is the physical updraft vertical velocity [m/s].

TODO: scale the velocity-proportional term with the relative velocity
`|wʲ - w⁰|`, as the formulation has it. Setting `w⁰ = 0` bounds the entrainment
of a draft with `wʲ ≈ 0`, whose area would otherwise grow without limit; the
analytic `ρa` stage solve holds the area fixed through the Newton iteration,
which closes that path. The implicit `u₃` solve already assumes the
relative-velocity form, so the two solves use rates that differ by `1/a⁰`.
"""
compute_entrainment(
    ᶜentr_vel_scale,
    ᶜentr_nonvel_rate,
    ᶜarea_bounding_entr_detr,
    ᶜwʲ,
) =
    ᶜentr_vel_scale * abs(ᶜwʲ) +
    ᶜentr_nonvel_rate +
    max(zero(ᶜarea_bounding_entr_detr), ᶜarea_bounding_entr_detr)

"""
    entrainment_velocity_scale(
        thermo_params, turbconv_params, ᶜz, z_sfc, ᶜp, ᶜρ,
        ᶜaʲ, ᶜwʲ, ᶜRHʲ, ᶜbuoyʲ, ᶜw⁰, ᶜRH⁰, ᶜbuoy⁰, ᶜtke,
        model_option::AbstractEntrainmentModel,
    )

Velocity-scaling prefactor [1/m] for the model-specific entrainment rate.
The total entrainment rate [1/s] is assembled by `compute_entrainment` as

    entr = entr_vel_scale * abs(wʲ) + entr_nonvel_rate
         + max(0, area_bounding_entr_detr)

where the second term comes from `entrainment_nonvel_rate` and the
third from `area_bounding_entr_detr` (its positive branch).
`model_option` selects the entrainment model:

  - `PiGroupsEntrainment`: `Π`-group closure
    (`calculate_pi_groups`), `entr_vel_scale = limiter · max(0, Σᵢ cᵢ|Πᵢ| + c₆) / (ᶜz - z_sfc)` with the coefficients
    `entr_param_vec`.
  - `InvZEntrainment`: `entr_vel_scale = limiter · (entr_inv_length + entr_coeff / (ᶜz - z_sfc))`.

Both non-trivial models multiply by `upper_area_limiter_factor` and
return zero at or below the surface, where `1/(ᶜz - z_sfc)` is singular.

# Arguments

All state arguments are cell-centered.

  - `thermo_params`: Thermodynamic parameters.
  - `turbconv_params`: Turbulence-convection parameters.
  - `ᶜz`: Height [m].
  - `z_sfc`: Surface elevation [m].
  - `ᶜp`: Pressure [Pa].
  - `ᶜρ`: Air density [kg/m³].
  - `ᶜaʲ`: Updraft area fraction [-].
  - `ᶜwʲ`: Updraft physical vertical velocity [m/s].
  - `ᶜRHʲ`: Updraft relative humidity [-].
  - `ᶜbuoyʲ`: Updraft buoyancy [m/s²].
  - `ᶜw⁰`: Environment physical vertical velocity [m/s].
  - `ᶜRH⁰`: Environment relative humidity [-].
  - `ᶜbuoy⁰`: Environment buoyancy [m/s²].
  - `ᶜtke`: Turbulent kinetic energy [m²/s²].
  - `model_option`: Entrainment model dispatch tag.

# Returns

The non-negative velocity-scaling prefactor [1/m].
"""
function entrainment_velocity_scale(
    thermo_params,
    turbconv_params,
    ᶜz,
    z_sfc,
    ᶜp,
    ᶜρ,
    ᶜaʲ,
    ᶜwʲ,
    ᶜRHʲ,
    ᶜbuoyʲ,
    ᶜw⁰,
    ᶜRH⁰,
    ᶜbuoy⁰,
    ᶜtke,
    ::PiGroupsEntrainment,
)
    FT = eltype(thermo_params)

    elev_above_sfc = ᶜz - z_sfc
    # If elevation above surface is not positive, terms like 1/elev_above_sfc
    # become singular. Model assumes operation above the surface.
    if elev_above_sfc <= eps(FT)
        return 0
    end

    g = TDP.grav(thermo_params)
    ref_H = ᶜp / (ᶜρ * g) # Pressure scale height
    z_eff = effective_plume_height(elev_above_sfc, turbconv_params)

    Π₁, Π₂, Π₃, Π₄, Π₅ = calculate_pi_groups(
        z_eff,
        ref_H,
        ᶜaʲ,
        ᶜwʲ,
        ᶜRHʲ,
        ᶜbuoyʲ,
        ᶜw⁰,
        ᶜRH⁰,
        ᶜbuoy⁰,
        ᶜtke,
    )

    entr_param_vec = CAP.entr_param_vec(turbconv_params)
    entr_inv_length = CAP.entr_inv_length(turbconv_params)
    form = CAP.entr_functional_form(turbconv_params)
    area_limiter_factor = upper_area_limiter_factor(ᶜaʲ, turbconv_params)

    pi_sum =
        entr_param_vec[1] * abs(Π₁) +
        entr_param_vec[2] * abs(Π₂) +
        entr_param_vec[3] * abs(Π₃) +
        entr_param_vec[4] * abs(Π₄) +
        entr_param_vec[5] * abs(Π₅) +
        entr_param_vec[6]

    if form < FT(0.5)
        entr_vel_scale = area_limiter_factor * (max(FT(0), pi_sum) / z_eff)
        return max(FT(0), entr_vel_scale)
    else
        w_min = max(eps(FT), CAP.entr_buoy_w_min(turbconv_params))
        Π₁_local = (ᶜbuoyʲ - ᶜbuoy⁰) * z_eff / ((ᶜwʲ - ᶜw⁰)^2 + w_min^2)
        π1 = tanh(Π₁_local)
        π2 = clamp(FT(1) - ᶜRH⁰, FT(0), FT(1))
        d2_core = ((π1 - FT(0.50)) / FT(0.30))^2 + ((π2 - FT(0.20)) / FT(0.25))^2
        d2_cong = ((π1 - FT(0.10)) / FT(0.22))^2 + ((π2 - FT(0.45)) / FT(0.25))^2
        d2_over = ((π1 - FT(-0.15)) / FT(0.25))^2 + ((π2 - FT(0.55)) / FT(0.25))^2
        gp2d_corr =
            FT(-0.30) * exp(FT(-0.5) * d2_core) +
            FT(-0.15) * exp(FT(-0.5) * d2_cong) +
            FT(0.40) * exp(FT(-0.5) * d2_over)
        c_tke = CAP.entr_tke_coeff(turbconv_params)
        tke_pos = max(ᶜtke, FT(0))
        tke_scale =
            c_tke * sqrt(tke_pos / ((ᶜwʲ - ᶜw⁰)^2 + FT(2) * tke_pos + w_min^2)) /
            z_eff
        base_scale = entr_inv_length + max(FT(0), pi_sum) / z_eff + tke_scale
        return max(
            FT(0),
            area_limiter_factor * base_scale * max(FT(0.35), FT(1) + gp2d_corr),
        )
    end
end

function entrainment_velocity_scale(
    thermo_params,
    turbconv_params,
    ᶜz,
    z_sfc,
    ᶜp,
    ᶜρ,
    ᶜaʲ,
    ᶜwʲ,
    ᶜRHʲ,
    ᶜbuoyʲ,
    ᶜw⁰,
    ᶜRH⁰,
    ᶜbuoy⁰,
    ᶜtke,
    ::InvZEntrainment,
)
    FT = eltype(thermo_params)
    entr_inv_length = CAP.entr_inv_length(turbconv_params)
    entr_coeff = CAP.entr_coeff(turbconv_params)

    elev_above_sfc = ᶜz - z_sfc
    # If elevation above surface is not positive, terms like 1/elev_above_sfc
    # become singular. Model assumes operation above the surface.
    if elev_above_sfc <= eps(FT)
        return 0
    end

    area_limiter_factor = upper_area_limiter_factor(ᶜaʲ, turbconv_params)
    entr_vel_scale =
        area_limiter_factor * (entr_inv_length + entr_coeff / elev_above_sfc)
    return max(0, entr_vel_scale)
end

"""
    entrainment_nonvel_rate(turbconv_params, ᶜaʲ, ᶜwʲ, ᶜbuoyʲ, ᶜw⁰, ᶜbuoy⁰,
                            model_option::AbstractEntrainmentModel)

The entrainment rate [1/s] carried by the terms that are not proportional to
`|wʲ|`.
"""
function entrainment_nonvel_rate(
    turbconv_params,
    ᶜaʲ,
    ᶜwʲ,
    ᶜbuoyʲ,
    ᶜw⁰,
    ᶜbuoy⁰,
    ::AbstractEntrainmentModel,
)
    return zero(ᶜaʲ)
end

function entrainment_nonvel_rate(
    turbconv_params,
    ᶜaʲ,
    ᶜwʲ,
    ᶜbuoyʲ,
    ᶜw⁰,
    ᶜbuoy⁰,
    ::InvZEntrainment,
)
    entr_buoy_coeff = CAP.entr_buoy_coeff(turbconv_params)
    entr_detr_buoy_inv_tau_max = CAP.entr_detr_buoy_inv_tau_max(turbconv_params)
    entr_inv_tau = CAP.entr_inv_tau(turbconv_params)

    buoy_rate =
        entr_buoy_coeff * entr_buoy_inv_time_scale(
            ᶜwʲ - ᶜw⁰,
            ᶜbuoyʲ - ᶜbuoy⁰,
            entr_detr_buoy_inv_tau_max,
        )

    area_limiter_factor = upper_area_limiter_factor(ᶜaʲ, turbconv_params)
    return area_limiter_factor * (buoy_rate + entr_inv_tau)
end

function entrainment_nonvel_rate(
    thermo_params,
    turbconv_params,
    ᶜz,
    z_sfc,
    ᶜp,
    ᶜρ,
    ᶜaʲ,
    ᶜwʲ,
    ᶜRHʲ,
    ᶜbuoyʲ,
    ᶜw⁰,
    ᶜRH⁰,
    ᶜbuoy⁰,
    ᶜtke,
    model_option::AbstractEntrainmentModel,
)
    return entrainment_nonvel_rate(
        turbconv_params,
        ᶜaʲ,
        ᶜwʲ,
        ᶜbuoyʲ,
        ᶜw⁰,
        ᶜbuoy⁰,
        model_option,
    )
end

function entrainment_nonvel_rate(
    thermo_params,
    turbconv_params,
    ᶜz,
    z_sfc,
    ᶜp,
    ᶜρ,
    ᶜaʲ,
    ᶜwʲ,
    ᶜRHʲ,
    ᶜbuoyʲ,
    ᶜw⁰,
    ᶜRH⁰,
    ᶜbuoy⁰,
    ᶜtke,
    ::PiGroupsEntrainment,
)
    return zero(ᶜaʲ)
end

"""
    area_bounding_entr_detr(turbconv_params, a)

Signed velocity-independent rate [1/s] that smoothly relaxes the updraft
area `a` back into `[a_min, a_max]`, applied unconditionally (no
dispatch on the entrainment/detrainment model). The sign convention is

    > 0  ⇒  acts as additional entrainment (drives the area up; a < a_min)
    < 0  ⇒  acts as additional detrainment (drives the area down; a > a_max)
    = 0  ⇒  inactive (a_min ≤ a ≤ a_max)

The result is

    +min_area_limiter_scale · (max(0, a_min − a)/a_min)^min_area_limiter_power
    −max_area_limiter_scale · (max(0, a − a_max)/(1 − a_max))^max_area_limiter_power.

The two ranges (`a < a_min` and `a > a_max`) are mutually exclusive, so
exactly one term is non-zero outside `[a_min, a_max]` and both are zero
inside it.

The positive part is consumed by `compute_entrainment` via
`max(0, area_bounding_entr_detr)`; the negative part is consumed by
`compute_detrainment` via `max(0, -area_bounding_entr_detr)`.
"""
@inline function area_bounding_entr_detr(turbconv_params, a)
    FT = typeof(a)
    a_min = CAP.min_area(turbconv_params)
    a_max = CAP.max_area(turbconv_params)
    min_scale = CAP.min_area_limiter_scale(turbconv_params)
    min_power = CAP.min_area_limiter_power(turbconv_params)
    max_scale = CAP.max_area_limiter_scale(turbconv_params)
    max_power = CAP.max_area_limiter_power(turbconv_params)

    # Extra entrainment for a < a_min (0 at a_min, min_scale at a = 0).
    min_relax = min_scale * (max(FT(0), a_min - a) / a_min)^min_power
    # Extra detrainment for a > a_max (0 at a_max, max_scale at a = 1).
    max_relax = max_scale * (max(FT(0), a - a_max) / (1 - a_max))^max_power

    return min_relax - max_relax
end

"""
    compute_detrainment(turbconv_params, aʲ, ρaʲ, buoy_inv_time_scale,
                        massflux_vert_div, area_bounding_entr_detr, detr_model)

Total detrainment rate [1/s] as the sum of the model-specific rate from
`detrainment_rate` (which internally applies
`lower_area_limiter_factor` so that detrainment is damped as
`a → a_min`) with the negative part of the signed area-bounding rate:

    detr = detrainment_rate(...) + max(0, -area_bounding_entr_detr)

`area_bounding_entr_detr` is produced by `area_bounding_entr_detr`;
its positive branch feeds `compute_entrainment` while the
negative branch (the only one that contributes here) drives the area back
below `a_max`.

Arguments:

  - `turbconv_params`: Turbulence convection parameters.
  - `aʲ`: Updraft area fraction [-].
  - `ρaʲ`: Updraft density-area product [kg/m³].
  - `buoy_inv_time_scale`: Clipped inverse buoyancy time scale [1/s] from
    `detr_buoy_inv_time_scale`. The caller chooses where to evaluate
    it (centers, or faces with subsequent `ᶜinterp`) to control smoothness.
  - `massflux_vert_div`: Vertical divergence of the updraft mass flux [kg/(m³ s)].
  - `area_bounding_entr_detr`: Signed area-bounding rate [1/s] from
    `area_bounding_entr_detr`.
  - `detr_model`: Object specifying the detrainment model.

Returns the total detrainment rate [1/s].
"""
@inline function massflux_divergence_detrainment_coeff(
    turbconv_params,
    ᶜbuoy_inv_time_scale,
)
    FT = typeof(ᶜbuoy_inv_time_scale)
    c_div = CAP.detr_massflux_vertdiv_coeff(turbconv_params)
    c_div_buoy = CAP.detr_massflux_buoy_coeff(turbconv_params)
    inv_tau_max = CAP.entr_detr_buoy_inv_tau_max(turbconv_params)
    b_nondim = max(FT(0), ᶜbuoy_inv_time_scale) / max(inv_tau_max, eps(FT))
    return clamp(
        c_div + c_div_buoy * b_nondim / (b_nondim + FT(1)),
        FT(0),
        FT(1),
    )
end

function compute_detrainment(
    turbconv_params,
    aʲ,
    ρaʲ,
    buoy_inv_time_scale,
    massflux_vert_div,
    area_bounding_entr_detr,
    detr_model,
    buoy_and_shell_detr_rate = lower_area_limiter_factor(aʲ, turbconv_params) *
                               CAP.detr_buoy_coeff(turbconv_params) *
                               buoy_inv_time_scale,
)

    detr = detrainment_rate(
        turbconv_params,
        aʲ,
        ρaʲ,
        buoy_inv_time_scale,
        massflux_vert_div,
        detr_model,
        buoy_and_shell_detr_rate,
    )

    return detr + max(zero(area_bounding_entr_detr), -area_bounding_entr_detr)
end

"""
    detrainment_rate(turbconv_params, ᶜaʲ, ᶜρaʲ, ᶜbuoy_inv_time_scale,
                     ᶜmassflux_vert_div, detr_model::AbstractDetrainmentModel,
                     [ᶜbuoy_and_shell_detr_rate])
    detrainment_rate(turbconv_params, ᶜaʲ, ᶜρaʲ, ᶜbuoy_inv_time_scale,
                     ᶜmassflux_vert_div, detr_model::BuoyancyVelocityDetrainment,
                     [ᶜbuoy_and_shell_detr_rate])

Model-specific detrainment rate [1/s] for a given detrainment model.

The `AbstractDetrainmentModel` fallback returns zero. For
`BuoyancyVelocityDetrainment` the rate combines `ᶜbuoy_and_shell_detr_rate`
(which defaults to `lower_area_limiter_factor(aʲ) · c_buoy τ⁻¹_buoy` and may
include relative-humidity and shell detrainment from
`buoyancy_and_shell_detrainment_rate`) with mass-flux convergence detrainment
scaled by `massflux_divergence_detrainment_coeff`:

    detr = max(0, buoy_and_shell_detr_rate
                  − lower_area_limiter_factor(aʲ) · c_div_eff ·
                    min(massflux_vert_div, 0) / ρaʲ).

Only a converging (negative) mass-flux divergence detrains, and
`lower_area_limiter_factor` damps the rate as `a → a_min`.

# Arguments

  - `turbconv_params`: Turbulence-convection parameters.
  - `ᶜaʲ`: Updraft area fraction [-].
  - `ᶜρaʲ`: Updraft area-weighted density `ρ a` [kg/m³].
  - `ᶜbuoy_inv_time_scale`: Clipped inverse buoyancy time scale [1/s] from
    `detr_buoy_inv_time_scale`.
  - `ᶜmassflux_vert_div`: Vertical divergence of the updraft mass flux
    [kg/(m³ s)].
  - `detr_model`: Detrainment model dispatch tag.

# Returns

The non-negative model-specific detrainment rate [1/s]; zero for the abstract
fallback.
"""
function detrainment_rate(
    turbconv_params,
    ᶜaʲ,
    ᶜρaʲ,
    ᶜbuoy_inv_time_scale,
    ᶜmassflux_vert_div,
    ::AbstractDetrainmentModel,
    ᶜbuoy_and_shell_detr_rate = zero(eltype(ᶜρaʲ)),
)
    return zero(eltype(ᶜρaʲ))
end

function detrainment_rate(
    turbconv_params,
    ᶜaʲ,
    ᶜρaʲ,
    ᶜbuoy_inv_time_scale,
    ᶜmassflux_vert_div,
    ::BuoyancyVelocityDetrainment,
    ᶜbuoy_and_shell_detr_rate = lower_area_limiter_factor(ᶜaʲ, turbconv_params) *
                                CAP.detr_buoy_coeff(turbconv_params) *
                                ᶜbuoy_inv_time_scale,
)
    FT = eltype(ᶜρaʲ)
    c_div_eff = massflux_divergence_detrainment_coeff(
        turbconv_params,
        ᶜbuoy_inv_time_scale,
    )

    area_limiter_factor = lower_area_limiter_factor(ᶜaʲ, turbconv_params)
    detr =
        ᶜbuoy_and_shell_detr_rate -
        area_limiter_factor * c_div_eff * min(ᶜmassflux_vert_div, 0) /
        max(eps(FT), ᶜρaʲ)

    return max(0, detr)
end

"""
    turbulent_entrainment(turbconv_params, ᶜaʲ)

Turbulent (diffusive) entrainment rate [1/s] of an updraft of area fraction
`ᶜaʲ`:

    turb_entr = max(0, c₁ exp(-c₂ aʲ)),

with `(c₁, c₂) = turb_entr_param_vec`. Unlike the dynamical entrainment of
`compute_entrainment`, it does not scale with the updraft velocity, and
it decays as the updraft fills the grid box. Precomputed into
`p.precomputed.ᶜturb_entrʲs` and added to the dynamical rate in
`edmfx_entr_detr_tendency!`.
"""
function turbulent_entrainment(turbconv_params, ᶜaʲ)
    turb_entr_param_vec = CAP.turb_entr_param_vec(turbconv_params)
    return max(turb_entr_param_vec[1] * exp(-turb_entr_param_vec[2] * ᶜaʲ), 0)
end

"""
    edmfx_entr_detr_tendency!(Yₜ, Y, p, t, turbconv_model)

Add the entrainment contribution to the EDMF scalar tendencies (`mse`,
`q_tot`, and the microphysics tracers carried by the updraft).

Detrainment is **not** applied here because it is absorbed into the
analytic implicit ρa solve (see
`solve_sgs_ρa_implicit_stage_analytic!`); scalars are detrained
implicitly through the area divergence of the mass flux.

The entrainment rate is assembled lazily from the precomputed
`ᶜentr_vel_scaleʲs`, `ᶜentr_nonvel_rateʲs`, `ᶜarea_bounding_entr_detrʲs`, and the updraft physical
velocity via `compute_entrainment`, and the turbulent entrainment
`ᶜturb_entrʲs` (`turbulent_entrainment`) is added to it. Each updraft
scalar `χʲ` is relaxed toward its **environment** value,

    ∂χʲ/∂t += (entr + turb_entr) (χ⁰ - χʲ).

The generic method is a no-op; the `PrognosticEDMFX` method mutates
`Yₜ.c.sgsʲs` and returns `nothing`. See the "PROPHET: Closures" page
(`docs/src/prophet_closures.md`).
"""
@inline function buoyancy_sorted_entrainment_deficit(χ⁰, χʲ, σ_χ)
    FT = typeof(σ_χ)
    Δχ = χʲ - χ⁰
    σ_pos = max(σ_χ, FT(0))
    f_dilute = ifelse(
        (Δχ > FT(0)) & (σ_pos > FT(0)),
        Δχ / max(Δχ + σ_pos, eps(FT)),
        FT(1),
    )
    return -f_dilute * Δχ
end

"""
    environment_tail_shift(
        thermo_params, T_0, q_tot_0, q_liq_0, q_ice_0, T_var, q_var, corr_Tq,
        selector,
    )

Return the half-normal conditional shifts `(ΔT, Δq_tot)` from the environment
mean `(T_0, q_tot_0)` to the upper half `s' > 0` of a bivariate Gaussian
subgrid `(T, q_tot)` distribution with variances `(T_var, q_var)` and
correlation `corr_Tq`, conditioned on the linear selector
`s' = α_T T' + α_q q' > 0`:

    ΔT     = √(2/π) · (α_T σ_T² + α_q r σ_T σ_q) / σ_s,
    Δq_tot = √(2/π) · (α_q σ_q² + α_T r σ_T σ_q) / σ_s,

with `σ_s = √(α_T² σ_T² + 2 α_T α_q r σ_T σ_q + α_q² σ_q²)`.

When `selector > 0.5`, `(α_T, α_q) = (c_pm, L_v0)` selects high-MSE air;
otherwise `(α_T, α_q) = (1/T_0, ε_vi - 1)` selects positively buoyant air.
"""
@inline function environment_tail_shift(
    thermo_params,
    T_0::FT,
    q_tot_0::FT,
    q_liq_0::FT,
    q_ice_0::FT,
    T_var::FT,
    q_var::FT,
    corr_Tq::FT,
    selector::FT,
) where {FT}
    cpm = TD.cp_m(thermo_params, q_tot_0, q_liq_0, q_ice_0)
    LH_v0 = TDP.LH_v0(thermo_params)
    ε_vi = TDP.Rv_over_Rd(thermo_params)

    α_T = ifelse(selector > FT(0.5), cpm, FT(1) / max(T_0, eps(FT)))
    α_q = ifelse(selector > FT(0.5), LH_v0, ε_vi - FT(1))

    σ_T = sqrt(max(T_var, zero(FT)))
    σ_q = sqrt(max(q_var, zero(FT)))
    r = clamp(corr_Tq, -one(FT), one(FT))

    σ_s_sq = α_T^2 * σ_T^2 + 2 * α_T * α_q * r * σ_T * σ_q + α_q^2 * σ_q^2
    σ_s = sqrt(max(σ_s_sq, zero(FT)))

    c_tail = sqrt(FT(2) / FT(π))
    ΔT = ifelse(
        σ_s > zero(FT),
        c_tail * (α_T * σ_T^2 + α_q * r * σ_T * σ_q) / σ_s,
        zero(FT),
    )
    Δq_tot = ifelse(
        σ_s > zero(FT),
        c_tail * (α_q * σ_q^2 + α_T * r * σ_T * σ_q) / σ_s,
        zero(FT),
    )
    return (ΔT, Δq_tot)
end

@inline function tail_entrainment_Δmse(
    thermo_params,
    T_0::FT,
    q_tot_0::FT,
    q_liq_0::FT,
    q_ice_0::FT,
    Φ::FT,
    mse_0::FT,
    T_var::FT,
    q_var::FT,
    corr_Tq::FT,
    w::FT,
    selector::FT,
    ρa_0::FT,
    ρa_j::FT,
    entr_phys_j::FT,
    dt::FT,
) where {FT}
    ΔT, Δq_tot = environment_tail_shift(
        thermo_params,
        T_0,
        q_tot_0,
        q_liq_0,
        q_ice_0,
        T_var,
        q_var,
        corr_Tq,
        selector,
    )
    w_cap = max(ρa_0, zero(FT)) / (2 * max(entr_phys_j * ρa_j * dt, eps(FT)))
    w_eff = w * min(one(FT), w_cap)
    T_e = T_0 + w_eff * ΔT
    q_tot_e = min(q_tot_0 + w_eff * Δq_tot, one(FT))
    mse_e = TD.moist_static_energy(
        thermo_params,
        T_e,
        Φ,
        q_tot_e,
        q_liq_0,
        q_ice_0,
    )
    return mse_e - mse_0
end

@inline function tail_entrainment_Δq_tot(
    thermo_params,
    T_0::FT,
    q_tot_0::FT,
    q_liq_0::FT,
    q_ice_0::FT,
    T_var::FT,
    q_var::FT,
    corr_Tq::FT,
    w::FT,
    selector::FT,
    ρa_0::FT,
    ρa_j::FT,
    entr_phys_j::FT,
    dt::FT,
) where {FT}
    _, Δq_tot = environment_tail_shift(
        thermo_params,
        T_0,
        q_tot_0,
        q_liq_0,
        q_ice_0,
        T_var,
        q_var,
        corr_Tq,
        selector,
    )
    w_cap = max(ρa_0, zero(FT)) / (2 * max(entr_phys_j * ρa_j * dt, eps(FT)))
    w_eff = w * min(one(FT), w_cap)
    return min(w_eff * Δq_tot, one(FT) - q_tot_0)
end

function tail_entrainment_increments(Y, p, ᶜmse⁰, ᶜentr_physʲ, ᶜρaʲ, ᶜρa⁰)
    (; params, dt) = p
    FT = eltype(params)
    turbconv_params = CAP.turbconv_params(params)
    w = CAP.edmf_tail_entr_weight(turbconv_params)
    if w == zero(FT) || !hasproperty(p.precomputed, :ᶜT′T′)
        return (zero(FT), zero(FT))
    end
    thermo_params = CAP.thermodynamics_params(params)
    selector = CAP.edmf_tail_entr_selector(turbconv_params)
    (; ᶜT⁰, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰, ᶜT′T′, ᶜq′q′, ᶜcorr_Tq) =
        p.precomputed
    (; ᶜΦ) = p.core

    ᶜΔmse = @. lazy(
        tail_entrainment_Δmse(
            thermo_params,
            ᶜT⁰,
            ᶜq_tot_nonneg⁰,
            ᶜq_liq⁰,
            ᶜq_ice⁰,
            ᶜΦ,
            ᶜmse⁰,
            ᶜT′T′,
            ᶜq′q′,
            ᶜcorr_Tq,
            w,
            selector,
            ᶜρa⁰,
            ᶜρaʲ,
            ᶜentr_physʲ,
            FT(dt),
        ),
    )
    ᶜΔq_tot = @. lazy(
        tail_entrainment_Δq_tot(
            thermo_params,
            ᶜT⁰,
            ᶜq_tot_nonneg⁰,
            ᶜq_liq⁰,
            ᶜq_ice⁰,
            ᶜT′T′,
            ᶜq′q′,
            ᶜcorr_Tq,
            w,
            selector,
            ᶜρa⁰,
            ᶜρaʲ,
            ᶜentr_physʲ,
            FT(dt),
        ),
    )
    return (ᶜΔmse, ᶜΔq_tot)
end

edmfx_entr_detr_tendency!(Yₜ, Y, p, t, turbconv_model) = nothing

function edmfx_entr_detr_tendency!(Yₜ, Y, p, t, turbconv_model::PrognosticEDMFX)

    n = n_mass_flux_subdomains(turbconv_model)
    (; params) = p
    FT = eltype(params)
    turbconv_params = CAP.turbconv_params(params)
    c_sort = CAP.entr_buoyancy_sorting_coeff(params)
    Lv0 = CAP.LH_v0(params)
    cp_d = CAP.cp_d(params)
    (;
        ᶜturb_entrʲs,
        ᶜentr_vel_scaleʲs,
        ᶜentr_nonvel_rateʲs,
        ᶜarea_bounding_entr_detrʲs,
        ᶜuʲs,
        ᶜρʲs,
    ) = p.precomputed

    ᶜmse⁰ = ᶜspecific_env_mse(Y, p)
    ᶜq_tot⁰ = ᶜspecific_env_value(@name(q_tot), Y, p)
    ᶜρa⁰ = @. lazy(ρa⁰(Y.c.ρ, Y.c.sgsʲs, turbconv_model))
    ᶜlg = Fields.local_geometry_field(Y.c)

    for j in 1:n
        ᶜwʲ = @. lazy(get_physical_w(ᶜuʲs.:($$j), ᶜlg))
        ᶜentr_physʲ = @. lazy(
            ᶜentr_vel_scaleʲs.:($$j) * abs(ᶜwʲ) + ᶜentr_nonvel_rateʲs.:($$j),
        )
        ᶜentrʲ = @. lazy(
            compute_entrainment(
                ᶜentr_vel_scaleʲs.:($$j),
                ᶜentr_nonvel_rateʲs.:($$j),
                ᶜarea_bounding_entr_detrʲs.:($$j),
                ᶜwʲ,
            ),
        )
        ᶜturb_entrʲ = ᶜturb_entrʲs.:($j)
        ᶜmseʲ = Y.c.sgsʲs.:($j).mse
        ᶜq_totʲ = Y.c.sgsʲs.:($j).q_tot
        ᶜρaʲ = Y.c.sgsʲs.:($j).ρa
        a_max = CAP.max_area(turbconv_params)
        ᶜρa⁰_safe = @. lazy(max(ᶜρa⁰, Y.c.ρ * (1 - a_max), eps(FT)))

        if c_sort > FT(0) && hasproperty(p.precomputed, :ᶜq′q′)
            (; ᶜT′T′, ᶜq′q′, ᶜcorr_Tq) = p.precomputed
            @. Yₜ.c.sgsʲs.:($$j).mse +=
                (ᶜentrʲ .+ ᶜturb_entrʲ) *
                buoyancy_sorted_entrainment_deficit(
                    ᶜmse⁰,
                    ᶜmseʲ,
                    c_sort * (
                        Lv0 * sqrt(max(FT(0), ᶜq′q′)) +
                        cp_d * max(FT(0), ᶜcorr_Tq) * sqrt(max(FT(0), ᶜT′T′))
                    ),
                )

            @. Yₜ.c.sgsʲs.:($$j).q_tot +=
                (ᶜentrʲ .+ ᶜturb_entrʲ) *
                buoyancy_sorted_entrainment_deficit(
                    ᶜq_tot⁰,
                    ᶜq_totʲ,
                    c_sort * sqrt(max(FT(0), ᶜq′q′)),
                )
        elseif CAP.edmf_tail_entr_weight(turbconv_params) > FT(0) &&
               hasproperty(p.precomputed, :ᶜT′T′)
            # Bivariate Gaussian tail shifts (Δmse, Δq_tot) for physical
            # dynamical entrainment when `c_sort == 0`. The factor `1 + ρaʲ/ρa⁰`
            # accounts for the opposite shift of the residual environment at
            # fixed grid-mean scalar, and the shift is bounded by
            # `max(0, χʲ - χ⁰)` when `χ⁰ < χʲ` so entraining cooler/drier
            # environmental air never increases updraft `mseʲ` or `q_totʲ`.
            ᶜΔmse, ᶜΔq_tot =
                tail_entrainment_increments(Y, p, ᶜmse⁰, ᶜentr_physʲ, ᶜρaʲ, ᶜρa⁰)
            @. Yₜ.c.sgsʲs.:($$j).mse +=
                (ᶜentrʲ .+ ᶜturb_entrʲ) * (ᶜmse⁰ - ᶜmseʲ) +
                ᶜentr_physʲ * min(
                    (1 + ᶜρaʲ / ᶜρa⁰_safe) * ᶜΔmse,
                    max(FT(0), ᶜmseʲ - ᶜmse⁰),
                )

            @. Yₜ.c.sgsʲs.:($$j).q_tot +=
                (ᶜentrʲ .+ ᶜturb_entrʲ) * (ᶜq_tot⁰ - ᶜq_totʲ) +
                ᶜentr_physʲ * min(
                    (1 + ᶜρaʲ / ᶜρa⁰_safe) * ᶜΔq_tot,
                    max(FT(0), ᶜq_totʲ - ᶜq_tot⁰),
                )
        else
            @. Yₜ.c.sgsʲs.:($$j).mse +=
                (ᶜentrʲ .+ ᶜturb_entrʲ) * (ᶜmse⁰ - ᶜmseʲ)

            @. Yₜ.c.sgsʲs.:($$j).q_tot +=
                (ᶜentrʲ .+ ᶜturb_entrʲ) * (ᶜq_tot⁰ - ᶜq_totʲ)
        end

        # Auto-discovered SGS tracers (microphysics species and any
        # user-defined passive tracers)
        for χ_name in sgs_tracer_names(Y)
            ᶜχ⁰ = ᶜspecific_env_value(χ_name, Y, p)
            ᶜχʲ = MatrixFields.get_field(Y.c.sgsʲs.:($j), χ_name)
            ᶜχʲₜ = MatrixFields.get_field(Yₜ.c.sgsʲs.:($j), χ_name)
            @. ᶜχʲₜ += (ᶜentrʲ .+ ᶜturb_entrʲ) * (ᶜχ⁰ - ᶜχʲ)
        end
    end
    return nothing
end

# Stability limiters that cap entrainment and detrainment rates below the
# inverse timescales w/dz and 1/dt. Written for the (since removed) diagnostic
# EDMF; currently unused.
limit_entrainment(entr::FT, a, dt) where {FT} = max(
    min(
        entr,
        FT(0.9) * (1 - a) / max(a, eps(FT)) / dt,
        FT(0.9) * 1 / dt,
    ),
    0,
)
limit_entrainment(entr::FT, a, w, dz) where {FT} =
    max(min(entr, FT(0.9) * w / dz), 0)

limit_detrainment(detr::FT, a, dt) where {FT} =
    max(min(detr, FT(0.9) * 1 / dt), 0)
limit_detrainment(detr::FT, a, w, dz, dt) where {FT} =
    limit_detrainment(max(min(detr, FT(0.9) * w / dz), 0), a, dt)

function limit_turb_entrainment(dyn_entr::FT, turb_entr, w, dz) where {FT}
    return max(min((FT(0.9) * w / dz) - dyn_entr, turb_entr), 0)
end
