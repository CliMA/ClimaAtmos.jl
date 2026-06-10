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

Calculates non-dimensional Π-groups used in EDMF entrainment/detrainment models.

Arguments:
- `elev_above_sfc`: Difference between cell-center height and surface elevation (ᶜz - z_sfc) [m].
- `ref_H`: Reference pressure scale height [m].
- `ᶜaʲ`: Updraft area fraction [-].
- `ᶜwʲ`: Updraft physical vertical velocity [m/s].
- `ᶜRHʲ`: Updraft relative humidity [-].
- `ᶜbuoyʲ`: Updraft buoyancy [m/s²].
- `ᶜw⁰`: Environment physical vertical velocity [m/s].
- `ᶜRH⁰`: Environment relative humidity [-].
- `ᶜbuoy⁰`: Environment buoyancy [m/s²].
- `ᶜtke`: Turbulent kinetic energy [m²/s²].

Returns a tuple of five Π-groups: (Π₁, Π₂, Π₃, Π₄, Π₅).
Π₁: Related to buoyancy difference and velocity difference.
Π₂: Related to TKE and velocity difference.
Π₃: Related to updraft area.
Π₄: Related to relative humidity difference.
Π₅: Related to normalized elevation above surface.
Π₁ and Π₂ are scaled by empirical factors (100 and 2 respectively) to 
become O(1) and then clipped to the range [-1, 1].
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
    detr_area_limiter_factor(entr::FT, detr, a, turbconv_params) where {FT}

Return a multiplicative limiter for the net area tendency based on the current
subdomain area `a`.

If `entr > detr`, the net tendency would increase area, so the limiter damps
growth as `a` approaches the upper bound `a_max`. In that case the returned
factor is

    max(0, 1 - a / a_max)^max_area_limiter_power.

If `entr < detr`, the net tendency would decrease area, so the limiter damps
shrinkage as `a` approaches the lower bound `a_min`. In that case the returned
factor is

    max(0, 1 - a_min / a)^min_area_limiter_power.

Thus the limiter is close to one away from the active bound and smoothly
approaches zero at the corresponding bound. The lower and upper bounds are
obtained from `turbconv_params` through `CAP.min_area` and `CAP.max_area`.
"""
@inline function detr_area_limiter_factor(entr::FT, detr, a, turbconv_params) where {FT}
    a_min = CAP.min_area(turbconv_params)
    a_max = CAP.max_area(turbconv_params)
    min_area_limiter_power = CAP.min_area_limiter_power(turbconv_params)
    max_area_limiter_power = CAP.max_area_limiter_power(turbconv_params)
    a_safe = max(eps(FT), a)
    return ifelse(
        entr > detr,
        (max(0, (1 - a_safe / a_max)))^max_area_limiter_power,
        (max(0, (1 - a_min / a_safe)))^min_area_limiter_power,
    )
end

"""
    compute_entrainment(ᶜentr_vel_scale, ᶜentr_nonvel, ᶜwʲ)

Total entrainment rate [1/s] as the sum of a velocity-proportional term
and a background (time-scale) term:

    entr = entr_vel_scale * |wʲ| + entr_nonvel

`entr_vel_scale` [1/m] and `entr_nonvel` [1/s] are precomputed by
`entrainment_velocity_scale` and `nonvelocity_entrainment` respectively.
`ᶜwʲ` is the physical updraft vertical velocity [m/s] (not a difference
from the environment); the environment velocity is approximated as zero
when computing the scale (see `set_prognostic_edmf_precomputed_quantities_explicit_closures!`).
"""
compute_entrainment(
    ᶜentr_vel_scale,
    ᶜentr_nonvel,
    ᶜwʲ,
) = ᶜentr_vel_scale * abs(ᶜwʲ) + ᶜentr_nonvel

"""
    entr_area_limiter_factor(ᶜaʲ, turbconv_params)

Return a multiplicative limiter for the entrainment rate based on the 
current subdraft area `ᶜaʲ`.

The factor smoothly damps entrainment as the area approaches 1 (fully
occupied):

    (1 - clamp(ᶜaʲ, 0, 1))^entr_mult_limiter_coeff

where `entr_mult_limiter_coeff` is obtained from `turbconv_params`.
"""
@inline function entr_area_limiter_factor(ᶜaʲ, turbconv_params)
    FT = eltype(ᶜaʲ)
    entr_mult_limiter_coeff = CAP.entr_mult_limiter_coeff(turbconv_params)
    return (FT(1) - min(max(ᶜaʲ, FT(0)), FT(1)))^entr_mult_limiter_coeff
end

"""
    entrainment_velocity_scale(
        thermo_params, turbconv_params, ᶜz, z_sfc, ᶜp, ᶜρ,
        ᶜaʲ, ᶜwʲ, ᶜRHʲ, ᶜbuoyʲ, ᶜw⁰, ᶜRH⁰, ᶜbuoy⁰, ᶜtke,
        model_option::AbstractEntrainmentModel,
    )

    and

    nonvelocity_entrainment(
        turbconv_params, ᶜaʲ,
        model_option::AbstractEntrainmentModel,
    )

These two functions together decompose the total entrainment rate into
a velocity-dependent part and a background (time-scale) part:

    entr = entr_vel_scale * abs(wʲ) + entr_nonvel

where `entr` has units of [1/s], `entr_vel_scale` has units of [1/m],
and `entr_nonvel` has units of [1/s]. `entrainment_velocity_scale` computes
the velocity-scaling prefactor [1/m] and `nonvelocity_entrainment` computes
the velocity-independent rate [1/s], both according to the model specified 
by `model_option`. The total entrainment rate [1/s] is assembled by 
`compute_entrainment`.

The specific formulation depends on the concrete type passed as
`model_option`. This argument dispatches to different entrainment models,
such as `NoEntrainment` (which returns zero entrainment),
`PiGroupsEntrainment` (which computes an entrainment scale based on
a linear combination of scaled non-dimensional Pi-groups together with a
constant background entrainment term), or `InvZEntrainment`
(a physically inspired formulation with a scale proportional to `1/z`).
Each model implements a distinct physical or empirical representation
of entrainment in updrafts.

Arguments for `entrainment_velocity_scale` (all cell-centered):
- `thermo_params`: Thermodynamic parameters.
- `turbconv_params`: Turbulence convection parameters.
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
- `model_option`: Object specifying the entrainment model
                  (e.g., `NoEntrainment`, `PiGroupsEntrainment`,
                  or `InvZEntrainment`).

Arguments for `nonvelocity_entrainment`:
- `turbconv_params`: Turbulence convection parameters.
- `ᶜaʲ`: Updraft area fraction [-].
- `model_option`: Object specifying the entrainment model.

Returns (for each function):
- `entrainment_velocity_scale`: Velocity-scaling prefactor [1/m].
- `nonvelocity_entrainment`: Background (velocity-independent) entrainment rate [1/s].

These quantities are combined by `compute_entrainment` as

    entr = entr_vel_scale * abs(w⁰ - wʲ) + entr_nonvel

to obtain the total entrainment rate [1/s].
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
    ::NoEntrainment,
)
    return zero(eltype(thermo_params))
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

    Π₁, Π₂, Π₃, Π₄, Π₅ = calculate_pi_groups(
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

    entr_param_vec = CAP.entr_param_vec(turbconv_params)
    pi_sum =
        entr_param_vec[1] * abs(Π₁) +
        entr_param_vec[2] * abs(Π₂) +
        entr_param_vec[3] * abs(Π₃) +
        entr_param_vec[4] * abs(Π₄) +
        entr_param_vec[5] * abs(Π₅) +
        entr_param_vec[6]

    area_limiter_factor = entr_area_limiter_factor(ᶜaʲ, turbconv_params)
    entr_vel_scale = area_limiter_factor * max(0, pi_sum) / elev_above_sfc
    return max(0, entr_vel_scale)
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
    entr_vel_scale_param = CAP.entr_coeff(turbconv_params)

    elev_above_sfc = ᶜz - z_sfc
    # If elevation above surface is not positive, terms like 1/elev_above_sfc
    # become singular. Model assumes operation above the surface.
    if elev_above_sfc <= eps(FT)
        return 0
    end

    area_limiter_factor = entr_area_limiter_factor(ᶜaʲ, turbconv_params)
    entr_vel_scale = area_limiter_factor * entr_vel_scale_param / elev_above_sfc
    return max(0, entr_vel_scale)
end

function nonvelocity_entrainment(
    turbconv_params,
    ᶜaʲ,
    ::NoEntrainment,
)
    return zero(eltype(ᶜaʲ))
end

function nonvelocity_entrainment(
    turbconv_params,
    ᶜaʲ,
    ::PiGroupsEntrainment,
)
    entr_inv_tau = CAP.entr_inv_tau(turbconv_params)
    area_limiter_factor = entr_area_limiter_factor(ᶜaʲ, turbconv_params)
    entr_nonvel = area_limiter_factor * entr_inv_tau
    return max(0, entr_nonvel)
end

function nonvelocity_entrainment(
    turbconv_params,
    ᶜaʲ,
    ::InvZEntrainment,
)
    entr_inv_tau = CAP.entr_inv_tau(turbconv_params)
    min_area_limiter_scale = CAP.min_area_limiter_scale(turbconv_params)
    min_area_limiter_power = CAP.min_area_limiter_power(turbconv_params)
    a_min = CAP.min_area(turbconv_params)

    # Extra entrainment for a < a_min that smoothly relaxes the area
    # toward a_min (0 at a_min, min_area_limiter_scale at a = 0).
    min_area_limiter =
        min_area_limiter_scale * (max(0, a_min - ᶜaʲ) / a_min)^min_area_limiter_power

    area_limiter_factor = entr_area_limiter_factor(ᶜaʲ, turbconv_params)
    entr_nonvel = area_limiter_factor * (entr_inv_tau + min_area_limiter)
    return max(0, entr_nonvel)
end

"""
    compute_detrainment(turbconv_params, aʲ, ρaʲ, Δwʲ, Δbuoyʲ,
                        massflux_vert_div, entr, detr_nonvel, detr_model)

Total detrainment rate [1/s].

Calls `detrainment_rate` for the model-specific rate, adds `detr_nonvel`
(e.g., background detrainment), then applies an area limiter so that near
`a_min`/`a_max` the net area tendency `(entr - detr)` smoothly approaches zero.

Arguments:
- `turbconv_params`: Turbulence convection parameters.
- `aʲ`: Updraft area fraction [-].
- `ρaʲ`: Updraft density-area product [kg/m³].
- `Δwʲ`: Updraft physical vertical velocity [m/s].
- `Δbuoyʲ`: Updraft buoyancy (vertical acceleration) [m/s²].
- `massflux_vert_div`: Vertical divergence of the updraft mass flux [kg/(m³ s)].
- `entr`: Total entrainment rate [1/s].
- `detr_nonvel`: Additional (e.g., background) detrainment rate [1/s].
- `detr_model`: Object specifying the detrainment model.

Returns the total detrainment rate [1/s].
"""
function compute_detrainment(
    turbconv_params,
    aʲ,
    ρaʲ,
    Δwʲ,
    Δbuoyʲ,
    massflux_vert_div,
    entr,
    detr_nonvel,
    detr_model,
)

    detr =
        detrainment_rate(
            turbconv_params,
            ρaʲ,
            Δwʲ,
            Δbuoyʲ,
            massflux_vert_div,
            detr_model,
        ) + detr_nonvel

    # Adjust detrainment so that near the area bounds it approaches entrainment,
    # causing the net area tendency (entr - detr) to smoothly go to zero.
    factor = detr_area_limiter_factor(entr, detr, aʲ, turbconv_params)
    return factor * detr + (1 - factor) * entr
end

"""
    detrainment_rate(turbconv_params, ᶜρaʲ, ᶜΔwʲ, ᶜΔbuoyʲ, ᶜmassflux_vert_div,
                     detr_model::AbstractDetrainmentModel)

Model-specific detrainment rate [1/s] for a given detrainment model.

This abstract fallback returns zero; concrete subtypes of
`AbstractDetrainmentModel` should override this method to provide a
non-trivial rate.

Arguments:
- `turbconv_params`: Turbulence convection parameters.
- `ᶜρaʲ`: Updraft density-area product [kg/m³].
- `ᶜΔwʲ`: Updraft physical vertical velocity [m/s].
- `ᶜΔbuoyʲ`: Updraft buoyancy (vertical acceleration) [m/s²].
- `ᶜmassflux_vert_div`: Vertical divergence of the updraft mass flux [kg/(m³ s)].
- `detr_model`: Detrainment model dispatch tag.

Returns the model-specific detrainment rate [1/s] (zero for the abstract fallback).
"""
function detrainment_rate(
    turbconv_params,
    ᶜρaʲ,
    ᶜΔwʲ,
    ᶜΔbuoyʲ,
    ᶜmassflux_vert_div,
    ::AbstractDetrainmentModel,
)
    return zero(eltype(ᶜρaʲ))
end
function nonvelocity_detrainment(
    turbconv_params,
    ᶜaʲ,
    ::AbstractDetrainmentModel,
)
    return zero(eltype(ᶜaʲ))
end

function detrainment_rate(
    turbconv_params,
    ᶜρaʲ,
    ᶜΔwʲ,
    ᶜΔbuoyʲ,
    ᶜmassflux_vert_div,
    ::BuoyancyVelocityDetrainment,
)
    FT = eltype(ᶜρaʲ)
    detr_buoy_coeff = CAP.detr_buoy_coeff(turbconv_params)
    detr_buoy_inv_tau_max = CAP.detr_buoy_inv_tau_max(turbconv_params)
    detr_massflux_vertdiv_coeff =
        CAP.detr_massflux_vertdiv_coeff(turbconv_params)

    # Clip buoyancy time scale to `buoy_inv_time_scale` to avoid too fast detrainment
    # when velocity is small
    buoy_inv_time_scale =
        min(
            detr_buoy_inv_tau_max,
            abs(min(ᶜΔbuoyʲ, 0)) / max(eps(FT), abs(ᶜΔwʲ)),
        )

    detr =
        detr_buoy_coeff * buoy_inv_time_scale -
        detr_massflux_vertdiv_coeff * min(ᶜmassflux_vert_div, 0) / max(eps(FT), ᶜρaʲ)

    return max(0, detr)
end

function nonvelocity_detrainment(
    turbconv_params,
    ᶜaʲ,
    ::BuoyancyVelocityDetrainment,
)
    detr_inv_tau = CAP.detr_inv_tau(turbconv_params)
    max_area_limiter_scale = CAP.max_area_limiter_scale(turbconv_params)
    max_area_limiter_power = CAP.max_area_limiter_power(turbconv_params)
    a_max = CAP.max_area(turbconv_params)

    # Extra detrainment for a > a_max that smoothly relaxes the area back
    # toward a_max (0 at a_max, max_area_limiter_scale at a = 1).
    max_area_limiter =
        max_area_limiter_scale * (max(0, ᶜaʲ - a_max) / (1 - a_max))^max_area_limiter_power

    detr =
        detr_inv_tau +
        max_area_limiter

    return max(0, detr)
end

function turbulent_entrainment(turbconv_params, ᶜaʲ)
    turb_entr_param_vec = CAP.turb_entr_param_vec(turbconv_params)
    return max(turb_entr_param_vec[1] * exp(-turb_entr_param_vec[2] * ᶜaʲ), 0)
end

edmfx_entr_detr_tendency!(Yₜ, Y, p, t, turbconv_model) = nothing

function edmfx_entr_detr_tendency!(Yₜ, Y, p, t, turbconv_model::PrognosticEDMFX)

    n = n_mass_flux_subdomains(turbconv_model)
    (; ᶜturb_entrʲs, ᶜentrʲs, ᶜdetrʲs) = p.precomputed

    ᶜmse⁰ = ᶜspecific_env_mse(Y, p)
    ᶜq_tot⁰ = ᶜspecific_env_value(@name(q_tot), Y, p)

    microphysics_tracers = (
        (@name(c.sgsʲs.:(1).q_lcl), @name(q_lcl)),
        (@name(c.sgsʲs.:(1).q_icl), @name(q_icl)),
        (@name(c.sgsʲs.:(1).q_rai), @name(q_rai)),
        (@name(c.sgsʲs.:(1).q_sno), @name(q_sno)),
        (@name(c.sgsʲs.:(1).n_lcl), @name(n_lcl)),
        (@name(c.sgsʲs.:(1).n_rai), @name(n_rai)),
        (@name(c.sgsʲs.:(1).n_ice), @name(n_ice)),
        (@name(c.sgsʲs.:(1).q_rim), @name(q_rim)),
        (@name(c.sgsʲs.:(1).b_rim), @name(b_rim)),
    )

    for j in 1:n
        ᶜentrʲ = ᶜentrʲs.:($j)
        ᶜdetrʲ = ᶜdetrʲs.:($j)
        ᶜturb_entrʲ = ᶜturb_entrʲs.:($j)
        ᶜmseʲ = Y.c.sgsʲs.:($j).mse
        ᶜq_totʲ = Y.c.sgsʲs.:($j).q_tot

        @. Yₜ.c.sgsʲs.:($$j).ρa += Y.c.sgsʲs.:($$j).ρa * (ᶜentrʲ - ᶜdetrʲ)

        @. Yₜ.c.sgsʲs.:($$j).mse += (ᶜentrʲ .+ ᶜturb_entrʲ) * (ᶜmse⁰ - ᶜmseʲ)

        @. Yₜ.c.sgsʲs.:($$j).q_tot +=
            (ᶜentrʲ .+ ᶜturb_entrʲ) * (ᶜq_tot⁰ - ᶜq_totʲ)

        MatrixFields.unrolled_foreach(microphysics_tracers) do (χʲ_name, χ_name)
            MatrixFields.has_field(Y, χʲ_name) || return
            ᶜχ⁰ = ᶜspecific_env_value(χ_name, Y, p)
            ᶜχʲ = MatrixFields.get_field(Y, χʲ_name)
            ᶜχʲₜ = MatrixFields.get_field(Yₜ, χʲ_name)
            @. ᶜχʲₜ += (ᶜentrʲ .+ ᶜturb_entrʲ) * (ᶜχ⁰ - ᶜχʲ)
        end

        # SCRATCH PROTOTYPE (precip-detrainment, ENV-gated by TRMM_TAU_PD; see
        # results/ws4c-precip-detr-impl.md). Relax the FALLING-PRECIP updraft
        # tracers toward the environment at rate 1/τ_pd, on top of the
        # entr-driven relaxation, to emulate the rain shaft separating from the
        # buoyant core that a single-updraft EDMF cannot resolve. Cloud condensate
        # (q_lcl/q_icl/n_lcl/n_ice) is NOT detrained — only the precip shaft.
        # Grid-mean conserving: q_rai⁰ is diagnostic, so reducing q_raiʲ moves rain
        # from the updraft partition to the environment partition (verify in-run).
        # The matching Jacobian diagonal is in manual_sparse_jacobian.jl.
        let τ_pd = _precip_detr_tau(eltype(Y.c.ρ))
            if isfinite(τ_pd)
                precip_detr = inv(τ_pd)
                precip_tracers = (
                    (@name(c.sgsʲs.:(1).q_rai), @name(q_rai)),
                    (@name(c.sgsʲs.:(1).n_rai), @name(n_rai)),
                    (@name(c.sgsʲs.:(1).q_sno), @name(q_sno)),
                    (@name(c.sgsʲs.:(1).q_rim), @name(q_rim)),
                    (@name(c.sgsʲs.:(1).b_rim), @name(b_rim)),
                )
                MatrixFields.unrolled_foreach(precip_tracers) do (χʲ_name, χ_name)
                    MatrixFields.has_field(Y, χʲ_name) || return
                    ᶜχ⁰ = ᶜspecific_env_value(χ_name, Y, p)
                    ᶜχʲ = MatrixFields.get_field(Y, χʲ_name)
                    ᶜχʲₜ = MatrixFields.get_field(Yₜ, χʲ_name)
                    @. ᶜχʲₜ += precip_detr * (ᶜχ⁰ - ᶜχʲ)
                end
            end
        end
    end
    return nothing
end

"""
    _precip_detr_tau(::Type{FT})

SCRATCH-prototype knob for the precip-detrainment timescale τ_pd [s], read from
the `TRMM_TAU_PD` environment variable. Returns `FT(Inf)` (term disabled =
baseline) when unset or `"inf"`. See results/ws4c-precip-detr-impl.md.
"""
function _precip_detr_tau(::Type{FT}) where {FT}
    v = get(ENV, "TRMM_TAU_PD", "inf")
    return (v == "" || lowercase(v) == "inf") ? FT(Inf) : FT(parse(Float64, v))
end

"""
    edmfx_first_interior_entr_tendency!(Yₜ, Y, p, t, turbconv_model::PrognosticEDMFX)

Apply first-interior–level entrainment tendencies for each EDMF updraft.

This routine adds entrainment tendencies for moist static energy (`mse`) and total
humidity (`q_tot`) in the first model cell.  
The entrained tracer value is taken from `sgs_scalar_first_interior_bc`.
"""
edmfx_first_interior_entr_tendency!(Yₜ, Y, p, t, turbconv_model) = nothing
function edmfx_first_interior_entr_tendency!(
    Yₜ,
    Y,
    p,
    t,
    turbconv_model::PrognosticEDMFX,
)

    (; params, dt) = p
    (; ᶜK, ᶜρʲs, ᶜentrʲs) = p.precomputed

    FT = eltype(params)
    n = n_mass_flux_subdomains(p.atmos.turbconv_model)
    thermo_params = CAP.thermodynamics_params(params)
    turbconv_params = CAP.turbconv_params(params)
    ᶜaʲ_int_val = p.scratch.temp_data_level
    ᶜmse_buoyant_air_int_val = p.scratch.temp_data_level_2
    ᶜq_tot_buoyant_air_int_val = p.scratch.temp_data_level_3

    (;
        ustar,
        obukhov_length,
        buoyancy_flux,
        ρ_flux_h_tot,
        ρ_flux_q_tot,
        ustar,
        obukhov_length,
    ) =
        p.precomputed.sfc_conditions

    ᶜz_int_val = Fields.field_values(Fields.level(Fields.coordinate_field(Y.c).z, 1))
    z_sfc_val =
        Fields.field_values(Fields.level(Fields.coordinate_field(Y.f).z, Fields.half))
    ᶜρ_int_val = Fields.field_values(Fields.level(Y.c.ρ, 1))

    buoyancy_flux_val = Fields.field_values(buoyancy_flux)
    ρ_flux_h_tot_val = Fields.field_values(ρ_flux_h_tot)
    ρ_flux_q_tot_val = Fields.field_values(ρ_flux_q_tot)

    ustar_val = Fields.field_values(ustar)
    obukhov_length_val = Fields.field_values(obukhov_length)
    sfc_local_geometry_val = Fields.field_values(
        Fields.local_geometry_field(Fields.level(Y.f, Fields.half)),
    )

    (; ᶜh_tot) = p.precomputed
    ᶜh_tot_int_val = Fields.field_values(Fields.level(ᶜh_tot, 1))
    ᶜK_int_val = Fields.field_values(Fields.level(ᶜK, 1))
    ᶜmse⁰ = ᶜspecific_env_mse(Y, p)
    env_mse_int_val = Fields.field_values(Fields.level(ᶜmse⁰, 1))

    ᶜq_tot = @. lazy(specific(Y.c.ρq_tot, Y.c.ρ))
    ᶜq_tot_int_val = Fields.field_values(Fields.level(ᶜq_tot, 1))
    ᶜq_tot⁰ = ᶜspecific_env_value(@name(q_tot), Y, p)
    env_q_tot_int_val = Fields.field_values(Fields.level(ᶜq_tot⁰, 1))

    for j in 1:n

        # Apply entrainment tendencies in the first model cell for moist static energy (mse) 
        # and total humidity (q_tot). The entrained fluid is assumed to have a scalar value 
        # given by `sgs_scalar_first_interior_bc` (mean + SGS perturbation). Since 
        # `edmfx_entr_detr_tendency!` computes entrainment based on the environment–updraft 
        # contrast, we supply the high-value (entrained) tracer minus the environment value 
        # here to form the correct tendency.
        entr_int_val = Fields.field_values(Fields.level(ᶜentrʲs.:($j), 1))
        sgsʲs_ρ_int_val = Fields.field_values(Fields.level(ᶜρʲs.:($j), 1))
        sgsʲs_ρa_int_val = Fields.field_values(Fields.level(Y.c.sgsʲs.:($j).ρa, 1))
        @. ᶜaʲ_int_val = max(
            FT(turbconv_params.surface_area),
            draft_area(sgsʲs_ρa_int_val, sgsʲs_ρ_int_val),
        )

        sgsʲs_mseₜ_int_val =
            Fields.field_values(Fields.level(Yₜ.c.sgsʲs.:($j).mse, 1))
        @. ᶜmse_buoyant_air_int_val = sgs_scalar_first_interior_bc(
            ᶜz_int_val - z_sfc_val,
            ᶜρ_int_val,
            ᶜaʲ_int_val,
            ᶜh_tot_int_val - ᶜK_int_val,
            buoyancy_flux_val,
            ρ_flux_h_tot_val,
            ustar_val,
            obukhov_length_val,
            sfc_local_geometry_val,
        )
        @. sgsʲs_mseₜ_int_val += entr_int_val * (ᶜmse_buoyant_air_int_val - env_mse_int_val)

        sgsʲs_q_totₜ_int_val =
            Fields.field_values(Fields.level(Yₜ.c.sgsʲs.:($j).q_tot, 1))
        @. ᶜq_tot_buoyant_air_int_val = sgs_scalar_first_interior_bc(
            ᶜz_int_val - z_sfc_val,
            ᶜρ_int_val,
            ᶜaʲ_int_val,
            ᶜq_tot_int_val,
            buoyancy_flux_val,
            ρ_flux_q_tot_val,
            ustar_val,
            obukhov_length_val,
            sfc_local_geometry_val,
        )
        @. sgsʲs_q_totₜ_int_val +=
            entr_int_val * (ᶜq_tot_buoyant_air_int_val - env_q_tot_int_val)

    end
end

"""
    set_first_cell_entr_detr_bc!(
        ρaʲ_int, ρʲ_int, entr_nonvel_int, entr_vel_scale_int, detr_nonvel_int,
        buoyancy_flux, dz_int, surface_area, dt, FT,
    )

Apply surface boundary conditions for entrainment and detrainment at the first
model cell for a single EDMF updraft.

When `buoyancy_flux < 0` (convectively stable): the background entrainment and
entrainment velocity scale are left unchanged; `detr_nonvel` is floored at
`entr_nonvel` to prevent net area growth from the lower boundary.

When `buoyancy_flux ≥ 0` (convectively unstable): the updraft area is nudged
toward `surface_area`:
- `ρaʲ` is seeded so a zero-area plume can start growing.
- If `a < surface_area`: `entr_nonvel` is set to `(surface_area/a − 1)/dt` and
  `entr_vel_scale` is set to `2/dz` (the kinematic one-sided estimate of
  `∂(ρaw)/∂z` at the surface), so entrainment grows the area.
- If `a ≥ surface_area`: `detr_nonvel` is set to drive area back down;
  `entr_vel_scale` is still `2/dz`.

!!! note "Two-place first-cell treatment"
    The first-cell boundary condition is applied in two stages:
    1. This function sets the input coefficients (bg_entr, entr_coeff, detr_nonvel).
    2. `update_prognostic_edmfx_entr_detr!` overrides the total detrainment at
        level 1 to strip the area limiter added by `compute_detrainment`.
    These two stages should be kept in sync.

Arguments (all level-1 field-value slices unless noted):
- `ρaʲ_int`             — updraft `ρa` [kg/m³] (read/write; seeded first)
- `ρʲ_int`              — updraft density [kg/m³] (read-only)
- `entr_nonvel_int`         — background entrainment [1/s] (read/write)
- `entr_vel_scale_int`  — entrainment velocity scale [1/m] (read/write)
- `detr_nonvel_int`     — additional detrainment [1/s] (read/write)
- `aʲ_int`              — scratch for draft area [-] (write-only; pre-allocated by caller)
- `buoyancy_flux`       — surface buoyancy flux [m²/s³] (read-only scalar field values)
- `dz_int`              — first-cell height [m] (read-only)
- `surface_area`        — target updraft area fraction [-] (scalar)
- `dt`                  — timestep [s] (scalar)
- `FT`                  — float type
"""
function set_first_cell_entr_detr_bc!(
    ρaʲ_int,
    ρʲ_int,
    entr_nonvel_int,
    entr_vel_scale_int,
    detr_nonvel_int,
    aʲ_int,
    buoyancy_flux,
    dz_int,
    surface_area,
    dt,
    ::Type{FT},
) where {FT}
    # Seed a small positive area when buoyancy flux is non-negative so that
    # an initially zero-area plume can grow toward `surface_area`.
    @. ρaʲ_int += ifelse(buoyancy_flux < 0,
        FT(0),
        max(FT(0), ρʲ_int * eps(FT) - ρaʲ_int),
    )
    @. aʲ_int = draft_area(ρaʲ_int, ρʲ_int)

    @. entr_nonvel_int = ifelse(
        buoyancy_flux < 0 || aʲ_int >= FT(surface_area),
        entr_nonvel_int,
        (FT(surface_area) / aʲ_int - 1) / FT(dt),
    )
    # Replace entrainment coefficient with the kinematic estimate 2/dz when
    # buoyancy flux is positive so the total entrainment includes the advective
    # area flux ∂(ρaw)/∂z at the lower boundary (one-sided, zero below surface).
    @. entr_vel_scale_int = ifelse(
        buoyancy_flux < 0,
        entr_vel_scale_int,
        FT(2) / dz_int,
    )
    @. detr_nonvel_int = ifelse(
        buoyancy_flux < 0,
        max(detr_nonvel_int, entr_nonvel_int),
        ifelse(
            aʲ_int < FT(surface_area),
            FT(0),
            entr_nonvel_int - (FT(surface_area) / aʲ_int - 1) / FT(dt),
        ),
    )
    return nothing
end

# limit entrainment and detrainment rates for prognostic EDMFX
# limit rates approximately below the inverse timescale 1/dt
limit_entrainment(entr::FT, a, dt) where {FT} = max(
    min(
        entr,
        FT(0.9) * (1 - a) / max(a, eps(FT)) / dt,
        FT(0.9) * 1 / dt,
    ),
    0,
)
limit_detrainment(detr::FT, a, dt) where {FT} =
    max(min(detr, FT(0.9) * 1 / dt), 0)
limit_detrainment(detr::FT, entr::FT, a, dt) where {FT} =
    max(detr, entr - FT(0.9) * 1 / dt)

function limit_turb_entrainment(dyn_entr::FT, turb_entr, dt) where {FT}
    return max(min((FT(0.9) * 1 / dt) - dyn_entr, turb_entr), 0)
end

# limit entrainment and detrainment rates for diagnostic EDMF
# limit rates approximately below the inverse timescale w/dz
limit_entrainment(entr::FT, a, w, dz) where {FT} =
    max(min(entr, FT(0.9) * w / dz), 0)

limit_detrainment(detr::FT, a, w, dz, dt) where {FT} =
    limit_detrainment(max(min(detr, FT(0.9) * w / dz), 0), a, dt)

function limit_turb_entrainment(dyn_entr::FT, turb_entr, w, dz) where {FT}
    return max(min((FT(0.9) * w / dz) - dyn_entr, turb_entr), 0)
end
