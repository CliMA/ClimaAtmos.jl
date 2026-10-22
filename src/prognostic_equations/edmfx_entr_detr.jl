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
@inline function detr_buoy_inv_time_scale(Δwʲ, Δbuoyʲ, entr_detr_buoy_inv_tau_max)
    FT = typeof(Δwʲ)
    return min(
        entr_detr_buoy_inv_tau_max,
        abs(min(Δbuoyʲ, FT(0))) / max(eps(FT), abs(Δwʲ)),
    )
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

    area_limiter_factor = upper_area_limiter_factor(ᶜaʲ, turbconv_params)
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
function compute_detrainment(
    turbconv_params,
    aʲ,
    ρaʲ,
    buoy_inv_time_scale,
    massflux_vert_div,
    area_bounding_entr_detr,
    detr_model,
)

    detr = detrainment_rate(
        turbconv_params,
        aʲ,
        ρaʲ,
        buoy_inv_time_scale,
        massflux_vert_div,
        detr_model,
    )

    return detr + max(zero(area_bounding_entr_detr), -area_bounding_entr_detr)
end

"""
    detrainment_rate(turbconv_params, ᶜaʲ, ᶜρaʲ, ᶜbuoy_inv_time_scale,
                     ᶜmassflux_vert_div, detr_model::AbstractDetrainmentModel)
    detrainment_rate(turbconv_params, ᶜaʲ, ᶜρaʲ, ᶜbuoy_inv_time_scale,
                     ᶜmassflux_vert_div, detr_model::BuoyancyVelocityDetrainment)

Model-specific detrainment rate [1/s] for a given detrainment model.

The `AbstractDetrainmentModel` fallback returns zero. For
`BuoyancyVelocityDetrainment` the rate is

    detr = max(0, lower_area_limiter_factor(aʲ) ·
                  (c_buoy τ⁻¹_buoy − c_div min(massflux_vert_div, 0) / ρaʲ)),

with `c_buoy = detr_buoy_coeff` and `c_div = detr_massflux_vertdiv_coeff`. Only
a converging (negative) mass-flux divergence detrains, and
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
)
    FT = eltype(ᶜρaʲ)
    detr_buoy_coeff = CAP.detr_buoy_coeff(turbconv_params)
    detr_massflux_vertdiv_coeff =
        CAP.detr_massflux_vertdiv_coeff(turbconv_params)

    area_limiter_factor = lower_area_limiter_factor(ᶜaʲ, turbconv_params)
    detr =
        area_limiter_factor *
        (
            detr_buoy_coeff * ᶜbuoy_inv_time_scale -
            detr_massflux_vertdiv_coeff * min(ᶜmassflux_vert_div, 0) / max(eps(FT), ᶜρaʲ)
        )

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
    environment_tail_shift(T′T′, q′q′, corr_Tq, α_T, α_q)

Shift `(ΔT, Δq)` of the mean of the upper half `s′ > 0` of the environment's
Gaussian subgrid `(T′, q′)` distribution relative to its mean, for the linear
selector `s′ = α_T T′ + α_q q′`:

    Δx = √(2/π) Cov(x′, s′) / σ_s,
    σ_s² = α_T² σ_T² + 2 α_T α_q r σ_T σ_q + α_q² σ_q²,

with `r` the T–q correlation (the conditional mean of a jointly Gaussian
variable over a half-space). Returns zeros where the subgrid variance vanishes,
so the environment mean is recovered as the variance goes to zero (fine grids).
"""
@inline function environment_tail_shift(T′T′, q′q′, corr_Tq, α_T, α_q)
    FT = typeof(T′T′)
    σ_T = sqrt(max(T′T′, zero(FT)))
    σ_q = sqrt(max(q′q′, zero(FT)))
    r = clamp(corr_Tq, -one(FT), one(FT))
    cov_Ts = α_T * σ_T^2 + α_q * r * σ_T * σ_q
    cov_qs = α_q * σ_q^2 + α_T * r * σ_T * σ_q
    σ_s² = α_T * cov_Ts + α_q * cov_qs
    has_variance = σ_s² > floatmin(FT)
    k = sqrt(FT(2) / FT(π)) / sqrt(max(σ_s², floatmin(FT)))
    return (
        ifelse(has_variance, k * cov_Ts, zero(FT)),
        ifelse(has_variance, k * cov_qs, zero(FT)),
    )
end

"""
    tail_selector_coefficients(thermo_params, selector, T⁰)

Coefficients `(α_T, α_q)` of the tail selector `s′ = α_T T′ + α_q q′`:
`selector ≥ 0.5` selects by moist static energy (`c_p`, `L_v`); otherwise by
virtual buoyancy (`1/T⁰`, `R_v/R_d − 1`, condensate loading neglected).
"""
@inline function tail_selector_coefficients(thermo_params, selector, T⁰)
    FT = typeof(T⁰)
    use_mse = selector >= FT(0.5)
    α_T = ifelse(use_mse, FT(TDP.cp_d(thermo_params)), one(FT) / max(T⁰, one(FT)))
    α_q = ifelse(
        use_mse,
        FT(TDP.LH_v0(thermo_params)),
        FT(TDP.R_v(thermo_params)) / FT(TDP.R_d(thermo_params)) - one(FT),
    )
    return (α_T, α_q)
end

"""
    tail_entrainment_increments(
        thermo_params, w, selector, T⁰, q_tot⁰, q_liq⁰, q_ice⁰,
        T′T′, q′q′, corr_Tq, ρa⁰, ρaʲ, entr, dt,
    )

Increments `(Δmse, Δq_tot)` of the air drawn in by the dynamical entrainment
`entr` [1/s], relative to the environment mean, when a fraction `w` of it comes
from the upper half of the environment's subgrid distribution
(`environment_tail_shift`, selector from `tail_selector_coefficients`).

Both increments come from the same tail state `(T⁰ + w ΔT, q_tot⁰ + w Δq)`, and
the moist static energy uses the model's enthalpy with the environment's
condensate, so the entrained air is one realizable parcel. Realizability: the
mass entrained in one step, `entr ρaʲ dt`, cannot exceed the tail's mass
`ρa⁰ / 2`; beyond that the weight is reduced. Because the environment is
diagnosed as grid mean minus updrafts, the environment gives up exactly what the
updraft entrains and total water and energy are unchanged. The environment's
diagnosed subgrid variance does not account for its tail being removed
preferentially (second order at small updraft area).
"""
@inline function tail_entrainment_increments(
    thermo_params, w, selector, T⁰, q_tot⁰, q_liq⁰, q_ice⁰,
    T′T′, q′q′, corr_Tq, ρa⁰, ρaʲ, entr, dt,
)
    FT = typeof(T⁰)
    α_T, α_q = tail_selector_coefficients(thermo_params, selector, T⁰)
    ΔT, Δq = environment_tail_shift(T′T′, q′q′, corr_Tq, α_T, α_q)
    entrained_per_step = max(entr, zero(FT)) * max(ρaʲ, zero(FT)) * FT(dt)
    w_eff =
        FT(w) * min(
            one(FT),
            FT(0.5) * max(ρa⁰, zero(FT)) / max(entrained_per_step, floatmin(FT)),
        )
    q_tot_e = max(q_tot⁰ + w_eff * Δq, zero(FT))
    T_e = T⁰ + w_eff * ΔT
    Δmse =
        TD.enthalpy(thermo_params, T_e, q_tot_e, q_liq⁰, q_ice⁰) -
        TD.enthalpy(thermo_params, T⁰, q_tot⁰, q_liq⁰, q_ice⁰)
    return (Δmse, q_tot_e - q_tot⁰)
end
@inline tail_entrainment_Δmse(args...) = tail_entrainment_increments(args...)[1]
@inline tail_entrainment_Δq_tot(args...) = tail_entrainment_increments(args...)[2]

"""
    satinst_source_rate(c_s, N²_sat, ρa⁰, f⁰, aʲ, turbconv_params, dt)

Volumetric mass source [kg/m³/s] of the updraft from the saturated part of the
environment where that part is unstable to moist-adiabatic displacement,

    S = c_s ρa⁰ f⁰ √max(0, −N²_sat) · Λ(aʲ),

with `ρa⁰` the environment's effective density, `f⁰` its cloud fraction,
`N²_sat` the saturated buoyancy gradient of the grid mean (the growth rate of
the moist instability is `√(−N²_sat)`), and `Λ` the upper area limiter
(`upper_area_limiter_factor`). Saturated environment air in a moist-unstable
layer would accelerate away from the environment as soon as it is displaced;
the source moves it into the updraft before the resolved flow releases the
instability. The mass moved in one step is capped at half of the environment's
saturated part. The source is a pairwise exchange: the environment, diagnosed as
grid mean minus updrafts, loses what the updraft gains.
"""
@inline function satinst_source_rate(c_s, N²_sat, ρa⁰, f⁰, aʲ, turbconv_params, dt)
    FT = typeof(N²_sat)
    ρa⁰_cld = max(ρa⁰, zero(FT)) * clamp(f⁰, zero(FT), one(FT))
    S = FT(c_s) * ρa⁰_cld * sqrt(max(-N²_sat, zero(FT))) *
        upper_area_limiter_factor(aʲ, turbconv_params)
    return min(S, FT(0.5) * ρa⁰_cld / FT(dt))
end

"""
    satinst_source_increments(thermo_params, T⁰, ρ⁰, q_tot⁰, q_liq⁰, q_ice⁰, T′T′, q′q′, corr_Tq, f⁰, selector)

Increments `(Δmse, Δq_tot)`, relative to the environment mean, of the air moved
by the saturated-instability source: the mean of the top fraction `f⁰` of the
environment's Gaussian subgrid `(T, q_tot)` distribution in the selector `s′`.
For `selector < 0` this is the saturation excess `s′ = q′ − (∂q_sat/∂T) T′`, so
the moved air is the saturated part; otherwise the selector of
`tail_selector_coefficients` (`0` virtual buoyancy, `1` moist static energy), so
the moved air is the equally large part that is most buoyant or has the highest
moist static energy. For a
Gaussian the conditional mean of the top fraction `f` is
`D(f) Cov(x′, s′)/σ_s` with `D(f) = φ(Φ⁻¹(1 − f))/f` (the factor of the surface
buoyant-air value, `percentile_bounds_mean_norm`). The increments vanish with
the subgrid variance and as `f⁰ → 1`.
"""
@inline function satinst_source_increments(
    thermo_params, T⁰, ρ⁰, q_tot⁰, q_liq⁰, q_ice⁰, T′T′, q′q′, corr_Tq, f⁰, selector,
)
    FT = typeof(T⁰)
    L_v = FT(TDP.LH_v0(thermo_params))
    R_v = FT(TDP.R_v(thermo_params))
    q_sat = TD.q_vap_saturation(thermo_params, T⁰, ρ⁰, q_liq⁰, q_ice⁰)
    ∂qsat∂T = q_sat * L_v / (R_v * T⁰^2)
    α_T_sel, α_q_sel = tail_selector_coefficients(thermo_params, selector, T⁰)
    use_saturation = selector < 0
    α_T = ifelse(use_saturation, -∂qsat∂T, α_T_sel)
    α_q = ifelse(use_saturation, one(FT), α_q_sel)
    ΔT½, Δq½ = environment_tail_shift(T′T′, q′q′, corr_Tq, α_T, α_q)
    f = clamp(f⁰, FT(0.01), one(FT) - FT(1e-4))
    D = percentile_bounds_mean_norm(one(FT) - f, one(FT)) / sqrt(FT(2) / FT(π))
    q_tot_e = max(q_tot⁰ + D * Δq½, zero(FT))
    T_e = T⁰ + D * ΔT½
    Δmse =
        TD.enthalpy(thermo_params, T_e, q_tot_e, q_liq⁰, q_ice⁰) -
        TD.enthalpy(thermo_params, T⁰, q_tot⁰, q_liq⁰, q_ice⁰)
    return (Δmse, q_tot_e - q_tot⁰)
end

"""
    set_edmf_satinst_source!(Y, p)

Fill `p.precomputed.ᶜsatinst_source` [kg/m³/s] (`satinst_source_rate`) and the
increments `ᶜsatinst_Δmse`, `ᶜsatinst_Δq_tot` of the moved air
(`satinst_source_increments`) for the single updraft, and its vertical-velocity
excess over the environment mean `ᶜsatinst_Δw = r D(f⁰) √(2 tke/3)` [m/s]: the
saturated top fraction is also the rising part of the environment, to the degree
`r = edmf_satinst_w_corr` that velocity and saturation are correlated. Requires the cloud
fraction and the buoyancy-gradient coefficients of the current state, so it is
called after `set_covariance_cache_and_cloud_fraction!`. No-op (fields stay
zero) unless `edmf_satinst_source_coeff > 0`.
"""
function set_edmf_satinst_source!(Y, p)
    p.atmos.turbconv_model isa PrognosticEDMFX || return nothing
    FT = eltype(p.params)
    c_s = FT(CAP.edmf_satinst_source_coeff(p.params))
    c_s > 0 || return nothing
    hasproperty(p.precomputed, :ᶜT′T′) || error("edmf_satinst_source_coeff > 0 needs the SGS quadrature")
    n_mass_flux_subdomains(p.atmos.turbconv_model) == 1 ||
        error("edmf_satinst_source_coeff > 0 is implemented for one updraft")
    thermo_params = CAP.thermodynamics_params(p.params)
    turbconv_params = CAP.turbconv_params(p.params)
    turbconv_model = p.atmos.turbconv_model
    (; dt) = p
    (; ᶜsatinst_source, ᶜsatinst_Δmse, ᶜsatinst_Δq_tot, ᶜsatinst_Δw) = p.precomputed
    r_w = FT(CAP.edmf_satinst_w_corr(p.params))
    selector = FT(CAP.edmf_satinst_selector(p.params))
    (; ᶜbg_coeffs, ᶜgradᵥ_θ_liq_ice, ᶜgradᵥ_q_tot, ᶜcloud_fraction, ᶜρʲs) = p.precomputed
    (; ᶜT⁰, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰, ᶜT′T′, ᶜq′q′, ᶜcorr_Tq, ᶜp) = p.precomputed
    ᶜlg = Fields.local_geometry_field(Y.c)
    ᶜρ⁰ = @. lazy(TD.air_density(thermo_params, ᶜT⁰, ᶜp, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰))
    ᶜρa⁰ = @. lazy(ρa⁰(Y.c.ρ, Y.c.sgsʲs, turbconv_model))
    ᶜaʲ = @. lazy(draft_area(Y.c.sgsʲs.:(1).ρa, ᶜρʲs.:(1)))
    ᶜq_liqʲ, ᶜq_iceʲ = _updraft_cloud_condensate(Y, p, 1, p.atmos.microphysics_model)
    # Environment cloud fraction: the grid-mean cloud fraction minus the
    # (binary) updraft cloud cover, per unit environment area.
    ᶜf⁰ = @. lazy(clamp(
        (ᶜcloud_fraction - ifelse(
            TD.has_condensate(thermo_params, max(0, ᶜq_liqʲ + ᶜq_iceʲ)), ᶜaʲ, zero(ᶜaʲ),
        )) / max(1 - ᶜaʲ, eps(FT)),
        zero(FT), one(FT),
    ))
    ᶜN²_sat = @. lazy(blended_N²(
        ᶜbg_coeffs, one(FT),
        projected_vector_data(C3, ᶜgradᵥ_θ_liq_ice, ᶜlg),
        projected_vector_data(C3, ᶜgradᵥ_q_tot, ᶜlg),
    ))
    @. ᶜsatinst_source = satinst_source_rate(c_s, ᶜN²_sat, ᶜρa⁰, ᶜf⁰, ᶜaʲ, turbconv_params, dt)
    @. ᶜsatinst_Δmse = first(satinst_source_increments(
        thermo_params, ᶜT⁰, ᶜρ⁰, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰, ᶜT′T′, ᶜq′q′, ᶜcorr_Tq, ᶜf⁰, selector,
    ))
    @. ᶜsatinst_Δq_tot = last(satinst_source_increments(
        thermo_params, ᶜT⁰, ᶜρ⁰, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰, ᶜT′T′, ᶜq′q′, ᶜcorr_Tq, ᶜf⁰, selector,
    ))
    @. ᶜsatinst_Δw =
        r_w * percentile_bounds_mean_norm(1 - clamp(ᶜf⁰, FT(0.01), 1 - FT(1e-4)), one(FT)) *
        sqrt(2 * max(specific(Y.c.ρtke, Y.c.ρ), 0) / 3)
    return nothing
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
edmfx_entr_detr_tendency!(Yₜ, Y, p, t, turbconv_model) = nothing

function edmfx_entr_detr_tendency!(Yₜ, Y, p, t, turbconv_model::PrognosticEDMFX)

    n = n_mass_flux_subdomains(turbconv_model)
    (;
        ᶜturb_entrʲs,
        ᶜentr_vel_scaleʲs,
        ᶜentr_nonvel_rateʲs,
        ᶜarea_bounding_entr_detrʲs,
        ᶜuʲs,
    ) = p.precomputed

    ᶜmse⁰ = ᶜspecific_env_mse(Y, p)
    ᶜq_tot⁰ = ᶜspecific_env_value(@name(q_tot), Y, p)
    ᶜlg = Fields.local_geometry_field(Y.c)
    FT = eltype(p.params)
    w_tail = FT(CAP.edmf_tail_entr_weight(p.params))
    use_tail_entr = w_tail > 0 && hasproperty(p.precomputed, :ᶜT′T′)
    if use_tail_entr
        thermo_params = CAP.thermodynamics_params(p.params)
        tail_selector = FT(CAP.edmf_tail_entr_selector(p.params))
        (; dt) = p
        (; ᶜT⁰, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰, ᶜT′T′, ᶜq′q′, ᶜcorr_Tq) =
            p.precomputed
    end

    use_satinst = CAP.edmf_satinst_source_coeff(p.params) > 0
    a_min = FT(CAP.min_area(CAP.turbconv_params(p.params)))

    for j in 1:n
        ᶜentrʲ = @. lazy(
            compute_entrainment(
                ᶜentr_vel_scaleʲs.:($$j),
                ᶜentr_nonvel_rateʲs.:($$j),
                ᶜarea_bounding_entr_detrʲs.:($$j),
                get_physical_w(ᶜuʲs.:($$j), ᶜlg),
            ),
        )
        ᶜturb_entrʲ = ᶜturb_entrʲs.:($j)
        ᶜmseʲ = Y.c.sgsʲs.:($j).mse
        ᶜq_totʲ = Y.c.sgsʲs.:($j).q_tot

        if use_tail_entr
            # Dynamical entrainment draws from the environment's tail
            # (`tail_entrainment_increments`); turbulent entrainment, a two-way
            # exchange outside the ρa equation, keeps the environment mean.
            ᶜρa⁰ = @. lazy(ρa⁰(Y.c.ρ, Y.c.sgsʲs, turbconv_model))
            ᶜρaʲ = Y.c.sgsʲs.:($j).ρa
            @. Yₜ.c.sgsʲs.:($$j).mse +=
                ᶜentrʲ * (
                    ᶜmse⁰ +
                    tail_entrainment_Δmse(
                        thermo_params, w_tail, tail_selector, ᶜT⁰,
                        ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰, ᶜT′T′, ᶜq′q′,
                        ᶜcorr_Tq, ᶜρa⁰, ᶜρaʲ, ᶜentrʲ, dt,
                    ) - ᶜmseʲ
                ) + ᶜturb_entrʲ * (ᶜmse⁰ - ᶜmseʲ)
            @. Yₜ.c.sgsʲs.:($$j).q_tot +=
                ᶜentrʲ * (
                    ᶜq_tot⁰ +
                    tail_entrainment_Δq_tot(
                        thermo_params, w_tail, tail_selector, ᶜT⁰,
                        ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰, ᶜT′T′, ᶜq′q′,
                        ᶜcorr_Tq, ᶜρa⁰, ᶜρaʲ, ᶜentrʲ, dt,
                    ) - ᶜq_totʲ
                ) + ᶜturb_entrʲ * (ᶜq_tot⁰ - ᶜq_totʲ)
        else
            @. Yₜ.c.sgsʲs.:($$j).mse += (ᶜentrʲ .+ ᶜturb_entrʲ) * (ᶜmse⁰ - ᶜmseʲ)

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

        # Saturated-instability mass source (`set_edmf_satinst_source!`): the
        # moved air carries the saturated part's mse and q_tot and the
        # environment-mean tracers; its mass enters the implicit ρa solve.
        if use_satinst
            (; ᶜsatinst_source, ᶜsatinst_Δmse, ᶜsatinst_Δq_tot, ᶜρʲs) = p.precomputed
            ᶜrateʲ = @. lazy(satinst_specific_rate(
                ᶜsatinst_source, Y.c.sgsʲs.:($$j).ρa, ᶜρʲs.:($$j), a_min,
            ))
            @. Yₜ.c.sgsʲs.:($$j).mse += ᶜrateʲ * (ᶜmse⁰ + ᶜsatinst_Δmse - ᶜmseʲ)
            @. Yₜ.c.sgsʲs.:($$j).q_tot += ᶜrateʲ * (ᶜq_tot⁰ + ᶜsatinst_Δq_tot - ᶜq_totʲ)
            for χ_name in sgs_tracer_names(Y)
                ᶜχ⁰ = ᶜspecific_env_value(χ_name, Y, p)
                ᶜχʲ = MatrixFields.get_field(Y.c.sgsʲs.:($j), χ_name)
                ᶜχʲₜ = MatrixFields.get_field(Yₜ.c.sgsʲs.:($j), χ_name)
                @. ᶜχʲₜ += ᶜrateʲ * (ᶜχ⁰ - ᶜχʲ)
            end
        end
    end
    return nothing
end

"""
    satinst_specific_rate(S, ρaʲ, ρʲ, a_min)

Specific relaxation rate [1/s] of the updraft's intensive variables toward the
air moved by the volumetric source `S` [kg/m³/s]: `S / max(ρaʲ, ρʲ a_min)`
(the same floor as the surface mass-flux source).
"""
@inline satinst_specific_rate(S, ρaʲ, ρʲ, a_min) = S / max(ρaʲ, ρʲ * a_min)

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
