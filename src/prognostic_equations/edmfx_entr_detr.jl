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
    entrainment(
        thermo_params, turbconv_params, ᶜz, z_sfc, ᶜp, ᶜρ,
        ᶜaʲ, ᶜwʲ, ᶜRHʲ, ᶜbuoyʲ, ᶜw⁰, ᶜRH⁰, ᶜbuoy⁰, ᶜtke,
        model_option::AbstractEntrainmentModel,
    )

Calculates the entrainment rate [1/s] based on the specified `model_option`.

The specific formulation for entrainment depends on the concrete type passed as
`model_option`. This argument dispatches to different entrainment models,
such as `NoEntrainment` (which returns a zero rate), `PiGroupsEntrainment`
(which uses an entrainment rate based on a linear combination of scaled absolute
values of non-dimensional Pi-groups and a constant term), or `InvZEntrainment` (a "physically
inspired" formulation with a term proportional to `1/z`). Each model implements a distinct 
physical or empirical approach to quantify the entrainment process in updrafts.

Arguments (all cell-centered):
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
- `model_option`: An object whose type specifies the entrainment model to use
                  (e.g., an instance of `NoEntrainment`, `PiGroupsEntrainment`,
                  or `InvZEntrainment`). This corresponds to the `AbstractEntrainmentModel`
                  in the function signature.

Returns:
- Entrainment rate [1/s], computed according to the model selected by `model_option`.
"""
function entrainment(
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

function entrainment(
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
    entr_inv_tau = CAP.entr_inv_tau(turbconv_params)
    a_min = CAP.min_area(turbconv_params)
    limit_inv_tau = CAP.entr_detr_limit_inv_tau(turbconv_params)
    # Entrainment is not well-defined if updraft area is negligible.
    # Fix at limit_inv_tau to ensure some mixing with the environment.
    if ᶜaʲ <= a_min
        return limit_inv_tau
    end

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
        #entr_param_vec[1] * abs(Π₁) +
        entr_param_vec[1] * abs(Π₂) +
        entr_param_vec[2] * abs(Π₃) +
        #entr_param_vec[4] * abs(Π₄) +
        #entr_param_vec[5] * abs(Π₅) +
        entr_param_vec[3]

    inv_timescale_factor = abs(ᶜwʲ - ᶜw⁰) / elev_above_sfc

    entr_mult_limiter_coeff = CAP.entr_mult_limiter_coeff(turbconv_params)
    area_limiter_factor =
        (FT(1) - min(max(ᶜaʲ, 0), FT(1)))^entr_mult_limiter_coeff # Ensure ᶜaʲ is clipped to [0,1] for exponent

    entr = area_limiter_factor * inv_timescale_factor * pi_sum + entr_inv_tau
    return max(entr, entr_inv_tau) # Ensure non-negative entrainment
end

function entrainment(
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
    entr_mult_limiter_coeff = CAP.entr_mult_limiter_coeff(turbconv_params)
    entr_inv_tau = CAP.entr_inv_tau(turbconv_params)
    entr_coeff = CAP.entr_coeff(turbconv_params)
    min_area_limiter_scale = CAP.min_area_limiter_scale(turbconv_params)
    min_area_limiter_power = CAP.min_area_limiter_power(turbconv_params)
    a_min = CAP.min_area(turbconv_params)
    limit_inv_tau = CAP.entr_detr_limit_inv_tau(turbconv_params)

    # Entrainment is not well-defined if updraft area is negligible,
    # as some limiters depend on ᶜaʲ.
    if ᶜaʲ <= a_min && min_area_limiter_scale == FT(0) # If no area and no base min_area_limiter
        return limit_inv_tau
    end

    min_area_limiter =
        min_area_limiter_scale *
        exp(-min_area_limiter_power * (max(ᶜaʲ, 0) - a_min))

    elev_above_sfc = ᶜz - z_sfc
    # Velocity difference term divided by elev_above_sfc; set to zero if elev_above_sfc is 
    # not positive to prevent division by zero or excessively large values.
    vel_diff_term = if elev_above_sfc > eps(FT)
        entr_coeff * abs(ᶜwʲ - ᶜw⁰) / elev_above_sfc
    else
        FT(0)
    end

    area_limiter_factor =
        (FT(1) - min(max(ᶜaʲ, 0), FT(1)))^entr_mult_limiter_coeff

    entr =
        area_limiter_factor * (entr_inv_tau + vel_diff_term + min_area_limiter)
    return max(entr, 0) # Ensure non-negative
end

"""
    detrainment(
        thermo_params, turbconv_params, ᶜz, z_sfc, ᶜp, ᶜρ, ᶜρaʲ, ᶜaʲ,
        ᶜwʲ, ᶜRHʲ, ᶜbuoyʲ, ᶜw⁰, ᶜRH⁰, ᶜbuoy⁰, ᶜentr, ᶜvert_div,
        ᶜmassflux_vert_div, ᶜw_vert_div, ᶜtke, model_option::AbstractDetrainmentModel
    )

Calculates the detrainment rate [1/s] based on the specified `model_option`.

The specific formulation for detrainment depends on the concrete type passed as
`model_option`. This argument dispatches to different detrainment models,
such as `NoDetrainment` (which returns a zero rate), `PiGroupsDetrainment`
(which uses non-dimensional Pi-groups and mass flux divergence),
`BuoyancyVelocityDetrainment` (a physically inspired formulation, based on `buoyancy/vertical velocity`), 
or `SmoothAreaDetrainment` (based on entrainment and updraft vertical velocity divergence). 
Each model implements a distinct physical or empirical approach to quantify the detrainment
process from updrafts.

Arguments (all cell-centered):
- `thermo_params`: Thermodynamic parameters.
- `turbconv_params`: Turbulence convection parameters.
- `ᶜz`: Height [m].
- `z_sfc`: Surface elevation [m].
- `ᶜp`: Pressure [Pa].
- `ᶜρ`: Air density [kg/m³].
- `ᶜρaʲ`: Updraft effective density (`ρ * a`) [kg/m³].
- `ᶜaʲ`: Updraft area fraction [-].
- `ᶜwʲ`: Updraft physical vertical velocity [m/s].
- `ᶜRHʲ`: Updraft relative humidity [-].
- `ᶜbuoyʲ`: Updraft buoyancy [m/s²].
- `ᶜw⁰`: Environment physical vertical velocity [m/s].
- `ᶜRH⁰`: Environment relative humidity [-].
- `ᶜbuoy⁰`: Environment buoyancy [m/s²].
- `ᶜentr`: Entrainment rate [1/s].
- `ᶜvert_div`: Grid-mean vertical divergence [1/s].
- `ᶜmassflux_vert_div`: Vertical divergence of updraft mass flux [kg/m²/s²].
- `ᶜw_vert_div`: Vertical divergence term related to updraft vertical velocity [1/s].
- `ᶜtke`: Turbulent kinetic energy [m²/s²].
- `model_option`: An object whose type specifies the detrainment model to use
                  (e.g., an instance of `NoDetrainment`, `PiGroupsDetrainment`, 
                  `BuoyancyVelocityDetrainment`, `SmoothAreaDetrainment`, etc.).
                  This corresponds to the `AbstractDetrainmentModel` in the function signature.

Returns:
- Detrainment rate [1/s], computed according to the model selected by `model_option`.
"""
function detrainment(
    thermo_params,
    turbconv_params,
    ᶜz,
    z_sfc,
    ᶜp,
    ᶜρ,
    ᶜρaʲ,
    ᶜaʲ,
    ᶜwʲ,
    ᶜRHʲ,
    ᶜbuoyʲ,
    ᶜw⁰,
    ᶜRH⁰,
    ᶜbuoy⁰,
    ᶜentr,
    ᶜvert_div,
    ᶜmassflux_vert_div,
    ᶜw_vert_div,
    ᶜtke,
    ::NoDetrainment,
)
    return zero(eltype(thermo_params))
end

function detrainment(
    thermo_params,
    turbconv_params,
    ᶜz,
    z_sfc,
    ᶜp,
    ᶜρ,
    ᶜρaʲ,
    ᶜaʲ,
    ᶜwʲ,
    ᶜRHʲ,
    ᶜbuoyʲ,
    ᶜw⁰,
    ᶜRH⁰,
    ᶜbuoy⁰,
    ᶜentr,
    ᶜvert_div,
    ᶜmassflux_vert_div,
    ᶜw_vert_div,
    ᶜtke,
    ::PiGroupsDetrainment,
)
    FT = eltype(thermo_params)
    a_min = CAP.min_area(turbconv_params)
    limit_inv_tau = CAP.entr_detr_limit_inv_tau(turbconv_params)

    # If ᶜaʲ (updraft area fraction) is negligible, detrainment is considered
    # to be fixed at limit_inv_tau. This also protects division by ᶜρaʲ later.
    if ᶜaʲ <= a_min
        return limit_inv_tau
    end

    elev_above_sfc = ᶜz - z_sfc
    # If elevation above surface is not positive, some Pi-group terms
    # might be ill-defined or the model assumptions might not hold.
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

    entr_param_vec = CAP.entr_param_vec(turbconv_params) # Note: Uses indices 7-12 for detrainment
    pi_sum_detr =
        entr_param_vec[7] * abs(Π₁) +
        entr_param_vec[8] * abs(Π₂) +
        entr_param_vec[9] * abs(Π₃) +
        entr_param_vec[10] * abs(Π₄) +
        entr_param_vec[11] * abs(Π₅) +
        entr_param_vec[12]

    # Detrainment proportional to negative mass flux divergence
    detr_factor_mass_flux_div = -min(ᶜmassflux_vert_div, FT(0)) / ᶜρaʲ
    detr = detr_factor_mass_flux_div * pi_sum_detr

    return max(detr, 0) # Ensure non-negative detrainment
end

function detrainment(
    thermo_params,
    turbconv_params,
    ᶜz,
    z_sfc,
    ᶜp,
    ᶜρ,
    ᶜρaʲ,
    ᶜaʲ,
    ᶜwʲ,
    ᶜRHʲ,
    ᶜbuoyʲ,
    ᶜw⁰,
    ᶜRH⁰,
    ᶜbuoy⁰,
    ᶜentr,
    ᶜvert_div,
    ᶜmassflux_vert_div,
    ᶜw_vert_div,
    ᶜtke,
    ::BuoyancyVelocityDetrainment,
)
    FT = eltype(thermo_params)
    detr_inv_tau = CAP.detr_inv_tau(turbconv_params)
    detr_coeff = CAP.detr_coeff(turbconv_params)
    detr_buoy_coeff = CAP.detr_buoy_coeff(turbconv_params)
    detr_vertdiv_coeff = CAP.detr_vertdiv_coeff(turbconv_params)
    detr_massflux_vertdiv_coeff =
        CAP.detr_massflux_vertdiv_coeff(turbconv_params)
    max_area_limiter_scale = CAP.max_area_limiter_scale(turbconv_params)
    max_area_limiter_power = CAP.max_area_limiter_power(turbconv_params)
    a_min = CAP.min_area(turbconv_params)
    a_max = CAP.max_area(turbconv_params)
    limit_inv_tau = CAP.entr_detr_limit_inv_tau(turbconv_params)

    # If ᶜaʲ (updraft area fraction) is negligible, detrainment is not well defined.
    # Fix at limit_inv_tau to ensure some mixing with the environment.
    if ᶜaʲ <= a_min
        return limit_inv_tau
    end

    max_area_limiter =
        max_area_limiter_scale *
        exp(-max_area_limiter_power * (a_max - min(ᶜaʲ, 1)))
    detr =
        detr_inv_tau +
        detr_coeff * abs(ᶜwʲ) +
        detr_buoy_coeff * abs(min(ᶜbuoyʲ - ᶜbuoy⁰, 0)) /
        max(eps(FT), abs(ᶜwʲ - ᶜw⁰)) - detr_vertdiv_coeff * ᶜvert_div -
        detr_massflux_vertdiv_coeff * min(ᶜmassflux_vert_div, 0) / ᶜρaʲ + # Protected by ᶜρaʲ check above
        max_area_limiter

    return max(detr, 0)
end

function detrainment(
    thermo_params,
    turbconv_params,
    ᶜz,
    z_sfc,
    ᶜp,
    ᶜρ,
    ᶜρaʲ,
    ᶜaʲ,
    ᶜwʲ,
    ᶜRHʲ,
    ᶜbuoyʲ,
    ᶜw⁰,
    ᶜRH⁰,
    ᶜbuoy⁰,
    ᶜentr,
    ᶜvert_div,
    ᶜmassflux_vert_div,
    ᶜw_vert_div,
    ᶜtke,
    ::SmoothAreaDetrainment,
)
    FT = eltype(thermo_params)
    a_min = CAP.min_area(turbconv_params)
    limit_inv_tau = CAP.entr_detr_limit_inv_tau(turbconv_params)
    # If ᶜaʲ is negligible detrainment is fixed at limit_inv_tau.
    if (ᶜaʲ <= a_min) # Consistent check for ᶜaʲ
        detr = limit_inv_tau
        # If vertical velocity divergence term is non-negative detrainment is zero.
    elseif (ᶜw_vert_div >= 0)
        detr = FT(0)
    else
        detr = ᶜentr - ᶜw_vert_div
    end
    return max(detr, 0)
end

function turbulent_entrainment(turbconv_params, ᶜaʲ)
    turb_entr_param_vec = CAP.turb_entr_param_vec(turbconv_params)
    return max(turb_entr_param_vec[1] * exp(-turb_entr_param_vec[2] * ᶜaʲ), 0)
end

edmfx_entr_detr_tendency!(Yₜ, Y, p, t, turbconv_model) = nothing

function edmfx_entr_detr_tendency!(Yₜ, Y, p, t, turbconv_model::PrognosticEDMFX)

    n = n_mass_flux_subdomains(turbconv_model)
    (; ᶜturb_entrʲs, ᶜentrʲs, ᶜdetrʲs) = p.precomputed
    (; ᶠu₃⁰) = p.precomputed

    ᶜmse⁰ = ᶜspecific_env_mse(Y, p)
    ᶜq_tot⁰ = ᶜspecific_env_value(@name(q_tot), Y, p)

    microphysics_tracers = (
        (@name(c.sgsʲs.:(1).q_liq), @name(q_liq)),
        (@name(c.sgsʲs.:(1).q_ice), @name(q_ice)),
        (@name(c.sgsʲs.:(1).q_rai), @name(q_rai)),
        (@name(c.sgsʲs.:(1).q_sno), @name(q_sno)),
        (@name(c.sgsʲs.:(1).n_liq), @name(n_liq)),
        (@name(c.sgsʲs.:(1).n_rai), @name(n_rai)),
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

        @. Yₜ.f.sgsʲs.:($$j).u₃ +=
            (ᶠinterp(ᶜentrʲ) .+ ᶠinterp(ᶜturb_entrʲ)) *
            (ᶠu₃⁰ - Y.f.sgsʲs.:($$j).u₃)
    end
    return nothing
end

"""
    edmfx_first_interior_entr_tendency!(Yₜ, Y, p, t, turbconv_model::PrognosticEDMFX)

Apply first-interior–level entrainment tendencies for each EDMF updraft.

This routine (1) seeds a small positive updraft area fraction when surface
buoyancy flux is positive—allowing the plume to grow from zero—and  
(2) adds entrainment tendencies for moist static energy (`mse`) and total
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
