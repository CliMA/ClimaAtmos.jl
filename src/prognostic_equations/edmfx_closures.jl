#####
##### EDMF entrainment detrainment
#####

import StaticArrays as SA
import ClimaCore.Geometry as Geometry
import ClimaCore.Fields as Fields

"""
    Return buoyancy on cell centers.
"""
function ᶜphysical_buoyancy(params, ᶜρ_ref::FT, ᶜρ::FT) where {FT}
    # TODO - replace by ᶜgradᵥᶠΦ when we move to deep atmosphere
    g = CAP.grav(params)
    return (ᶜρ_ref - ᶜρ) / ᶜρ * g
end
"""
    Return buoyancy on cell faces.
"""
function ᶠbuoyancy(ᶠρ_ref, ᶠρ, ᶠgradᵥ_ᶜΦ)
    return (ᶠρ_ref - ᶠρ) / ᶠρ * ᶠgradᵥ_ᶜΦ
end

"""
   Return the nonhydrostatic pressure drag for updrafts [m/s2 * m]

   Inputs (everything defined on cell faces):
   - params - set with model parameters
   - nh_presssure_flag - bool flag for if we want/don't want to compute
                         pressure drag
   - ᶠlg - local geometry (needed to compute the norm inside a local function)
   - ᶠbuoyʲ - covariant3 or contravariant3 updraft buoyancy
   - ᶠu3ʲ, ᶠu3⁰ - covariant3 or contravariant3 velocity for updraft and environment.
                  covariant3 velocity is used in prognostic edmf, and contravariant3
                  velocity is used in diagnostic edmf.
   - updraft top height
"""
function ᶠupdraft_nh_pressure(
    params,
    nh_pressure_flag,
    ᶠlg,
    ᶠbuoyʲ,
    ᶠu3ʲ,
    ᶠu3⁰,
    updraft_top,
)

    if !nh_pressure_flag
        return zero(ᶠu3ʲ)
    else
        turbconv_params = CAP.turbconv_params(params)
        # factor multiplier for pressure buoyancy terms (effective buoyancy is (1-α_b))
        α_b = TCP.pressure_normalmode_buoy_coeff1(turbconv_params)
        # factor multiplier for pressure drag
        α_d = TCP.pressure_normalmode_drag_coeff(turbconv_params)

        # Independence of aspect ratio hardcoded: α₂_asp_ratio² = FT(0)

        H_up_min = TCP.min_updraft_top(turbconv_params)
        plume_scale_height = max(updraft_top, H_up_min)

        # We also used to have advection term here: α_a * w_up * div_w_up
        return α_b * ᶠbuoyʲ +
               α_d * (ᶠu3ʲ - ᶠu3⁰) * CC.Geometry._norm(ᶠu3ʲ - ᶠu3⁰, ᶠlg) /
               plume_scale_height
    end
end

edmfx_nh_pressure_cache(Y, turbconv_model) = (;)

edmfx_nh_pressure_tendency!(Yₜ, Y, p, t, colidx, turbconv_model) = nothing
function edmfx_nh_pressure_tendency!(Yₜ, Y, p, t, colidx, turbconv_model::EDMFX)

    n = n_mass_flux_subdomains(turbconv_model)
    (; params, ᶜρʲs, ᶜρ_ref, ᶠgradᵥ_ᶜΦ, ᶜuʲs, ᶜu⁰, ᶠu₃⁰) = p
    FT = eltype(Y)
    ᶜz = Fields.coordinate_field(Y.c).z
    ᶠlg = Fields.local_geometry_field(Y.f)

    turbconv_params = CAP.turbconv_params(params)
    a_min = TCP.min_area(turbconv_params)

    for j in 1:n

        # look for updraft top
        updraft_top = FT(0)
        for level in 1:Spaces.nlevels(axes(ᶜz))
            if Spaces.level(Y.c.sgsʲs.:($j).ρa[colidx], level)[] /
               Spaces.level(ᶜρʲs.:($j)[colidx], level)[] > a_min
                updraft_top = Spaces.level(ᶜz[colidx], level)[]
            end
        end

        @. Yₜ.f.sgsʲs.:($$j).u₃[colidx] -= ᶠupdraft_nh_pressure(
            params,
            p.atmos.edmfx_nh_pressure,
            ᶠlg[colidx],
            ᶠbuoyancy(
                ᶠinterp(ᶜρ_ref[colidx]),
                ᶠinterp(ᶜρʲs.:($$j)[colidx]),
                ᶠgradᵥ_ᶜΦ[colidx],
            ),
            Y.f.sgsʲs.:($$j).u₃[colidx],
            ᶠu₃⁰[colidx],
            updraft_top,
        )
    end
end

"""
   Return entrainment and detrainment rates [1/s].

   We want the rates to be based on linear combination of pi-groups.
   But for now:
    - entr = const * w / z
    - detr = const * w

   Inputs (everything defined on cell centers):
   - params set with model parameters
   - entr_detr_flag - bool flag for if we want/don't want to compute
                      entrainment and detrainment
   - ᶜz, ᶜp, ᶜρ, - grid mean height, pressure and density
   - buoy_flux_surface - buoyancy flux at the surface
   - ᶜaʲ, ᶜwʲ, ᶜRHʲ, ᶜbuoyʲ - updraft area, physical vertical velocity,
                                   relative humidity and buoyancy
   - ᶜw⁰, ᶜRH⁰, ᶜbuoy⁰ - environment physical vertical velocity,
                              relative humidity and buoyancy
"""
function pi_groups_entr_detr(
    params,
    entr_detr_flag::Bool,
    ᶜz::FT,
    ᶜp::FT,
    ᶜρ::FT,
    buoy_flux_surface::FT,
    ᶜaʲ::FT,
    ᶜwʲ::FT,
    ᶜRHʲ::FT,
    ᶜbuoyʲ::FT,
    ᶜw⁰::FT,
    ᶜRH⁰::FT,
    ᶜbuoy⁰::FT,
    dt::FT,
) where {FT}

    if ᶜaʲ <= FT(0) || !entr_detr_flag
        return (; entr = FT(0), detr = FT(0))
    else
        g = CAP.grav(params)

        entr_coeff = CAP.entr_coeff(params)
        detr_coeff = CAP.detr_coeff(params)

        turbconv_params = CAP.turbconv_params(params)
        ᶜaʲ_max = TCP.max_area(turbconv_params)
        max_area_limiter = 0.1 * exp(-10 * (ᶜaʲ_max - ᶜaʲ))

        # pressure scale height (height where pressure drops by 1/e)
        ref_H = ᶜp / (ᶜρ * g)
        # convective velocity
        w_star = get_wstar(buoy_flux_surface)

        # non-dimensional pi-groups
        # TODO - using Π₁ blows things up
        Π₁ = ᶜz * (ᶜbuoyʲ - ᶜbuoy⁰) / ((ᶜwʲ - ᶜw⁰)^2 + w_star^2 + eps(FT))
        Π₃ = sqrt(ᶜaʲ)
        Π₄ = ᶜRHʲ - ᶜRH⁰
        Π₆ = ᶜz / ref_H
        entr = max(
            0,
            ᶜwʲ / ᶜz * (
                -4.013288 - 0.000968 * Π₁ + 0.356974 * Π₃ - 0.403124 * Π₄ +
                1.503261 * Π₆
            ),
        )
        detr = max(
            0,
            ᶜwʲ * (
                3.535208 + 0.598496 * Π₁ + 1.583348 * Π₃ + 0.046275 * Π₄ -
                0.344836 * Π₆ + max_area_limiter
            ),
        )

        # TODO - Temporary: Switch to Π groups after simple tests are done
        # (kinematic, bubble, Bomex)
        # and/or we can calibrate things in ClimaAtmos
        entr = max(0, min(entr_coeff * ᶜwʲ / ᶜz, 1 / dt))
        detr = max(0, min(detr_coeff * ᶜwʲ, 1 / dt))

        return (; entr, detr)
    end
end

edmfx_entr_detr_tendency!(Yₜ, Y, p, t, colidx, turbconv_model) = nothing
function edmfx_entr_detr_tendency!(Yₜ, Y, p, t, colidx, turbconv_model::EDMFX)

    n = n_mass_flux_subdomains(turbconv_model)
    (; ᶜspecificʲs, ᶜh_totʲs, ᶜentr_detrʲs) = p
    (; ᶜu⁰, ᶜspecific⁰, ᶜh_tot⁰) = p

    for j in 1:n

        @. Yₜ.c.sgsʲs.:($$j).ρa[colidx] +=
            Y.c.sgsʲs.:($$j).ρa[colidx] * (
                ᶜentr_detrʲs.:($$j).entr[colidx] -
                ᶜentr_detrʲs.:($$j).detr[colidx]
            )

        @. Yₜ.c.sgsʲs.:($$j).ρae_tot[colidx] +=
            Y.c.sgsʲs.:($$j).ρa[colidx] * (
                ᶜentr_detrʲs.:($$j).entr[colidx] * ᶜh_tot⁰[colidx] -
                ᶜentr_detrʲs.:($$j).detr[colidx] * ᶜh_totʲs.:($$j)[colidx]
            )

        @. Yₜ.c.sgsʲs.:($$j).ρaq_tot[colidx] +=
            Y.c.sgsʲs.:($$j).ρa[colidx] * (
                ᶜspecific⁰.q_tot[colidx] * ᶜentr_detrʲs.:($$j).entr[colidx] -
                ᶜspecificʲs.:($$j).q_tot[colidx] *
                ᶜentr_detrʲs.:($$j).detr[colidx]
            )

        @. Yₜ.f.sgsʲs.:($$j).u₃[colidx] +=
            ᶠinterp(ᶜentr_detrʲs.:($$j).entr[colidx] * C3(ᶜu⁰[colidx])) -
            ᶠinterp(ᶜentr_detrʲs.:($$j).entr[colidx]) *
            Y.f.sgsʲs.:($$j).u₃[colidx]
    end
    return nothing
end

# lambert_2_over_e(::Type{FT}) where {FT} = FT(LambertW.lambertw(FT(2) / FT(MathConstants.e)))
lambert_2_over_e(::Type{FT}) where {FT} = FT(0.46305551336554884) # since we can evaluate

function lamb_smooth_minimum(
    l::SA.SVector,
    lower_bound::FT,
    upper_bound::FT,
) where {FT}
    x_min = minimum(l)
    λ_0 = max(x_min * lower_bound / lambert_2_over_e(FT), upper_bound)

    num = sum(l_i -> l_i * exp(-(l_i - x_min) / λ_0), l)
    den = sum(l_i -> exp(-(l_i - x_min) / λ_0), l)
    smin = num / den
    return smin
end

"""
    mixing_length(params, ustar, ᶜz, sfc_tke, ᶜlinear_buoygrad, ᶜtke, obukhov_length, shear², ᶜPr, ᶜtke_exch)

where:
- `params`: set with model parameters
- `ustar`: friction velocity
- `ᶜz`: height
- `tke_sfc`: env kinetic energy at first cell center
- `ᶜlinear_buoygrad`: buoyancy gradient
- `ᶜtke`: env turbulent kinetic energy
- `obukhov_length`: surface Monin Obukhov length
- `ᶜshear²`: shear term
- `ᶜPr`: Prandtl number
- `ᶜtke_exch`: subdomain exchange term

Returns mixing length as a smooth minimum between
wall-constrained length scale,
production-dissipation balanced length scale,
effective static stability length scale, and
Smagorinsky length scale.
"""
function mixing_length(
    params,
    ustar::FT,
    ᶜz::FT,
    ᶜdz::FT,
    sfc_tke::FT,
    ᶜlinear_buoygrad::FT,
    ᶜtke::FT,
    obukhov_length::FT,
    ᶜshear²::FT,
    ᶜPr::FT,
    ᶜtke_exch::FT,
) where {FT}

    turbconv_params = CAP.turbconv_params(params)
    c_m = TCP.tke_ed_coeff(turbconv_params)
    c_d = TCP.tke_diss_coeff(turbconv_params)
    smin_ub = TCP.smin_ub(turbconv_params)
    smin_rm = TCP.smin_rm(turbconv_params)
    l_max = TCP.l_max(turbconv_params)
    c_b = TCP.static_stab_coeff(turbconv_params)
    vkc = TCP.von_karman_const(turbconv_params)

    # compute the l_W - the wall constraint mixing length
    # which imposes an upper limit on the size of eddies near the surface
    # kz scale (surface layer)
    if obukhov_length < 0.0 #unstable
        l_W =
            vkc * ᶜz / (sqrt(sfc_tke / ustar / ustar) * c_m) *
            min((1 - 100 * ᶜz / obukhov_length)^FT(0.2), 1 / vkc)
    else # neutral or stable
        l_W = vkc * ᶜz / (sqrt(sfc_tke / ustar / ustar) * c_m)
    end

    # compute l_TKE - the production-dissipation balanced length scale
    a_pd = c_m * (ᶜshear² - ᶜlinear_buoygrad / ᶜPr) * sqrt(ᶜtke)
    # Dissipation term
    c_neg = c_d * ᶜtke * sqrt(ᶜtke)
    if abs(a_pd) > eps(FT) && 4 * a_pd * c_neg > -(ᶜtke_exch * ᶜtke_exch)
        l_TKE = max(
            -(ᶜtke_exch / 2 / a_pd) +
            sqrt(ᶜtke_exch * ᶜtke_exch + 4 * a_pd * c_neg) / 2 / a_pd,
            0,
        )
    elseif abs(a_pd) < eps(FT) && abs(ᶜtke_exch) > eps(FT)
        l_TKE = c_neg / ᶜtke_exch
    else
        l_TKE = FT(0)
    end

    # compute l_N - the effective static stability length scale.
    N_eff = sqrt(max(ᶜlinear_buoygrad, 0))
    if N_eff > 0.0
        l_N = min(sqrt(max(c_b * ᶜtke, 0)) / N_eff, l_max)
    else
        l_N = l_max
    end

    # compute l_smag - the Smagorinsky length scale.
    # TODO: This should be added to ClimaParameters
    c_smag = FT(0.2)
    N_eff = sqrt(max(ᶜlinear_buoygrad, 0))
    if N_eff > 0.0
        l_smag = c_smag * ᶜdz * max(0, 1 - N_eff^2 / ᶜPr / ᶜshear²)^(1 / 4)
    else
        l_smag = c_smag * ᶜdz
    end

    # add limiters
    l = SA.SVector(
        (l_N < eps(FT) || l_N > l_max) ? l_max : l_N,
        (l_TKE < eps(FT) || l_TKE > l_max) ? l_max : l_TKE,
        (l_W < eps(FT) || l_W > l_max) ? l_max : l_W,
    )
    # get soft minimum
    # TODO: limit it with l_smag
    return lamb_smooth_minimum(l, smin_ub, smin_rm)
end

"""
    turbulent_prandtl_number(params, obukhov_length, ᶜRi_grad)

where:
- `params`: set with model parameters
- `obukhov_length`: surface Monin Obukhov length
- `ᶜRi_grad`: gradient Richardson number

Returns the turbulent Prandtl number give the obukhov length sign and
the gradient Richardson number, which is calculated from the linearized
buoyancy gradient and shear production.
"""
function turbulent_prandtl_number(
    params,
    obukhov_length::FT,
    ᶜlinear_buoygrad::FT,
    ᶜshear²::FT,
) where {FT}
    turbconv_params = CAP.turbconv_params(params)
    Ri_c = TCP.Ri_crit(turbconv_params)
    ω_pr = TCP.Prandtl_number_scale(turbconv_params)
    Pr_n = TCP.Prandtl_number_0(turbconv_params)
    ᶜRi_grad = min(ᶜlinear_buoygrad / max(ᶜshear², eps(FT)), Ri_c)
    if obukhov_length > 0 && ᶜRi_grad > 0 #stable
        # CSB (Dan Li, 2019, eq. 75), where ω_pr = ω_1 + 1 = 53.0 / 13.0
        prandtl_nvec =
            Pr_n * (
                2 * ᶜRi_grad / (
                    1 + ω_pr * ᶜRi_grad -
                    sqrt((1 + ω_pr * ᶜRi_grad)^2 - 4 * ᶜRi_grad)
                )
            )
    else
        prandtl_nvec = Pr_n
    end
    return prandtl_nvec
end
