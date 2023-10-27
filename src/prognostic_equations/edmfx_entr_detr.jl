#####
##### EDMF entrainment detrainment
#####

# return a harmonic mean of (a, 1-a)
hm_limiter(a) = 2 * a * (1 - a)

"""
   Return entrainment rate [1/s].

   Inputs (everything defined on cell centers):
   - params set with model parameters
   - ᶜz, z_sfc, ᶜp, ᶜρ, - grid-scale height, surface height, grid-scale pressure and density
   - buoy_flux_surface - buoyancy flux at the surface
   - ᶜaʲ, ᶜwʲ, ᶜRHʲ, ᶜbuoyʲ - updraft area, physical vertical velocity,
                                   relative humidity and buoyancy
   - ᶜw⁰, ᶜRH⁰, ᶜbuoy⁰ - environment physical vertical velocity,
                              relative humidity and buoyancy
   - dt - timestep
"""

function entrainment(
    params,
    ᶜz::FT,
    z_sfc::FT,
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
    ::NoEntrainment,
) where {FT}
    return FT(0)
end

function entrainment(
    params,
    ᶜz::FT,
    z_sfc::FT,
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
    ::PiGroupsEntrainment,
) where {FT}
    if ᶜaʲ <= FT(0)
        return FT(0)
    else
        g = CAP.grav(params)

        # pressure scale height (height where pressure drops by 1/e)
        ref_H = ᶜp / (ᶜρ * g)
        # convective velocity
        w_star = get_wstar(buoy_flux_surface)

        # non-dimensional pi-groups
        # TODO - using Π₁ blows things up
        Π₁ =
            (ᶜz - z_sfc) * (ᶜbuoyʲ - ᶜbuoy⁰) /
            ((ᶜwʲ - ᶜw⁰)^2 + w_star^2 + eps(FT))
        Π₃ = sqrt(ᶜaʲ)
        Π₄ = ᶜRHʲ - ᶜRH⁰
        Π₆ = (ᶜz - z_sfc) / ref_H
        entr = max(
            0,
            min(
                abs(ᶜwʲ) / (ᶜz - z_sfc) * (
                    -4.013288 - 0.000968 * Π₁ + 0.356974 * Π₃ - 0.403124 * Π₄ + 1.503261 * Π₆
                ),
                1 / dt,
            ),
        )

        return entr
    end
end

function entrainment(
    params,
    ᶜz::FT,
    z_sfc::FT,
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
    ::GeneralizedEntrainment,
) where {FT}
    turbconv_params = CAP.turbconv_params(params)
    entr_inv_tau = CAP.entr_tau(turbconv_params)
    entr_coeff = CAP.entr_coeff(turbconv_params)
    min_area_limiter_scale = CAP.min_area_limiter_scale(turbconv_params)
    min_area_limiter_power = CAP.min_area_limiter_power(turbconv_params)
    a_min = CAP.min_area(turbconv_params)

    min_area_limiter =
        min_area_limiter_scale *
        exp(-min_area_limiter_power * (max(ᶜaʲ, 0) - a_min))
    entr = min(
        entr_inv_tau + entr_coeff * abs(ᶜwʲ) / (ᶜz - z_sfc) + min_area_limiter,
        1 / dt,
    )
    return entr
end

function entrainment(
    params,
    ᶜz::FT,
    z_sfc::FT,
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
    ::GeneralizedHarmonicsEntrainment,
) where {FT}
    turbconv_params = CAP.turbconv_params(params)
    entr_inv_tau = CAP.entr_tau(turbconv_params)
    entr_coeff = CAP.entr_coeff(turbconv_params)
    min_area_limiter_scale = CAP.min_area_limiter_scale(turbconv_params)
    min_area_limiter_power = CAP.min_area_limiter_power(turbconv_params)
    a_min = CAP.min_area(turbconv_params)

    min_area_limiter =
        min_area_limiter_scale *
        exp(-min_area_limiter_power * (max(ᶜaʲ, 0) - a_min))
    entr = min(
        entr_inv_tau + entr_coeff * abs(ᶜwʲ) / (ᶜz - z_sfc) + min_area_limiter,
        1 / dt,
    )
    return entr * FT(2) * hm_limiter(ᶜaʲ)
end

"""
   Return detrainment rate [1/s].

   Inputs (everything defined on cell centers):
   - params set with model parameters
   - ᶜz, z_sfc, ᶜp, ᶜρ, - grid-scale height, surface height, grid-scale pressure and density
   - buoy_flux_surface - buoyancy flux at the surface
   - ᶜaʲ, ᶜwʲ, ᶜRHʲ, ᶜbuoyʲ - updraft area, physical vertical velocity,
                                   relative humidity and buoyancy
   - ᶜw⁰, ᶜRH⁰, ᶜbuoy⁰ - environment physical vertical velocity,
                              relative humidity and buoyancy
   - dt - timestep
"""
function detrainment(
    params,
    ᶜz::FT,
    z_sfc::FT,
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
    ::NoDetrainment,
) where {FT}
    return FT(0)
end

function detrainment(
    params,
    ᶜz::FT,
    z_sfc::FT,
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
    ::PiGroupsDetrainment,
) where {FT}

    if ᶜaʲ <= FT(0)
        return FT(0)
    else
        g = CAP.grav(params)
        turbconv_params = CAP.turbconv_params(params)
        ᶜaʲ_max = CAP.max_area(turbconv_params)

        max_area_limiter = FT(0.1) * exp(-10 * (ᶜaʲ_max - ᶜaʲ))
        # pressure scale height (height where pressure drops by 1/e)
        ref_H = ᶜp / (ᶜρ * g)
        # convective velocity
        w_star = get_wstar(buoy_flux_surface)

        # non-dimensional pi-groups
        # TODO - using Π₁ blows things up
        Π₁ =
            (ᶜz - z_sfc) * (ᶜbuoyʲ - ᶜbuoy⁰) /
            ((ᶜwʲ - ᶜw⁰)^2 + w_star^2 + eps(FT))
        Π₃ = sqrt(ᶜaʲ)
        Π₄ = ᶜRHʲ - ᶜRH⁰
        Π₆ = (ᶜz - z_sfc) / ref_H
        detr = max(
            0,
            min(
                abs(ᶜwʲ) * (
                    3.535208 + 0.598496 * Π₁ + 1.583348 * Π₃ + 0.046275 * Π₄ - 0.344836 * Π₆ + max_area_limiter
                ),
                1 / dt,
            ),
        )
        return detr
    end
end

function detrainment(
    params,
    ᶜz::FT,
    z_sfc::FT,
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
    ::GeneralizedDetrainment,
) where {FT}
    turbconv_params = CAP.turbconv_params(params)
    detr_inv_tau = CAP.detr_tau(turbconv_params)
    detr_coeff = CAP.detr_coeff(turbconv_params)
    detr_buoy_coeff = CAP.detr_buoy_coeff(turbconv_params)
    max_area_limiter_scale = CAP.max_area_limiter_scale(turbconv_params)
    max_area_limiter_power = CAP.max_area_limiter_power(turbconv_params)
    a_max = CAP.max_area(turbconv_params)

    max_area_limiter =
        max_area_limiter_scale *
        exp(-max_area_limiter_power * (a_max - min(ᶜaʲ, 1)))
    detr = min(
        detr_inv_tau +
        detr_coeff * abs(ᶜwʲ) +
        detr_buoy_coeff * abs(min(ᶜbuoyʲ - ᶜbuoy⁰, 0)) /
        max(eps(FT), abs(ᶜwʲ - ᶜw⁰)) +
        max_area_limiter,
        1 / dt,
    )
    return detr
end

function detrainment(
    params,
    ᶜz::FT,
    z_sfc::FT,
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
    ::GeneralizedHarmonicsDetrainment,
) where {FT}
    turbconv_params = CAP.turbconv_params(params)
    detr_inv_tau = CAP.detr_tau(turbconv_params)
    detr_coeff = CAP.detr_coeff(turbconv_params)
    detr_buoy_coeff = CAP.detr_buoy_coeff(turbconv_params)
    max_area_limiter_scale = CAP.max_area_limiter_scale(turbconv_params)
    max_area_limiter_power = CAP.max_area_limiter_power(turbconv_params)
    a_max = CAP.max_area(turbconv_params)

    max_area_limiter =
        max_area_limiter_scale *
        exp(-max_area_limiter_power * (a_max - min(ᶜaʲ, 1)))
    detr = min(
        detr_inv_tau +
        detr_coeff * abs(ᶜwʲ) +
        detr_buoy_coeff * abs(min(ᶜbuoyʲ - ᶜbuoy⁰, 0)) /
        max(eps(FT), abs(ᶜwʲ - ᶜw⁰)) +
        max_area_limiter,
        1 / dt,
    )
    return detr * FT(2) * hm_limiter(ᶜaʲ)
end

edmfx_entr_detr_tendency!(Yₜ, Y, p, t, colidx, turbconv_model) = nothing

function edmfx_entr_detr_tendency!(
    Yₜ,
    Y,
    p,
    t,
    colidx,
    turbconv_model::PrognosticEDMFX,
)

    n = n_mass_flux_subdomains(turbconv_model)
    (; ᶜentrʲs, ᶜdetrʲs) = p
    (; ᶜq_tot⁰, ᶜh_tot⁰, ᶠu₃⁰) = p

    for j in 1:n

        @. Yₜ.c.sgsʲs.:($$j).ρa[colidx] +=
            Y.c.sgsʲs.:($$j).ρa[colidx] *
            (ᶜentrʲs.:($$j)[colidx] - ᶜdetrʲs.:($$j)[colidx])

        @. Yₜ.c.sgsʲs.:($$j).h_tot[colidx] +=
            ᶜentrʲs.:($$j)[colidx] *
            (ᶜh_tot⁰[colidx] - Y.c.sgsʲs.:($$j).h_tot[colidx])

        @. Yₜ.c.sgsʲs.:($$j).q_tot[colidx] +=
            ᶜentrʲs.:($$j)[colidx] *
            (ᶜq_tot⁰[colidx] - Y.c.sgsʲs.:($$j).q_tot[colidx])

        @. Yₜ.f.sgsʲs.:($$j).u₃[colidx] +=
            ᶠinterp(ᶜentrʲs.:($$j)[colidx]) *
            (ᶠu₃⁰[colidx] - Y.f.sgsʲs.:($$j).u₃[colidx])
    end
    return nothing
end
