#####
##### EDMF entrainment detrainment
#####

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
    ᶜaʲ::FT,
    ᶜwʲ::FT,
    ᶜRHʲ::FT,
    ᶜbuoyʲ::FT,
    ᶜw⁰::FT,
    ᶜRH⁰::FT,
    ᶜbuoy⁰::FT,
    ᶜtke⁰::FT,
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
    ᶜaʲ::FT,
    ᶜwʲ::FT,
    ᶜRHʲ::FT,
    ᶜbuoyʲ::FT,
    ᶜw⁰::FT,
    ᶜRH⁰::FT,
    ᶜbuoy⁰::FT,
    ᶜtke⁰::FT,
    ::PiGroupsEntrainment,
) where {FT}
    if ᶜaʲ <= FT(0)
        return FT(0)
    else
        g = CAP.grav(params)
        ref_H = ᶜp / (ᶜρ * g)

        # non-dimensional pi-groups
        Π₁ = (ᶜz - z_sfc) * (ᶜbuoyʲ - ᶜbuoy⁰) / ((ᶜwʲ - ᶜw⁰)^2 + eps(FT)) / 100
        Π₂ = max(ᶜtke⁰, 0) / ((ᶜwʲ - ᶜw⁰)^2 + eps(FT)) / 2
        Π₃ = sqrt(ᶜaʲ)
        Π₄ = ᶜRHʲ - ᶜRH⁰
        Π₅ = (ᶜz - z_sfc) / ref_H
        # Π₁, Π₂ are unbounded, so clip values that blow up
        Π₁ = min(max(Π₁, -1), 1)
        Π₂ = min(max(Π₂, -1), 1)
        entr =
            abs(ᶜwʲ - ᶜw⁰) / (ᶜz - z_sfc) * (
                -0.32332 + 4.79372 * Π₁ + 3.98108 * Π₂ - 21.64173 * Π₃ +
                18.395 * Π₄ +
                1.12799 * Π₅
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
    ᶜaʲ::FT,
    ᶜwʲ::FT,
    ᶜRHʲ::FT,
    ᶜbuoyʲ::FT,
    ᶜw⁰::FT,
    ᶜRH⁰::FT,
    ᶜbuoy⁰::FT,
    ᶜtke⁰::FT,
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
    entr =
        entr_inv_tau +
        entr_coeff * abs(ᶜwʲ - ᶜw⁰) / (ᶜz - z_sfc) +
        min_area_limiter

    return entr
end

"""
   Return detrainment rate [1/s].

   Inputs (everything defined on cell centers):
   - params set with model parameters
   - ᶜz, z_sfc, ᶜp, ᶜρ, - grid-scale height, surface height, grid-scale pressure and density
   - buoy_flux_surface - buoyancy flux at the surface
   - ᶜρaʲ, ᶜaʲ, ᶜwʲ, ᶜRHʲ, ᶜbuoyʲ - updraft effective density, updraft area, physical vertical velocity,
                                   relative humidity and buoyancy
   - ᶜw⁰, ᶜRH⁰, ᶜbuoy⁰ - environment physical vertical velocity,
                              relative humidity and buoyancy
   - ᶜentr - entrainment rate
   - ᶜvert_div,ᶜmassflux_vert_div - vertical divergence, vertical mass flux divergence
"""
function detrainment(
    params,
    ᶜz::FT,
    z_sfc::FT,
    ᶜp::FT,
    ᶜρ::FT,
    ᶜρaʲ::FT,
    ᶜaʲ::FT,
    ᶜwʲ::FT,
    ᶜRHʲ::FT,
    ᶜbuoyʲ::FT,
    ᶜw⁰::FT,
    ᶜRH⁰::FT,
    ᶜbuoy⁰::FT,
    ᶜentr::FT,
    ᶜvert_div::FT,
    ᶜmassflux_vert_div::FT,
    ᶜtke⁰::FT,
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
    ᶜρaʲ::FT,
    ᶜaʲ::FT,
    ᶜwʲ::FT,
    ᶜRHʲ::FT,
    ᶜbuoyʲ::FT,
    ᶜw⁰::FT,
    ᶜRH⁰::FT,
    ᶜbuoy⁰::FT,
    ᶜentr::FT,
    ᶜvert_div::FT,
    ᶜmassflux_vert_div::FT,
    ᶜtke⁰::FT,
    ::PiGroupsDetrainment,
) where {FT}

    if ᶜaʲ <= FT(0)
        return FT(0)
    else
        g = CAP.grav(params)
        ref_H = ᶜp / (ᶜρ * g)

        # non-dimensional pi-groups
        Π₁ = (ᶜz - z_sfc) * (ᶜbuoyʲ - ᶜbuoy⁰) / ((ᶜwʲ - ᶜw⁰)^2 + eps(FT)) / 100
        Π₂ = max(ᶜtke⁰, 0) / ((ᶜwʲ - ᶜw⁰)^2 + eps(FT)) / 2
        Π₃ = sqrt(ᶜaʲ)
        Π₄ = ᶜRHʲ - ᶜRH⁰
        Π₅ = (ᶜz - z_sfc) / ref_H
        # Π₁, Π₂ are unbounded, so clip values that blow up
        Π₁ = min(max(Π₁, -1), 1)
        Π₂ = min(max(Π₂, -1), 1)
        detr =
            -min(ᶜmassflux_vert_div, 0) / ᶜρaʲ * (
                0.3410 - 0.56153 * Π₁ - 0.53411 * Π₂ + 6.01925 * Π₃ -
                1.47516 * Π₄ - 3.85788 * Π₅
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
    ᶜρaʲ::FT,
    ᶜaʲ::FT,
    ᶜwʲ::FT,
    ᶜRHʲ::FT,
    ᶜbuoyʲ::FT,
    ᶜw⁰::FT,
    ᶜRH⁰::FT,
    ᶜbuoy⁰::FT,
    ᶜentr::FT,
    ᶜvert_div::FT,
    ᶜmassflux_vert_div::FT,
    ᶜtke⁰::FT,
    ::GeneralizedDetrainment,
) where {FT}
    turbconv_params = CAP.turbconv_params(params)
    detr_inv_tau = CAP.detr_tau(turbconv_params)
    detr_coeff = CAP.detr_coeff(turbconv_params)
    detr_buoy_coeff = CAP.detr_buoy_coeff(turbconv_params)
    detr_vertdiv_coeff = CAP.detr_vertdiv_coeff(turbconv_params)
    detr_massflux_vertdiv_coeff =
        CAP.detr_massflux_vertdiv_coeff(turbconv_params)
    max_area_limiter_scale = CAP.max_area_limiter_scale(turbconv_params)
    max_area_limiter_power = CAP.max_area_limiter_power(turbconv_params)
    a_max = CAP.max_area(turbconv_params)

    max_area_limiter =
        max_area_limiter_scale *
        exp(-max_area_limiter_power * (a_max - min(ᶜaʲ, 1)))

    if ᶜρaʲ <= 0
        detr = 0
    else
        detr =
            detr_inv_tau +
            detr_coeff * abs(ᶜwʲ) +
            detr_buoy_coeff * abs(min(ᶜbuoyʲ - ᶜbuoy⁰, 0)) /
            max(eps(FT), abs(ᶜwʲ - ᶜw⁰)) - detr_vertdiv_coeff * ᶜvert_div -
            detr_massflux_vertdiv_coeff * min(ᶜmassflux_vert_div, 0) / ᶜρaʲ +
            max_area_limiter
    end

    return detr
end

function detrainment(
    params,
    ᶜz::FT,
    z_sfc::FT,
    ᶜp::FT,
    ᶜρ::FT,
    ᶜρaʲ::FT,
    ᶜaʲ::FT,
    ᶜwʲ::FT,
    ᶜRHʲ::FT,
    ᶜbuoyʲ::FT,
    ᶜw⁰::FT,
    ᶜRH⁰::FT,
    ᶜbuoy⁰::FT,
    ᶜentr::FT,
    ᶜvert_div::FT,
    ᶜmassflux_vert_div::FT,
    ::ConstantAreaDetrainment,
) where {FT}
    detr = ᶜentr - ᶜvert_div
    return detr
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
    (; ᶜentrʲs, ᶜdetrʲs) = p.precomputed
    (; ᶜq_tot⁰, ᶜmse⁰, ᶠu₃⁰) = p.precomputed

    for j in 1:n

        @. Yₜ.c.sgsʲs.:($$j).ρa[colidx] +=
            Y.c.sgsʲs.:($$j).ρa[colidx] *
            (ᶜentrʲs.:($$j)[colidx] - ᶜdetrʲs.:($$j)[colidx])

        @. Yₜ.c.sgsʲs.:($$j).mse[colidx] +=
            ᶜentrʲs.:($$j)[colidx] *
            (ᶜmse⁰[colidx] - Y.c.sgsʲs.:($$j).mse[colidx])

        @. Yₜ.c.sgsʲs.:($$j).q_tot[colidx] +=
            ᶜentrʲs.:($$j)[colidx] *
            (ᶜq_tot⁰[colidx] - Y.c.sgsʲs.:($$j).q_tot[colidx])

        @. Yₜ.f.sgsʲs.:($$j).u₃[colidx] +=
            ᶠinterp(ᶜentrʲs.:($$j)[colidx]) *
            (ᶠu₃⁰[colidx] - Y.f.sgsʲs.:($$j).u₃[colidx])
    end
    return nothing
end

limit_entrainment(entr::FT, a, dt) where {FT} =
    max(min(entr, FT(0.9) * (1 - a) / max(a, eps(FT)) / dt), 0)
limit_entrainment(entr::FT, a, w, dz) where {FT} =
    max(min(entr, FT(0.9) * w / dz), 0)
limit_detrainment(detr::FT, a, dt) where {FT} =
    max(min(detr, FT(0.9) * 1 / dt), 0)
limit_detrainment(detr::FT, a, w, dz) where {FT} =
    max(min(detr, FT(0.9) * w / dz), 0)
