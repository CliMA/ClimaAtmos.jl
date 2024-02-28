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
    ᶜtke⁰,
    ::NoEntrainment,
) where {FT}
    return FT(0)
end

function entrainment(
    params,
    ᶜz::FT,
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
    ᶜtke⁰,
    ::PiGroupsEntrainment,
) where {FT}
    if ᶜaʲ <= FT(0)
        return FT(0)
    else
        g = CAP.grav(params)
        ref_H = ᶜp / (ᶜρ * g)

        entr_param_vec = CAP.entr_param_vec(params)

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
                entr_param_vec[1] * abs(Π₁) +
                entr_param_vec[2] * abs(Π₂) +
                entr_param_vec[3] * abs(Π₃) +
                entr_param_vec[4] * abs(Π₄) +
                entr_param_vec[5] * abs(Π₅) +
                entr_param_vec[6]
            )

        return max(entr, 0)
    end
end

function entrainment(
    params,
    ᶜz::FT,
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
    ᶜtke⁰,
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

    return max(entr, 0)
end

function detrainment_from_thermo_state(
    params,
    z_prev_level::FT,
    z_sfc_halflevel,
    p_prev_level,
    ρ_prev_level,
    ρaʲ_prev_level,
    tsʲ_prev_level,
    ρʲ_prev_level,
    u³ʲ_prev_halflevel,
    local_geometry_prev_halflevel,
    u³_prev_halflevel,
    ts_prev_level,
    ᶜbuoy⁰,
    entrʲ_prev_level,
    vert_div_level,
    ᶜmassflux_vert_div, # mass flux divergence is not implemented for diagnostic edmf
    tke_prev_level,
    edmfx_detr_model,
) where {FT}
    thermo_params = CAP.thermodynamics_params(params)
    detrainment(
        params,
        z_prev_level,
        z_sfc_halflevel,
        p_prev_level,
        ρ_prev_level,
        ρaʲ_prev_level,
        draft_area(ρaʲ_prev_level, ρʲ_prev_level),
        get_physical_w(u³ʲ_prev_halflevel, local_geometry_prev_halflevel),
        TD.relative_humidity(thermo_params, tsʲ_prev_level),
        ᶜphysical_buoyancy(params, ρ_prev_level, ρʲ_prev_level),
        get_physical_w(u³_prev_halflevel, local_geometry_prev_halflevel),
        TD.relative_humidity(thermo_params, ts_prev_level),
        FT(0),
        entrʲ_prev_level,
        vert_div_level,
        FT(0), # mass flux divergence is not implemented for diagnostic edmf
        tke_prev_level,
        edmfx_detr_model,
    )
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
    ᶜtke⁰,
    ::NoDetrainment,
) where {FT}
    return FT(0)
end

function detrainment(
    params,
    ᶜz::FT,
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
    ᶜtke⁰,
    ::PiGroupsDetrainment,
) where {FT}

    if ᶜaʲ <= FT(0)
        return FT(0)
    else
        g = CAP.grav(params)
        ref_H = ᶜp / (ᶜρ * g)

        entr_param_vec = CAP.entr_param_vec(params)

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
                entr_param_vec[7] * abs(Π₁) +
                entr_param_vec[8] * abs(Π₂) +
                entr_param_vec[9] * abs(Π₃) +
                entr_param_vec[10] * abs(Π₄) +
                entr_param_vec[11] * abs(Π₅) +
                entr_param_vec[12]
            )
        return max(detr, 0)
    end
end

function detrainment(
    params,
    ᶜz::FT,
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
    ᶜtke⁰,
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

    return max(detr, 0)
end

function detrainment(
    params,
    ᶜz::FT,
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
    ::ConstantAreaDetrainment,
) where {FT}
    detr = ᶜentr - ᶜvert_div
    return max(detr, 0)
end

function turbulent_entrainment(params, ᶜaʲ::FT) where {FT}
    turb_entr_param_vec = CAP.turb_entr_param_vec(params)
    return max(
        turb_entr_param_vec[1] * exp(-turb_entr_param_vec[2] * ᶜaʲ),
        FT(0),
    )
end

edmfx_entr_detr_tendency!(Yₜ, Y, p, t, turbconv_model) = nothing

function edmfx_entr_detr_tendency!(Yₜ, Y, p, t, turbconv_model::PrognosticEDMFX)

    n = n_mass_flux_subdomains(turbconv_model)
    (; ᶜturb_entrʲs, ᶜentrʲs, ᶜdetrʲs) = p.precomputed
    (; ᶜq_tot⁰, ᶜmse⁰, ᶠu₃⁰) = p.precomputed

    for j in 1:n

        @. Yₜ.c.sgsʲs.:($$j).ρa +=
            Y.c.sgsʲs.:($$j).ρa * (ᶜentrʲs.:($$j) - ᶜdetrʲs.:($$j))

        @. Yₜ.c.sgsʲs.:($$j).mse +=
            (ᶜentrʲs.:($$j) .+ ᶜturb_entrʲs.:($$j)) *
            (ᶜmse⁰ - Y.c.sgsʲs.:($$j).mse)

        @. Yₜ.c.sgsʲs.:($$j).q_tot +=
            (ᶜentrʲs.:($$j) .+ ᶜturb_entrʲs.:($$j)) *
            (ᶜq_tot⁰ - Y.c.sgsʲs.:($$j).q_tot)

        @. Yₜ.f.sgsʲs.:($$j).u₃ +=
            (ᶠinterp(ᶜentrʲs.:($$j)) .+ ᶠinterp(ᶜturb_entrʲs.:($$j))) *
            (ᶠu₃⁰ - Y.f.sgsʲs.:($$j).u₃)
    end
    return nothing
end

# limit entrainment and detrainment rates for prognostic EDMF
# limit rates approximately below the inverse timescale 1/dt
limit_entrainment(entr::FT, a, dt) where {FT} = max(
    min(entr, FT(0.9) * (1 - a) / max(a, eps(FT)) / dt, FT(0.9) * 1 / dt),
    0,
)
limit_detrainment(detr::FT, a, dt) where {FT} =
    max(min(detr, FT(0.9) * 1 / dt), 0)

function limit_turb_entrainment(dyn_entr::FT, turb_entr::FT, dt) where {FT}
    return max(min((FT(0.9) * 1 / dt) - dyn_entr, turb_entr), 0)
end

# limit entrainment and detrainment rates for diagnostic EDMF
# limit rates approximately below the inverse timescale w/dz
limit_entrainment(entr::FT, a, w, dz) where {FT} =
    max(min(entr, FT(0.9) * w / dz), 0)

limit_detrainment(detr::FT, a, w, dz, dt) where {FT} =
    limit_detrainment(max(min(detr, FT(0.9) * w / dz), 0), a, dt)

function limit_turb_entrainment(dyn_entr::FT, turb_entr::FT, w, dz) where {FT}
    return max(min((FT(0.9) * w / dz) - dyn_entr, turb_entr), 0)
end
