#####
##### Precipitation models
#####

import CloudMicrophysics.Microphysics1M as CM1
import CloudMicrophysics as CM
import Thermodynamics as TD
import ClimaCore.Spaces as Spaces
import ClimaCore.Operators as Operators
import ClimaCore.Fields as Fields
import ClimaCore.Utilities: half

precipitation_cache(Y, atmos::AtmosModel) =
    precipitation_cache(Y, atmos.precip_model)

#####
##### No Precipitation
#####

precipitation_cache(Y, precip_model::NoPrecipitation) = (;)
precipitation_tendency!(Yₜ, Y, p, t, colidx, ::NoPrecipitation) = nothing

#####
##### 0-Moment without sgs scheme or with diagnostic edmf
#####

function precipitation_cache(Y, precip_model::Microphysics0Moment)
    FT = Spaces.undertype(axes(Y.c))
    return (;
        ᶜS_ρq_tot = zeros(axes(Y.c)),
        ᶜ3d_rain = zeros(axes(Y.c)),
        ᶜ3d_snow = zeros(axes(Y.c)),
        col_integrated_rain = zeros(
            axes(Fields.level(Geometry.WVector.(Y.f.u₃), half)),
        ),
        col_integrated_snow = zeros(
            axes(Fields.level(Geometry.WVector.(Y.f.u₃), half)),
        ),
    )
end

function compute_precipitation_cache!(Y, p, colidx, ::Microphysics0Moment, _)
    (; params) = p
    (; ᶜts) = p.precomputed
    (; ᶜS_ρq_tot) = p.precipitation
    cm_params = CAP.microphysics_params(params)
    thermo_params = CAP.thermodynamics_params(params)
    #TODO missing limiting by q_tot/Δt
    @. ᶜS_ρq_tot[colidx] =
        Y.c.ρ[colidx] * CM.Microphysics0M.remove_precipitation(
            cm_params,
            TD.PhasePartition(thermo_params, ᶜts[colidx]),
        )
end

function compute_precipitation_cache!(
    Y,
    p,
    colidx,
    ::Microphysics0Moment,
    ::DiagnosticEDMFX,
)
    # For environment we multiply by grid mean ρ and not byᶜρa⁰
    # I.e. assuming a⁰=1

    (; ᶜS_q_tot⁰, ᶜS_q_totʲs, ᶜρaʲs) = p.precomputed
    (; ᶜS_ρq_tot) = p.precipitation
    n = n_mass_flux_subdomains(p.atmos.turbconv_model)
    ρ = Y.c.ρ

    @. ᶜS_ρq_tot[colidx] = ᶜS_q_tot⁰[colidx] * ρ[colidx]
    for j in 1:n
        @. ᶜS_ρq_tot[colidx] += ᶜS_q_totʲs.:($$j)[colidx] * ᶜρaʲs.:($$j)[colidx]
    end
end

function compute_precipitation_cache!(
    Y,
    p,
    colidx,
    ::Microphysics0Moment,
    ::PrognosticEDMFX,
)
    (; ᶜS_q_tot⁰, ᶜS_q_totʲs, ᶜρa⁰) = p.precomputed
    (; ᶜS_ρq_tot) = p.precipitation
    n = n_mass_flux_subdomains(p.atmos.turbconv_model)

    @. ᶜS_ρq_tot[colidx] = ᶜS_q_tot⁰[colidx] * ᶜρa⁰[colidx]
    for j in 1:n
        @. ᶜS_ρq_tot[colidx] +=
            ᶜS_q_totʲs.:($$j)[colidx] * Y.c.sgsʲs.:($$j).ρa[colidx]
    end
end

function precipitation_tendency!(
    Yₜ,
    Y,
    p,
    t,
    colidx,
    precip_model::Microphysics0Moment,
)
    (; ᶜT, ᶜΦ) = p.core
    (; ᶜts,) = p.precomputed  # assume ᶜts has been updated
    (; params) = p
    (; turbconv_model) = p.atmos
    (;
        ᶜ3d_rain,
        ᶜ3d_snow,
        ᶜS_ρq_tot,
        col_integrated_rain,
        col_integrated_snow,
    ) = p.precipitation
    col_precip_energy =
        p.conservation_check.col_integrated_precip_energy_tendency

    thermo_params = CAP.thermodynamics_params(params)
    compute_precipitation_cache!(Y, p, colidx, precip_model, turbconv_model)
    if !isnothing(Yₜ)
        @. Yₜ.c.ρq_tot[colidx] += ᶜS_ρq_tot[colidx]
        @. Yₜ.c.ρ[colidx] += ᶜS_ρq_tot[colidx]
    end
    T_freeze = TD.Parameters.T_freeze(thermo_params)

    # update precip in cache for coupler's use
    # 3d rain and snow
    @. ᶜT[colidx] = TD.air_temperature(thermo_params, ᶜts[colidx])
    @. ᶜ3d_rain[colidx] = ifelse(ᶜT[colidx] >= T_freeze, ᶜS_ρq_tot[colidx], 0)
    @. ᶜ3d_snow[colidx] = ifelse(ᶜT[colidx] < T_freeze, ᶜS_ρq_tot[colidx], 0)
    Operators.column_integral_definite!(
        col_integrated_rain[colidx],
        ᶜ3d_rain[colidx],
    )
    Operators.column_integral_definite!(
        col_integrated_snow[colidx],
        ᶜ3d_snow[colidx],
    )

    if :ρe_tot in propertynames(Y.c)
        #TODO - this is a hack right now. But it will be easier to clean up
        # once we drop the support for the old EDMF code
        if turbconv_model isa DiagnosticEDMFX && !isnothing(Yₜ)
            @. Yₜ.c.ρe_tot[colidx] +=
                sum(
                    p.precomputed.ᶜS_q_totʲs[colidx] *
                    p.precomputed.ᶜρaʲs[colidx] *
                    p.precomputed.ᶜS_e_totʲs_helper[colidx],
                ) +
                p.precomputed.ᶜS_q_tot⁰[colidx] *
                Y.c.ρ[colidx] *
                e_tot_0M_precipitation_sources_helper(
                    thermo_params,
                    ᶜts[colidx],
                    ᶜΦ[colidx],
                )
        elseif !isnothing(Yₜ)
            @. Yₜ.c.ρe_tot[colidx] +=
                ᶜS_ρq_tot[colidx] * e_tot_0M_precipitation_sources_helper(
                    thermo_params,
                    ᶜts[colidx],
                    ᶜΦ[colidx],
                )
            if p.atmos.surface_model isa PrognosticSurfaceTemperature
                Operators.column_integral_definite!(
                    col_precip_energy[colidx],
                    @. ᶜS_ρq_tot[colidx] *
                       e_tot_0M_precipitation_sources_helper(
                        thermo_params,
                        ᶜts[colidx],
                        ᶜΦ[colidx],
                    )
                )
            end
        end
    end
    return nothing
end

#####
##### 1-Moment without sgs scheme
#####

function precipitation_cache(Y, precip_model::Microphysics1Moment)
    FT = Spaces.undertype(axes(Y.c))
    return (;
        ᶜSqₜᵖ = similar(Y.c, FT),
        ᶜSqᵣᵖ = similar(Y.c, FT),
        ᶜSqₛᵖ = similar(Y.c, FT),
        ᶜSeₜᵖ = similar(Y.c, FT),
    )
end

function compute_precipitation_cache!(Y, p, colidx, ::Microphysics1Moment, _)
    FT = Spaces.undertype(axes(Y.c))
    (; params) = p
    (; dt) = p
    (; ᶜts) = p.precomputed
    (; ᶜΦ) = p.core
    (; ᶜSqₜᵖ, ᶜSqᵣᵖ, ᶜSqₛᵖ, ᶜSeₜᵖ) = p.precipitation
    ᶜz = Fields.coordinate_field(Y.c).z

    ᶜSᵖ = p.scratch.ᶜtemp_scalar

    # get thermodynamics and 1-moment microphysics params
    cmp = CAP.microphysics_params(params)
    thp = CAP.thermodynamics_params(params)

    # some helper functions to make the code more readable
    Iₗ(ts) = TD.internal_energy_liquid(thp, ts)
    Iᵢ(ts) = TD.internal_energy_ice(thp, ts)
    qₗ(ts) = TD.PhasePartition(thp, ts).liq
    qᵢ(ts) = TD.PhasePartition(thp, ts).ice
    qᵥ(ts) = TD.vapor_specific_humidity(thp, ts)
    Lf(ts) = TD.latent_heat_fusion(thp, ts)
    Tₐ(ts) = TD.air_temperature(thp, ts)
    α(ts) = TD.Parameters.cv_l(thp) / Lf(ts) * (Tₐ(ts) - cmp.ps.T_freeze)

    qₚ(ρqₚ, ρ) = max(FT(0), ρqₚ / ρ)

    # zero out the source terms
    @. ᶜSqₜᵖ[colidx] = FT(0)
    @. ᶜSeₜᵖ[colidx] = FT(0)
    @. ᶜSqᵣᵖ[colidx] = FT(0)
    @. ᶜSqₛᵖ[colidx] = FT(0)

    # All the tendencies are individually limited
    # by the available condensate (q_ / dt).

    # rain autoconversion: q_liq -> q_rain
    @. ᶜSᵖ[colidx] = min(
        qₗ(ᶜts[colidx]) / dt,
        CM1.conv_q_liq_to_q_rai(
            cmp.pr.acnv1M,
            qₗ(ᶜts[colidx]),
            smooth_transition = true,
        ),
    )
    @. ᶜSqₜᵖ[colidx] -= ᶜSᵖ[colidx]
    @. ᶜSqᵣᵖ[colidx] += ᶜSᵖ[colidx]
    @. ᶜSeₜᵖ[colidx] -= ᶜSᵖ[colidx] * (Iₗ(ᶜts[colidx]) + ᶜΦ[colidx])

    # snow autoconversion assuming no supersaturation: q_ice -> q_snow
    @. ᶜSᵖ[colidx] = min(
        qᵢ(ᶜts[colidx]) / dt,
        CM1.conv_q_ice_to_q_sno_no_supersat(
            cmp.ps.acnv1M,
            qᵢ(ᶜts[colidx]),
            smooth_transition = true,
        ),
    )
    @. ᶜSqₜᵖ[colidx] -= ᶜSᵖ[colidx]
    @. ᶜSqₛᵖ[colidx] += ᶜSᵖ[colidx]
    @. ᶜSeₜᵖ[colidx] -= ᶜSᵖ[colidx] * (Iᵢ(ᶜts[colidx]) + ᶜΦ[colidx])

    # accretion: q_liq + q_rain -> q_rain
    @. ᶜSᵖ[colidx] = min(
        qₗ(ᶜts[colidx]) / dt,
        CM1.accretion(
            cmp.cl,
            cmp.pr,
            cmp.tv.rain,
            cmp.ce,
            qₗ(ᶜts[colidx]),
            qₚ(Y.c.ρq_rai[colidx], Y.c.ρ[colidx]),
            Y.c.ρ[colidx],
        ),
    )
    @. ᶜSqₜᵖ[colidx] -= ᶜSᵖ[colidx]
    @. ᶜSqᵣᵖ[colidx] += ᶜSᵖ[colidx]
    @. ᶜSeₜᵖ[colidx] -= ᶜSᵖ[colidx] * (Iₗ(ᶜts[colidx]) + ᶜΦ[colidx])

    # accretion: q_ice + q_snow -> q_snow
    @. ᶜSᵖ[colidx] = min(
        qᵢ(ᶜts[colidx]) / dt,
        CM1.accretion(
            cmp.ci,
            cmp.ps,
            cmp.tv.snow,
            cmp.ce,
            qᵢ(ᶜts[colidx]),
            qₚ(Y.c.ρq_sno[colidx], Y.c.ρ[colidx]),
            Y.c.ρ[colidx],
        ),
    )
    @. ᶜSqₜᵖ[colidx] -= ᶜSᵖ[colidx]
    @. ᶜSqₛᵖ[colidx] += ᶜSᵖ[colidx]
    @. ᶜSeₜᵖ[colidx] -= ᶜSᵖ[colidx] * (Iᵢ(ᶜts[colidx]) + ᶜΦ[colidx])

    # accretion: q_liq + q_sno -> q_sno or q_rai
    # sink of cloud water via accretion cloud water + snow
    @. ᶜSᵖ[colidx] = min(
        qₗ(ᶜts[colidx]) / dt,
        CM1.accretion(
            cmp.cl,
            cmp.ps,
            cmp.tv.snow,
            cmp.ce,
            qₗ(ᶜts[colidx]),
            qₚ(Y.c.ρq_sno[colidx], Y.c.ρ[colidx]),
            Y.c.ρ[colidx],
        ),
    )
    # if T < T_freeze cloud droplets freeze to become snow
    # else the snow melts and both cloud water and snow become rain
    ᶜSᵖ_snow = p.scratch.ᶜtemp_scalar_2
    @. ᶜSᵖ_snow[colidx] = ifelse(
        Tₐ(ᶜts[colidx]) < cmp.ps.T_freeze,
        ᶜSᵖ[colidx],
        FT(-1) * min(
            ᶜSᵖ[colidx] * α(ᶜts[colidx]),
            qₚ(Y.c.ρq_sno[colidx], Y.c.ρ[colidx]) / dt,
        ),
    )
    @. ᶜSqₛᵖ[colidx] += ᶜSᵖ_snow[colidx]
    @. ᶜSqₜᵖ[colidx] -= ᶜSᵖ[colidx]
    @. ᶜSqᵣᵖ[colidx] += ifelse(
        Tₐ(ᶜts[colidx]) < cmp.ps.T_freeze,
        FT(0),
        ᶜSᵖ[colidx] - ᶜSᵖ_snow[colidx],
    )
    @. ᶜSeₜᵖ[colidx] -= ifelse(
        Tₐ(ᶜts[colidx]) < cmp.ps.T_freeze,
        ᶜSᵖ[colidx] * (Iᵢ(ᶜts[colidx]) + ᶜΦ[colidx]),
        ᶜSᵖ[colidx] * (Iₗ(ᶜts[colidx]) + ᶜΦ[colidx]) -
        ᶜSᵖ_snow[colidx] * (Iₗ(ᶜts[colidx]) - Iᵢ(ᶜts[colidx])),
    )

    # accretion: q_ice + q_rai -> q_sno
    @. ᶜSᵖ[colidx] = min(
        qᵢ(ᶜts[colidx]) / dt,
        CM1.accretion(
            cmp.ci,
            cmp.pr,
            cmp.tv.rain,
            cmp.ce,
            qᵢ(ᶜts[colidx]),
            qₚ(Y.c.ρq_rai[colidx], Y.c.ρ[colidx]),
            Y.c.ρ[colidx],
        ),
    )
    @. ᶜSqₜᵖ[colidx] -= ᶜSᵖ[colidx]
    @. ᶜSqₛᵖ[colidx] += ᶜSᵖ[colidx]
    @. ᶜSeₜᵖ[colidx] -= ᶜSᵖ[colidx] * (Iᵢ(ᶜts[colidx]) + ᶜΦ[colidx])
    # sink of rain via accretion cloud ice - rain
    @. ᶜSᵖ[colidx] = min(
        qₚ(Y.c.ρq_rai[colidx], Y.c.ρ[colidx]) / dt,
        CM1.accretion_rain_sink(
            cmp.pr,
            cmp.ci,
            cmp.tv.rain,
            cmp.ce,
            qᵢ(ᶜts[colidx]),
            qₚ(Y.c.ρq_rai[colidx], Y.c.ρ[colidx]),
            Y.c.ρ[colidx],
        ),
    )
    @. ᶜSqᵣᵖ[colidx] -= ᶜSᵖ[colidx]
    @. ᶜSqₛᵖ[colidx] += ᶜSᵖ[colidx]
    @. ᶜSeₜᵖ[colidx] += ᶜSᵖ[colidx] * Lf(ᶜts[colidx])

    # accretion: q_rai + q_sno -> q_rai or q_sno
    @. ᶜSᵖ[colidx] = ifelse(
        Tₐ(ᶜts[colidx]) < cmp.ps.T_freeze,
        min(
            qₚ(Y.c.ρq_rai[colidx], Y.c.ρ[colidx]) / dt,
            CM1.accretion_snow_rain(
                cmp.ps,
                cmp.pr,
                cmp.tv.rain,
                cmp.tv.snow,
                cmp.ce,
                qₚ(Y.c.ρq_sno[colidx], Y.c.ρ[colidx]),
                qₚ(Y.c.ρq_rai[colidx], Y.c.ρ[colidx]),
                Y.c.ρ[colidx],
            ),
        ),
        -min(
            qₚ(Y.c.ρq_sno[colidx], Y.c.ρ[colidx]) / dt,
            CM1.accretion_snow_rain(
                cmp.pr,
                cmp.ps,
                cmp.tv.snow,
                cmp.tv.rain,
                cmp.ce,
                qₚ(Y.c.ρq_rai[colidx], Y.c.ρ[colidx]),
                qₚ(Y.c.ρq_sno[colidx], Y.c.ρ[colidx]),
                Y.c.ρ[colidx],
            ),
        ),
    )
    @. ᶜSqₛᵖ[colidx] += ᶜSᵖ[colidx]
    @. ᶜSqᵣᵖ[colidx] -= ᶜSᵖ[colidx]
    @. ᶜSeₜᵖ[colidx] += ᶜSᵖ[colidx] * Lf(ᶜts[colidx])

    # evaporation: q_rai -> q_vap
    @. ᶜSᵖ[colidx] =
        -min(
            qₚ(Y.c.ρq_rai[colidx], Y.c.ρ[colidx]) / dt,
            -CM1.evaporation_sublimation(
                cmp.pr,
                cmp.tv.rain,
                cmp.aps,
                thp,
                TD.PhasePartition(thp, ᶜts[colidx]),
                qₚ(Y.c.ρq_rai[colidx], Y.c.ρ[colidx]),
                Y.c.ρ[colidx],
                Tₐ(ᶜts[colidx]),
            ),
        )
    @. ᶜSqₜᵖ[colidx] -= ᶜSᵖ[colidx]
    @. ᶜSqᵣᵖ[colidx] += ᶜSᵖ[colidx]
    @. ᶜSeₜᵖ[colidx] -= ᶜSᵖ[colidx] * (Iₗ(ᶜts[colidx]) + ᶜΦ[colidx])

    # melting: q_sno -> q_rai
    @. ᶜSᵖ[colidx] = min(
        qₚ(Y.c.ρq_sno[colidx], Y.c.ρ[colidx]) / dt,
        CM1.snow_melt(
            cmp.ps,
            cmp.tv.snow,
            cmp.aps,
            thp,
            qₚ(Y.c.ρq_sno[colidx], Y.c.ρ[colidx]),
            Y.c.ρ[colidx],
            Tₐ(ᶜts[colidx]),
        ),
    )
    @. ᶜSqᵣᵖ[colidx] += ᶜSᵖ[colidx]
    @. ᶜSqₛᵖ[colidx] -= ᶜSᵖ[colidx]
    @. ᶜSeₜᵖ[colidx] -= ᶜSᵖ[colidx] * Lf(ᶜts[colidx])

    # deposition/sublimation: q_vap <-> q_sno
    @. ᶜSᵖ[colidx] = CM1.evaporation_sublimation(
        cmp.ps,
        cmp.tv.snow,
        cmp.aps,
        thp,
        TD.PhasePartition(thp, ᶜts[colidx]),
        qₚ(Y.c.ρq_sno[colidx], Y.c.ρ[colidx]),
        Y.c.ρ[colidx],
        Tₐ(ᶜts[colidx]),
    )
    @. ᶜSᵖ[colidx] = ifelse(
        ᶜSᵖ[colidx] > FT(0),
        min(qᵥ(ᶜts[colidx]) / dt, ᶜSᵖ[colidx]),
        -min(qₚ(Y.c.ρq_sno[colidx], Y.c.ρ[colidx]) / dt, FT(-1) * ᶜSᵖ[colidx]),
    )
    @. ᶜSqₜᵖ[colidx] -= ᶜSᵖ[colidx]
    @. ᶜSqₛᵖ[colidx] += ᶜSᵖ[colidx]
    @. ᶜSeₜᵖ[colidx] -= ᶜSᵖ[colidx] * (Iᵢ(ᶜts[colidx]) + ᶜΦ[colidx])
end

function precipitation_tendency!(
    Yₜ,
    Y,
    p,
    t,
    colidx,
    precip_model::Microphysics1Moment,
)
    FT = Spaces.undertype(axes(Y.c))
    (; ᶜSqₜᵖ, ᶜSqᵣᵖ, ᶜSqₛᵖ, ᶜSeₜᵖ) = p.precipitation

    compute_precipitation_cache!(
        Y,
        p,
        colidx,
        precip_model,
        p.atmos.turbconv_model,
    )

    @. Yₜ.c.ρ[colidx] += Y.c.ρ[colidx] * ᶜSqₜᵖ[colidx]
    @. Yₜ.c.ρq_tot[colidx] += Y.c.ρ[colidx] * ᶜSqₜᵖ[colidx]
    @. Yₜ.c.ρe_tot[colidx] += Y.c.ρ[colidx] * ᶜSeₜᵖ[colidx]
    @. Yₜ.c.ρq_rai[colidx] += Y.c.ρ[colidx] * ᶜSqᵣᵖ[colidx]
    @. Yₜ.c.ρq_sno[colidx] += Y.c.ρ[colidx] * ᶜSqₛᵖ[colidx]

    return nothing
end
