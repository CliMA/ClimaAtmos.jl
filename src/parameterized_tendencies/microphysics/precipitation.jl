#####
##### Precipitation models
#####

import CloudMicrophysics.Microphysics1M as CM1
import CloudMicrophysics as CM
import Thermodynamics as TD
import ClimaCore.Spaces as Spaces
import ClimaCore.Operators as Operators
import ClimaCore.Fields as Fields

precipitation_cache(Y, atmos::AtmosModel) =
    precipitation_cache(Y, atmos.precip_model)

#####
##### No Precipitation
#####

precipitation_cache(Y, precip_model::NoPrecipitation) = (;)
precipitation_tendency!(Yₜ, Y, p, t, colidx, ::NoPrecipitation) = nothing

#####
##### 0-Moment without sgs scheme
#####

function precipitation_cache(Y, precip_model::Microphysics0Moment)
    FT = Spaces.undertype(axes(Y.c))
    return (;
        ᶜS_ρq_tot = similar(Y.c, FT),
        ᶜ3d_rain = similar(Y.c, FT),
        ᶜ3d_snow = similar(Y.c, FT),
        col_integrated_rain = similar(Fields.level(Y.c.ρ, 1), FT),
        col_integrated_snow = similar(Fields.level(Y.c.ρ, 1), FT),
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

    @. col_integrated_rain[colidx] =
        col_integrated_rain[colidx] / CAP.ρ_cloud_liq(params)
    @. col_integrated_snow[colidx] =
        col_integrated_snow[colidx] / CAP.ρ_cloud_liq(params)

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
        end
    end
    return nothing
end
