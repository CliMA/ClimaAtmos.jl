#####
##### Precipitation models
#####

import CloudMicrophysics.Microphysics1M as CM1
import CloudMicrophysics as CM
import Thermodynamics as TD
import ClimaCore.Spaces as Spaces
import ClimaCore.Operators as Operators
import ClimaCore.Fields as Fields

#####
##### No Precipitation
#####

precipitation_cache(Y, precip_model::NoPrecipitation) = (; precip_model)
precipitation_tendency!(Yₜ, Y, p, t, colidx, ::NoPrecipitation) = nothing

#####
##### 0-Moment without sgs scheme
#####

function precipitation_cache(Y, precip_model::Microphysics0Moment)
    FT = Spaces.undertype(axes(Y.c))
    return (;
        precip_model,
        ᶜS_ρq_tot = similar(Y.c, FT),
        ᶜ3d_rain = similar(Y.c, FT),
        ᶜ3d_snow = similar(Y.c, FT),
        col_integrated_rain = similar(Fields.level(Y.c.ρ, 1), FT),
        col_integrated_snow = similar(Fields.level(Y.c.ρ, 1), FT),
    )
end

function compute_precipitation_cache!(Y, p, colidx, ::Microphysics0Moment, _)
    (; ᶜts, ᶜS_ρq_tot, params) = p
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

    (; ᶜS_ρq_tot, ᶜS_q_tot⁰, ᶜS_q_totʲs, ᶜρaʲs) = p
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
    (;
        ᶜts,
        ᶜΦ,
        ᶜT,
        ᶜ3d_rain,
        ᶜ3d_snow,
        ᶜS_ρq_tot,
        col_integrated_rain,
        col_integrated_snow,
        params,
        turbconv_model,
    ) = p # assume ᶜts has been updated
    thermo_params = CAP.thermodynamics_params(params)
    cm_params = CAP.microphysics_params(params)
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
                    p.ᶜS_q_totʲs[colidx] *
                    p.ᶜρaʲs[colidx] *
                    p.ᶜS_e_totʲs_helper[colidx],
                ) +
                p.ᶜS_q_tot⁰[colidx] *
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


#####
##### 1-Moment coupled to sgs
#####
# TODO: move 1-moment microphysics cache / tendency here
function precipitation_cache(Y, precip_model::Microphysics1Moment)
    FT = Spaces.undertype(axes(Y.c))

    return (;
        precip_model,
        ᶜS_ρe_tot = similar(Y.c, FT),
        ᶜS_ρq_tot = similar(Y.c, FT),
        ᶜS_ρq_rai = similar(Y.c, FT),
        ᶜS_ρq_sno = similar(Y.c, FT),
        ᶜS_q_rai_evap = similar(Y.c, FT),
        ᶜS_q_sno_melt = similar(Y.c, FT),
        ᶜS_q_sno_sub_dep = similar(Y.c, FT),
        ᶜq_rai = similar(Y.c, FT),
        ᶜq_sno = similar(Y.c, FT),
    )
end

"""
Computes the rain and snow advection (down) tendency
"""
function precipitation_advection_tendency!(
    Yₜ,
    Y,
    p,
    colidx,
    ::Microphysics1Moment,
)
    FT = Spaces.undertype(axes(Y.c))
    (; params) = p

    ρ_c = Y.c.ρ[colidx]

    # helper to calculate the rain velocity
    # TODO: assuming w_gm = 0
    # TODO: verify translation

    ρq_rai = Y.c.ρq_rai[colidx]
    ρq_sno = Y.c.ρq_sno[colidx]

    RB = Operators.RightBiasedC2F(; top = Operators.SetValue(FT(0)))
    ᶜdivᵥ = Operators.DivergenceF2C(; bottom = Operators.Extrapolate())
    wvec = Geometry.WVector
    microphys_params = CAP.microphysics_params(params)
    rain_type = CM.CommonTypes.RainType()
    snow_type = CM.CommonTypes.SnowType()
    velo_type = CM.CommonTypes.Blk1MVelType()

    # TODO - some positivity limiters are needed

    # TODO: need to add horizontal advection + vertical velocity of air

    # TODO: use correct advection operators
    if !isnothing(Yₜ)
        @. Yₜ.c.ρq_rai[colidx] += ᶜdivᵥ(
            wvec(
                RB(
                    ρq_rai * CM1.terminal_velocity(
                        microphys_params,
                        rain_type,
                        velo_type,
                        ρ_c,
                        ρq_rai / ρ_c,
                    ),
                ),
            ),
        )

        @. Yₜ.c.ρq_sno[colidx] += ᶜdivᵥ(
            wvec(
                RB(
                    ρq_sno * CM1.terminal_velocity(
                        microphys_params,
                        snow_type,
                        velo_type,
                        ρ_c,
                        ρq_sno / ρ_c,
                    ),
                ),
            ),
        )
    end
    return nothing
end

function precipitation_tendency!(
    Yₜ,
    Y,
    p,
    t,
    colidx,
    precip_model::Microphysics1Moment,
)
    (; ᶜS_ρe_tot, ᶜS_ρq_tot, ᶜS_ρq_rai, ᶜS_ρq_sno) = p
    compute_precipitation_cache!(
        Y,
        p,
        colidx,
        precip_model,
        p.atmos.turbconv_model,
    )

    if !isnothing(Yₜ)
        @. Yₜ.c.ρ[colidx] += ᶜS_ρq_tot[colidx]
        @. Yₜ.c.ρq_tot[colidx] += ᶜS_ρq_tot[colidx]
        @. Yₜ.c.ρq_rai[colidx] += ᶜS_ρq_rai[colidx]
        @. Yₜ.c.ρq_sno[colidx] += ᶜS_ρq_sno[colidx]

        if :ρe_tot in propertynames(Y.c)
            @. Yₜ.c.ρe_tot[colidx] += ᶜS_ρe_tot[colidx]
        else
            error(
                "1-moment microphysics can only be coupled to ρe_tot energy variable",
            )
        end
    end

    precipitation_advection_tendency!(Yₜ, Y, p, colidx, precip_model)
    return nothing
end
