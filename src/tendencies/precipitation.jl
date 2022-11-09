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
        ᶜλ = similar(Y.c, FT),
        ᶜ3d_rain = similar(Y.c, FT),
        ᶜ3d_snow = similar(Y.c, FT),
        col_integrated_rain = similar(Fields.level(Y.c.ρ, 1), FT),
        col_integrated_snow = similar(Fields.level(Y.c.ρ, 1), FT),
    )
end

function compute_precipitation_cache!(
    Y,
    p,
    colidx,
    ::Microphysics0Moment,
    ::Nothing,
)
    (; ᶜts, ᶜS_ρq_tot, params) = p
    cm_params = CAP.microphysics_params(params)
    thermo_params = CAP.thermodynamics_params(params)
    @. ᶜS_ρq_tot[colidx] =
        Y.c.ρ[colidx] * CM.Microphysics0M.remove_precipitation(
            cm_params,
            TD.PhasePartition(thermo_params, ᶜts[colidx]),
        )
end

function precipitation_tendency!(
    Yₜ,
    Y,
    p,
    t,
    colidx,
    precip_model::Microphysics0Moment,
)
    FT = Spaces.undertype(axes(Y.c))
    (;
        ᶜts,
        ᶜΦ,
        ᶜT,
        ᶜ3d_rain,
        ᶜ3d_snow,
        ᶜS_ρq_tot,
        ᶜλ,
        col_integrated_rain,
        col_integrated_snow,
        params,
        turbconv_model,
    ) = p # assume ᶜts has been updated
    thermo_params = CAP.thermodynamics_params(params)
    cm_params = CAP.microphysics_params(params)
    compute_precipitation_cache!(Y, p, colidx, precip_model, turbconv_model)
    @. Yₜ.c.ρq_tot[colidx] += ᶜS_ρq_tot[colidx]
    @. Yₜ.c.ρ[colidx] += ᶜS_ρq_tot[colidx]

    # update precip in cache for coupler's use
    # 3d rain and snow
    @. ᶜT[colidx] = TD.air_temperature(thermo_params, ᶜts[colidx])
    @. ᶜ3d_rain[colidx] =
        ifelse(ᶜT[colidx] >= FT(273.15), ᶜS_ρq_tot[colidx], FT(0))
    @. ᶜ3d_snow[colidx] =
        ifelse(ᶜT[colidx] < FT(273.15), ᶜS_ρq_tot[colidx], FT(0))
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

    # liquid fraction
    @. ᶜλ[colidx] = TD.liquid_fraction(thermo_params, ᶜts[colidx])

    if :ρe_tot in propertynames(Y.c)
        @. Yₜ.c.ρe_tot[colidx] +=
            ᶜS_ρq_tot[colidx] * (
                ᶜλ[colidx] *
                TD.internal_energy_liquid(thermo_params, ᶜts[colidx]) +
                (1 - ᶜλ[colidx]) *
                TD.internal_energy_ice(thermo_params, ᶜts[colidx]) +
                ᶜΦ[colidx]
            )
    elseif :ρe_int in propertynames(Y.c)
        @. Yₜ.c.ρe_int[colidx] +=
            ᶜS_ρq_tot[colidx] * (
                ᶜλ[colidx] *
                TD.internal_energy_liquid(thermo_params, ᶜts[colidx]) +
                (1 - ᶜλ[colidx]) *
                TD.internal_energy_ice(thermo_params, ᶜts[colidx])
            )
    end
    return nothing
end


#####
##### 1-Moment coupled to sgs
#####
# TODO: move 1-moment microphysics cache / tendency here
function precipitation_cache(Y, precip_model::Microphysics1Moment)
    FT = Spaces.undertype(axes(Y.c))
    return (; precip_model)
end

function precipitation_tendency!(
    Yₜ,
    Y,
    p,
    t,
    colidx,
    precip_model::Microphysics1Moment,
)
    precipitation_advection_tendency!(Yₜ, Y, p, colidx, precip_model)
    return nothing
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

    q_rai = Y.c.q_rai[colidx] #./ precip_fraction
    q_sno = Y.c.q_sno[colidx] #./ precip_fraction

    RB = Operators.RightBiasedC2F(; top = Operators.SetValue(FT(0)))
    ᶜdivᵥ = Operators.DivergenceF2C(; bottom = Operators.Extrapolate())
    wvec = Geometry.WVector
    microphys_params = CAP.microphysics_params(params)
    rain_type = CM.CommonTypes.RainType()
    snow_type = CM.CommonTypes.SnowType()

    # TODO - some positivity limiters are needed

    # TODO: need to add horizontal advection + vertical velocity of air

    # TODO: use correct advection operators
    # TODO: use ρq_rai, ρq_sno
    @. Yₜ.c.q_rai[colidx] +=
        ᶜdivᵥ(
            wvec(
                RB(
                    ρ_c *
                    q_rai *
                    CM1.terminal_velocity(
                        microphys_params,
                        rain_type,
                        ρ_c,
                        q_rai,
                    ),
                ),
            ),
        ) / ρ_c

    @. Yₜ.c.q_sno[colidx] +=
        ᶜdivᵥ(
            wvec(
                RB(
                    ρ_c *
                    q_sno *
                    CM1.terminal_velocity(
                        microphys_params,
                        snow_type,
                        ρ_c,
                        q_sno,
                    ),
                ),
            ),
        ) / ρ_c
    return nothing
end
