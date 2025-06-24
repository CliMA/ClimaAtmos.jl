# A set of wrappers for using CloudMicrophysics.jl functions inside EDMFX loops

import Thermodynamics as TD
import CloudMicrophysics.Microphysics0M as CM0
import CloudMicrophysics.Microphysics1M as CM1
import CloudMicrophysics.Microphysics2M as CM2
import CloudMicrophysics.MicrophysicsNonEq as CMNe
import CloudMicrophysics.Parameters as CMP

# Define some aliases and functions to make the code more readable
const Tₐ = TD.air_temperature
const PP = TD.PhasePartition
const qᵥ = TD.vapor_specific_humidity

# Clip any specific humidity
function clip(q)
    FT = eltype(q)
    return max(FT(0), q)
end

# Helper function to compute the limit of the tendency in the traingle limiter.
# The limit is defined as the available amont / n times model time step.
function limit(q, dt, n::Int)
    FT = eltype(q)
    return max(FT(0), q) / float(dt) / n
end

"""
    ml_N_cloud_liquid_droplets(cmc, c_dust, c_seasalt, c_SO4, q_liq)

 - cmc - a struct with cloud and aerosol parameters
 - c_dust, c_seasalt, c_SO4 - dust, seasalt and ammonium sulfate mass concentrations [kg/kg]
 - q_liq - liquid water specific humidity

Returns the liquid cloud droplet number concentration diagnosed based on the
aerosol loading and cloud liquid water.
"""
function ml_N_cloud_liquid_droplets(cmc, c_dust, c_seasalt, c_SO4, q_liq)
    # We can also add w, T, RH, w' ...
    # Also consider lookind only at around cloud base height
    (; α_dust, α_seasalt, α_SO4, α_q_liq) = cmc.aml
    (; c₀_dust, c₀_seasalt, c₀_SO4, q₀_liq) = cmc.aml
    N₀ = cmc.N_cloud_liquid_droplets

    FT = eltype(N₀)
    return N₀ * (
        FT(1) +
        α_dust * (log(max(c_dust, eps(FT))) - log(c₀_dust)) +
        α_seasalt * (log(max(c_seasalt, eps(FT))) - log(c₀_seasalt)) +
        α_SO4 * (log(max(c_SO4, eps(FT))) - log(c₀_SO4)) +
        α_q_liq * (log(max(q_liq, eps(FT))) - log(q₀_liq))
    )
end

"""
    cloud_sources(cm_params, thp, qₜ, qₗ, qᵢ, qᵣ, qₛ, ρ, Tₐ, dt)

 - cm_params - CloudMicrophysics parameters struct for cloud water or ice condensate
 - thp - Thermodynamics parameters struct
 - qₜ - total specific humidity
 - qₗ - liquid specific humidity
 - qᵢ - ice specific humidity
 - qᵣ - rain specific humidity
 - qₛ - snow specific humidity
 - ρ - air density
 - Tₐ - air temperature
 - dt - model time step

Returns the condensation/evaporation or deposition/sublimation rate for
non-equilibrium Morrison and Milbrandt 2015 cloud formation.
"""
function cloud_sources(
    cm_params::CMP.CloudLiquid{FT},
    thp,
    qₜ,
    qₗ,
    qᵢ,
    qᵣ,
    qₛ,
    ρ,
    Tₐ,
    dt,
) where {FT}

    qᵥ = qₜ - qₗ - qᵢ - qᵣ - qₛ

    if qᵥ + qₗ > FT(0)
        S = CMNe.conv_q_vap_to_q_liq_ice_MM2015(
            cm_params,
            thp,
            qₜ,
            qₗ,
            qᵢ,
            qᵣ,
            qₛ,
            ρ,
            Tₐ,
        )
    else
        S = FT(0)
    end

    return ifelse(
        S > FT(0),
        triangle_inequality_limiter(S, limit(clip(qᵥ), dt, 2)),
        -triangle_inequality_limiter(abs(S), limit(clip(qₗ), dt, 2)),
    )
end
function cloud_sources(
    cm_params::CMP.CloudIce{FT},
    thp,
    qₜ,
    qₗ,
    qᵢ,
    qᵣ,
    qₛ,
    ρ,
    T,
    dt,
) where {FT}

    qᵥ = qₜ - qₗ - qᵢ - qᵣ - qₛ

    if qᵥ + qᵢ > FT(0)
        S = CMNe.conv_q_vap_to_q_liq_ice_MM2015(
            cm_params,
            thp,
            qₜ,
            qₗ,
            qᵢ,
            qᵣ,
            qₛ,
            ρ,
            T,
        )
    else
        S = FT(0)
    end

    return ifelse(
        S > FT(0),
        triangle_inequality_limiter(S, limit(clip(qᵥ), dt, 2)),
        -triangle_inequality_limiter(abs(S), limit(clip(qᵢ), dt, 2)),
    )
end

"""
    q_tot_0M_precipitation_sources(thp, cmp, dt, qₜ, ts)

 - thp, cmp - structs with thermodynamic and microphysics parameters
 - dt - model time step
 - qₜ - total water specific humidity
 - ts - thermodynamic state (see Thermodynamics.jl package for details)

Returns the qₜ source term due to precipitation formation
defined as Δm_tot / (m_dry + m_tot) for the 0-moment scheme
"""
function q_tot_0M_precipitation_sources(thp, cmp::CMP.Parameters0M, dt, qₜ, ts)
    return -triangle_inequality_limiter(
        -CM0.remove_precipitation(cmp, PP(thp, ts)),
        max(qₜ, 0) / float(dt),
    )
end

"""
    e_tot_0M_precipitation_sources_helper(thp, ts, Φ)

 - thp - set with thermodynamics parameters
 - ts - thermodynamic state (see td package for details)
 - Φ - geopotential

Returns the total energy source term multiplier from precipitation formation
for the 0-moment scheme
"""
function e_tot_0M_precipitation_sources_helper(thp, ts, Φ)

    λ = TD.liquid_fraction(thp, ts)
    Iₗ = TD.internal_energy_liquid(thp, ts)
    Iᵢ = TD.internal_energy_ice(thp, ts)

    return λ * Iₗ + (1 - λ) * Iᵢ + Φ
end

"""
    compute_precipitation_sources!(Sᵖ, Sᵖ_snow, Sqₗᵖ, Sqᵢᵖ, Sqᵣᵖ, Sqₛᵖ, ρ, qₜ, qₗ, qᵢ, qᵣ, qₛ, ts, dt, mp, thp)

 - Sᵖ, Sᵖ_snow - temporary containters to help compute precipitation source terms
 - Sqₗᵖ, Sqᵢᵖ, Sqᵣᵖ, Sqₛᵖ - cached storage for precipitation source terms
 - ρ - air density
 - qₜ, qₗ, qᵢ, qᵣ, qₛ - total water, liquid and ice, rain and snow specific humidity
 - ts - thermodynamic state (see td package for details)
 - dt - model time step
 - thp, cmp - structs with thermodynamic and microphysics parameters

Returns the q source terms due to precipitation formation from the 1-moment scheme.
The specific humidity source terms are defined as Δmᵢ / (m_dry + m_tot)
where i stands for total, rain or snow.
Also returns the total energy source term due to the microphysics processes.
"""
function compute_precipitation_sources!(
    Sᵖ,
    Sᵖ_snow,
    Sqₗᵖ,
    Sqᵢᵖ,
    Sqᵣᵖ,
    Sqₛᵖ,
    ρ,
    qₜ,
    qₗ,
    qᵢ,
    qᵣ,
    qₛ,
    ts,
    dt,
    mp,
    thp,
)
    FT = eltype(thp)
    @. Sqₗᵖ = FT(0)
    @. Sqᵢᵖ = FT(0)
    @. Sqᵣᵖ = FT(0)
    @. Sqₛᵖ = FT(0)

    #! format: off
    # rain autoconversion: q_liq -> q_rain
    @. Sᵖ = ifelse(
        mp.Ndp <= 0,
        CM1.conv_q_liq_to_q_rai(mp.pr.acnv1M, qₗ, true),
        CM2.conv_q_liq_to_q_rai(mp.var, qₗ, ρ, mp.Ndp),
    )
    @. Sᵖ = triangle_inequality_limiter(Sᵖ, limit(qₗ, dt, 5))
    @. Sqₗᵖ -= Sᵖ
    @. Sqᵣᵖ += Sᵖ

    # snow autoconversion assuming no supersaturation: q_ice -> q_snow
    @. Sᵖ = triangle_inequality_limiter(
        CM1.conv_q_ice_to_q_sno_no_supersat(mp.ps.acnv1M, qᵢ, true),
        limit(qᵢ, dt, 5),
    )
    @. Sqᵢᵖ -= Sᵖ
    @. Sqₛᵖ += Sᵖ

    # accretion: q_liq + q_rain -> q_rain
    @. Sᵖ = triangle_inequality_limiter(
        CM1.accretion(mp.cl, mp.pr, mp.tv.rain, mp.ce, qₗ, qᵣ, ρ),
        limit(qₗ, dt, 5),
    )
    @. Sqₗᵖ -= Sᵖ
    @. Sqᵣᵖ += Sᵖ

    # accretion: q_ice + q_snow -> q_snow
    @. Sᵖ = triangle_inequality_limiter(
        CM1.accretion(mp.ci, mp.ps, mp.tv.snow, mp.ce, qᵢ, qₛ, ρ),
        limit(qᵢ, dt, 5),
    )
    @. Sqᵢᵖ -= Sᵖ
    @. Sqₛᵖ += Sᵖ

    # accretion: q_liq + q_sno -> q_sno or q_rai
    # sink of cloud water via accretion cloud water + snow
    @. Sᵖ = triangle_inequality_limiter(
        CM1.accretion(mp.cl, mp.ps, mp.tv.snow, mp.ce, qₗ, qₛ, ρ),
        limit(qₗ, dt, 5),
    )
    # if T < T_freeze cloud droplets freeze to become snow
    # else the snow melts and both cloud water and snow become rain
    α(thp, ts) = TD.Parameters.cv_l(thp) / TD.latent_heat_fusion(thp, ts) * (Tₐ(thp, ts) - mp.ps.T_freeze)
    @. Sᵖ_snow = ifelse(
        Tₐ(thp, ts) < mp.ps.T_freeze,
        Sᵖ,
        FT(-1) * triangle_inequality_limiter(Sᵖ * α(thp, ts), limit(qₛ, dt, 5)),
    )
    @. Sqₛᵖ += Sᵖ_snow
    @. Sqₗᵖ -= Sᵖ
    @. Sqᵣᵖ += ifelse(Tₐ(thp, ts) < mp.ps.T_freeze, FT(0), Sᵖ - Sᵖ_snow)

    # accretion: q_ice + q_rai -> q_sno
    @. Sᵖ = triangle_inequality_limiter(
        CM1.accretion(mp.ci, mp.pr, mp.tv.rain, mp.ce, qᵢ, qᵣ, ρ),
        limit(qᵢ, dt, 5),
    )
    @. Sqᵢᵖ -= Sᵖ
    @. Sqₛᵖ += Sᵖ
    # sink of rain via accretion cloud ice - rain
    @. Sᵖ = triangle_inequality_limiter(
        CM1.accretion_rain_sink(mp.pr, mp.ci, mp.tv.rain, mp.ce, qᵢ, qᵣ, ρ),
        limit(qᵣ, dt, 5),
    )
    @. Sqᵣᵖ -= Sᵖ
    @. Sqₛᵖ += Sᵖ

    # accretion: q_rai + q_sno -> q_rai or q_sno
    @. Sᵖ = ifelse(
        Tₐ(thp, ts) < mp.ps.T_freeze,
        triangle_inequality_limiter(
            CM1.accretion_snow_rain(mp.ps, mp.pr, mp.tv.rain, mp.tv.snow, mp.ce, qₛ, qᵣ, ρ),
            limit(qᵣ, dt, 5),
        ),
        -triangle_inequality_limiter(
            CM1.accretion_snow_rain(mp.pr, mp.ps, mp.tv.snow, mp.tv.rain, mp.ce, qᵣ, qₛ, ρ),
            limit(qₛ, dt, 5),
        ),
    )
    @. Sqₛᵖ += Sᵖ
    @. Sqᵣᵖ -= Sᵖ
    #! format: on
end

"""
    compute_precipitation_sinks!(Sᵖ, Sqᵣᵖ, Sqₛᵖ, ρ, qₜ, qₗ, qᵢ, qᵣ, qₛ, ts, dt, mp, thp)

 - Sᵖ - a temporary containter to help compute precipitation source terms
 - Sqᵣᵖ, Sqₛᵖ - cached storage for precipitation source terms
 - ρ - air density
 - qₜ, qₗ, qᵢ, qᵣ, qₛ - total water, cloud liquid and ice, rain and snow specific humidities
 - ts - thermodynamic state (see td package for details)
 - dt - model time step
 - thp, cmp - structs with thermodynamic and microphysics parameters

Returns the q source terms due to precipitation sinks from the 1-moment scheme.
The specific humidity source terms are defined as Δmᵢ / (m_dry + m_tot)
where i stands for total, rain or snow.
Also returns the total energy source term due to the microphysics processes.
"""
function compute_precipitation_sinks!(
    Sᵖ,
    Sqᵣᵖ,
    Sqₛᵖ,
    ρ,
    qₜ,
    qₗ,
    qᵢ,
    qᵣ,
    qₛ,
    ts,
    dt,
    mp,
    thp,
)
    FT = eltype(thp)
    sps = (mp.ps, mp.tv.snow, mp.aps, thp)
    rps = (mp.pr, mp.tv.rain, mp.aps, thp)

    #! format: off
    # evaporation: q_rai -> q_vap
    @. Sᵖ = -triangle_inequality_limiter(
        -CM1.evaporation_sublimation(rps..., qₜ, qₗ, qᵢ, qᵣ, qₛ, ρ, Tₐ(thp, ts)),
        limit(qᵣ, dt, 5),
    )
    @. Sqᵣᵖ += Sᵖ

    # melting: q_sno -> q_rai
    @. Sᵖ = triangle_inequality_limiter(
        CM1.snow_melt(sps..., qₛ, ρ, Tₐ(thp, ts)),
        limit(qₛ, dt, 5),
    )
    @. Sqᵣᵖ += Sᵖ
    @. Sqₛᵖ -= Sᵖ

    # deposition/sublimation: q_vap <-> q_sno
    @. Sᵖ = CM1.evaporation_sublimation(sps..., qₜ, qₗ, qᵢ, qᵣ, qₛ, ρ, Tₐ(thp, ts))
    @. Sᵖ = ifelse(
        Sᵖ > FT(0),
        triangle_inequality_limiter(Sᵖ, limit(qᵥ(thp, ts), dt, 5)),
        -triangle_inequality_limiter(FT(-1) * Sᵖ, limit(qₛ, dt, 5)),
    )
    @. Sqₛᵖ += Sᵖ
    #! format: on
end

#####
##### 2M microphysics
#####

"""
    aerosol_activation_sources(cm_params, thp, ρ, Tₐ, qₜ, qₗ, qᵢ, qᵣ, qₛ, n_dp, n_dp_prescribed, dt)

 - cm_params - CloudMicrophysics parameters struct for cloud water or ice condensate
 - thp - Thermodynamics parameters struct
 - ρ - air density
 - Tₐ - air temperature
 - qₜ - total specific humidity
 - qₗ - liquid specific humidity
 - qᵢ - ice specific humidity
 - qᵣ - rain specific humidity
 - qₛ - snow specific humidity
 - n_dp - number concentration droplets (liquid or ice) per mass
 _ n_dp_prescribed - prescribed number concentration of droplets (liquid or ice) per mass
 - dt - model time step

Returns the activation rate. #TODO This function temporarily computes activation rate
based on mass rates and a prescribed droplet mass (no activation parameterization yet).
"""
function aerosol_activation_sources(
    cm_params::CMP.CloudLiquid{FT},
    thp,
    ρ,
    Tₐ,
    qₜ,
    qₗ,
    qᵢ,
    qᵣ,
    qₛ,
    n_dp,
    n_dp_prescribed,
    dt,
) where {FT}
    r_dp = FT(2e-6) # 2 μm
    m_dp = 4 / 3 * FT(π) * r_dp^3 * cm_params.ρw
    Sn = cloud_sources(cm_params, thp, qₜ, qₗ, qᵢ, qᵣ, qₛ, ρ, Tₐ, dt) / m_dp

    return ifelse(
        Sn > FT(0),
        triangle_inequality_limiter(Sn, limit((n_dp_prescribed - n_dp), dt, 2)),
        -triangle_inequality_limiter(abs(Sn), limit(n_dp, dt, 2)),
    )
end

"""
    compute_warm_precipitation_sources_2M!(Sᵖ, S₂ᵖ, Snₗᵖ, Snᵣᵖ, Sqₗᵖ, Sqᵣᵖ, ρ, nₗ, nᵣ, qₗ, qᵣ, ts, dt, sb, thp)

 - Sᵖ, S₂ᵖ - temporary containters to help compute precipitation source terms
 - Snₗᵖ, Snᵣᵖ, Sqₗᵖ, Sqᵣᵖ - cached storage for precipitation source terms
 - ρ - air density
 - nₗ, nᵣ - cloud liquid and rain number concentration per mass [1 / kg of moist air]
 - qₗ, qᵣ - cloud liquid and rain specific humidity
 - ts - thermodynamic state (see td package for details)
 - dt - model time step
 - thp, mp - structs with thermodynamic and microphysics parameters

Computes precipitation number and mass sources due to warm precipitation processes based on the 2-moment
[Seifert and Beheng (2006) scheme](https://clima.github.io/CloudMicrophysics.jl/dev/Microphysics2M/).
"""
function compute_warm_precipitation_sources_2M!(
    Sᵖ,
    S₂ᵖ,
    Snₗᵖ,
    Snᵣᵖ,
    Sqₗᵖ,
    Sqᵣᵖ,
    ρ,
    nₗ,
    nᵣ,
    qₗ,
    qᵣ,
    ts,
    dt,
    mp,
    thp,
)

    FT = eltype(thp)
    @. Snₗᵖ = FT(0)
    @. Snᵣᵖ = FT(0)
    @. Sqₗᵖ = FT(0)
    @. Sqᵣᵖ = FT(0)

    # auto-conversion (mass)
    @. Sᵖ = triangle_inequality_limiter(
        CM2.autoconversion(
            mp.sb.acnv,
            mp.sb.pdf_c,
            qₗ,
            qᵣ,
            ρ,
            ρ * nₗ,
        ).dq_rai_dt,
        limit(qₗ, dt, 5), # cap rate to at most 20% of qₗ per timestep to ensure stability
    )
    @. Sqₗᵖ -= Sᵖ
    @. Sqᵣᵖ += Sᵖ

    # auto-conversion (number) and liquid self-collection
    @. Sᵖ = triangle_inequality_limiter(
        CM2.autoconversion(
            mp.sb.acnv,
            mp.sb.pdf_c,
            qₗ,
            qᵣ,
            ρ,
            ρ * nₗ,
        ).dN_liq_dt / ρ,
        limit(nₗ, dt, 10),
    )
    # triangle_inequality_limiter assumes positive rates and limits.
    # Here the physical rate is negative (a sink), so we negate it before passing,
    # and negate the result again to preserve the original sign.
    @. S₂ᵖ =
        -triangle_inequality_limiter(
            -CM2.liquid_self_collection(mp.sb.acnv, mp.sb.pdf_c, qₗ, ρ, Sᵖ) / ρ,
            limit(nₗ / ρ, dt, 5),
        )
    @. Snₗᵖ += Sᵖ
    @. Snₗᵖ += S₂ᵖ
    @. Snᵣᵖ -= 0.5 * Sᵖ # each raindrop forms from two cloud particles → factor 0.5

    # rain self-collection and breakup
    @. Sᵖ =
        -triangle_inequality_limiter(
            -CM2.rain_self_collection(mp.sb.pdf_r, mp.sb.self, qᵣ, ρ, ρ * nᵣ) /
            ρ,
            limit(nᵣ, dt, 5),
        )
    @. S₂ᵖ = triangle_inequality_limiter(
        CM2.rain_breakup(mp.sb.pdf_r, mp.sb.brek, qᵣ, ρ, ρ * nᵣ, Sᵖ) / ρ,
        limit(nᵣ, dt, 5),
    )
    @. Snᵣᵖ += Sᵖ
    @. Snᵣᵖ += S₂ᵖ

    # accretion (mass)
    @. Sᵖ = triangle_inequality_limiter(
        CM2.accretion(mp.sb, qₗ, qᵣ, ρ, ρ * nₗ).dq_rai_dt,
        limit(qₗ, dt, 5),
    )
    @. Sqₗᵖ -= Sᵖ
    @. Sqᵣᵖ += Sᵖ

    # accretion (number)
    @. Sᵖ =
        -triangle_inequality_limiter(
            -CM2.accretion(mp.sb, qₗ, qᵣ, ρ, ρ * nₗ).dN_liq_dt / ρ,
            limit(nₗ, dt, 5),
        )
    @. Snₗᵖ += Sᵖ

    # evaporation (mass)
    @. Sᵖ =
        -triangle_inequality_limiter(
            -CM2.rain_evaporation(
                mp.sb,
                mp.aps,
                thp,
                PP(thp, ts),
                qᵣ,
                ρ,
                ρ * nᵣ,
                Tₐ(thp, ts),
            ).evap_rate_1,
            limit(qᵣ, dt, 5),
        )
    @. Sqᵣᵖ += Sᵖ

    # evaporation (number)
    @. Sᵖ =
        -triangle_inequality_limiter(
            -CM2.rain_evaporation(
                mp.sb,
                mp.aps,
                thp,
                PP(thp, ts),
                qᵣ,
                ρ,
                ρ * nᵣ,
                Tₐ(thp, ts),
            ).evap_rate_0 / ρ,
            limit(nᵣ, dt, 5),
        )
    @. Snᵣᵖ += Sᵖ

    # cloud liquid number adjustment for mass limits
    # TODO: Once CCN number becomes a prognostic variable, these number adjustment tendencies
    #       should be linked to it. Any increase in droplet number (source here) would imply 
    #       a corresponding sink in CCN, and vice versa.
    @. Sᵖ = CM2.number_increase_for_mass_limit(
        mp.sb.numadj,
        mp.sb.pdf_c.xc_max,
        qₗ,
        ρ,
        ρ * nₗ,
    )
    @. S₂ᵖ =
        -triangle_inequality_limiter(
            -CM2.number_decrease_for_mass_limit(
                mp.sb.numadj,
                mp.sb.pdf_c.xc_min,
                qₗ,
                ρ,
                ρ * nₗ,
            ),
            limit(nₗ, dt, 5),
        )
    @. Snₗᵖ = Sᵖ + S₂ᵖ
    # rain number adjustment for mass limits
    @. Sᵖ = CM2.number_increase_for_mass_limit(
        mp.sb.numadj,
        mp.sb.pdf_r.xr_max,
        qᵣ,
        ρ,
        ρ * nᵣ,
    )
    @. S₂ᵖ =
        -triangle_inequality_limiter(
            -CM2.number_decrease_for_mass_limit(
                mp.sb.numadj,
                mp.sb.pdf_r.xr_min,
                qᵣ,
                ρ,
                ρ * nᵣ,
            ),
            limit(nᵣ, dt, 5),
        )
    @. Snᵣᵖ += Sᵖ + S₂ᵖ

end
