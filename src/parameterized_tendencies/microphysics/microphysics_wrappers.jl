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
qₜ(thp, ts) = TD.PhasePartition(thp, ts).tot

# Get q_liq and q_ice out of phase partition
function qₗ(thp, ts, qᵣ)
    FT = eltype(ts)
    return max(FT(0), TD.PhasePartition(thp, ts).liq - qᵣ)
end
function qᵢ(thp, ts, qₛ)
    FT = eltype(ts)
    return max(FT(0), TD.PhasePartition(thp, ts).ice - qₛ)
end

# Clip precipitation to avoid negative numbers
function qₚ(q_rain_snow)
    FT = eltype(q_rain_snow)
    return max(FT(0), q_rain_snow)
end

# Helper function to limit the tendency
function limit(q, dt, n::Int)
    return q / float(dt) / n
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
    cloud_sources(cm_params, thp, ts, dt)

 - cm_params - CloudMicrophysics parameters struct for cloud water or ice condensate
 - thp - Thermodynamics parameters struct
 - ts - thermodynamics state
 - dt - model time step

Returns the condensation/evaporation or deposition/sublimation rate for
non-equilibrium Morrison and Milbrandt 2015 cloud formation.
"""
function cloud_sources(
    cm_params::CMP.CloudLiquid{FT},
    thp,
    ts,
    qᵣ,
    dt,
) where {FT}

    q = TD.PhasePartition(thp, ts)
    ρ = TD.air_density(thp, ts)

    S = CMNe.conv_q_vap_to_q_liq_ice_MM2015(cm_params, thp, q, ρ, Tₐ(thp, ts))

    # keeping the same limiter for now
    return ifelse(
        S > FT(0),
        min(S, limit(qᵥ(thp, ts), dt, 2)),
        -min(abs(S), limit(qₗ(thp, ts, qₚ(qᵣ)), dt, 2)),
    )
end
function cloud_sources(cm_params::CMP.CloudIce{FT}, thp, ts, qₛ, dt) where {FT}

    q = TD.PhasePartition(thp, ts)
    ρ = TD.air_density(thp, ts)

    S = CMNe.conv_q_vap_to_q_liq_ice_MM2015(cm_params, thp, q, ρ, Tₐ(thp, ts))

    # keeping the same limiter for now
    return ifelse(
        S > FT(0),
        min(S, limit(qᵥ(thp, ts), dt, 2)),
        -min(abs(S), limit(qᵢ(thp, ts, qₚ(qₛ)), dt, 2)),
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
    return -min(
        max(qₜ, 0) / float(dt),
        -CM0.remove_precipitation(cmp, PP(thp, ts)),
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
    compute_precipitation_sources!(Sᵖ, Sᵖ_snow, Sqₗᵖ, Sqᵢᵖ, Sqᵣᵖ, Sqₛᵖ, ρ, qᵣ, qₛ, ts, dt, mp, thp)

 - Sᵖ, Sᵖ_snow - temporary containters to help compute precipitation source terms
 - Sqₗᵖ, Sqᵢᵖ, Sqᵣᵖ, Sqₛᵖ - cached storage for precipitation source terms
 - ρ - air density
 - qᵣ, qₛ - precipitation (rain and snow) specific humidity
 - ts - thermodynamic state (see td package for details)
 - dt - model time step
 - thp, cmp - structs with thermodynamic and microphysics parameters

Returns the q source terms due to precipitation formation from the 1-moment scheme.
The specific humidity source terms are defined as defined as Δmᵢ / (m_dry + m_tot)
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
    qᵣ,
    qₛ,
    ts,
    dt,
    mp,
    thp,
    tmp_accr_sno_ice,
    tmp_accr_rai_liq,
    tmp_acnv_ice_sno,
    tmp_acnv_liq_rai,
    tmp_accr_sno_liq_sno_part,
    tmp_accr_sno_liq_liq_part,
    tmp_accr_rai_ice_sno_part,
    tmp_accr_rai_ice_rai_part,
    tmp_accr_rai_sno, #9
)
    FT = eltype(thp)
    # @. Sqₜᵖ = FT(0) should work after fixing
    # https://github.com/CliMA/ClimaCore.jl/issues/1786
    @. Sqₗᵖ = ρ * FT(0)
    @. Sqᵢᵖ = ρ * FT(0)
    @. Sqᵣᵖ = ρ * FT(0)
    @. Sqₛᵖ = ρ * FT(0)

    #! format: off
    # rain autoconversion: q_liq -> q_rain
    @. Sᵖ = ifelse(
        mp.Ndp <= 0,
        CM1.conv_q_liq_to_q_rai(mp.pr.acnv1M, qₗ(thp, ts, qₚ(qᵣ)), true),
        CM2.conv_q_liq_to_q_rai(mp.var, qₗ(thp, ts, qₚ(qᵣ)), ρ, mp.Ndp),
    )
    @. Sᵖ = min(limit(qₗ(thp, ts, qₚ(qᵣ)), dt, 5), Sᵖ)
    @. Sqₗᵖ -= Sᵖ
    @. Sqᵣᵖ += Sᵖ
    @. tmp_acnv_liq_rai = Sᵖ

    # snow autoconversion assuming no supersaturation: q_ice -> q_snow
    @. Sᵖ = min(
        limit(qᵢ(thp, ts, qₚ(qₛ)), dt, 5),
        CM1.conv_q_ice_to_q_sno_no_supersat(mp.ps.acnv1M, qᵢ(thp, ts, qₚ(qₛ)), true),
    )
    @. Sqᵢᵖ -= Sᵖ
    @. Sqₛᵖ += Sᵖ
    @. tmp_acnv_ice_sno = Sᵖ

    # accretion: q_liq + q_rain -> q_rain
    @. Sᵖ = min(
        limit(qₗ(thp, ts, qₚ(qᵣ)), dt, 5),
        CM1.accretion(mp.cl, mp.pr, mp.tv.rain, mp.ce, qₗ(thp, ts, qₚ(qᵣ)), qₚ(qᵣ), ρ),
    )
    @. Sqₗᵖ -= Sᵖ
    @. Sqᵣᵖ += Sᵖ
    @. tmp_accr_rai_liq = Sᵖ

    # accretion: q_ice + q_snow -> q_snow
    @. Sᵖ = min(
        limit(qᵢ(thp, ts, qₚ(qₛ)), dt, 5),
        CM1.accretion(mp.ci, mp.ps, mp.tv.snow, mp.ce, qᵢ(thp, ts, qₚ(qₛ)), qₚ(qₛ), ρ),
    )
    @. Sqᵢᵖ -= Sᵖ
    @. Sqₛᵖ += Sᵖ
    @. tmp_accr_sno_ice = Sᵖ

    # accretion: q_liq + q_sno -> q_sno or q_rai
    # sink of cloud water via accretion cloud water + snow
    @. Sᵖ = min(
        limit(qₗ(thp, ts, qₚ(qᵣ)), dt, 5),
        CM1.accretion(mp.cl, mp.ps, mp.tv.snow, mp.ce, qₗ(thp, ts, qₚ(qᵣ)), qₚ(qₛ), ρ),
    )
    # if T < T_freeze cloud droplets freeze to become snow
    # else the snow melts and both cloud water and snow become rain
    α(thp, ts) = TD.Parameters.cv_l(thp) / TD.latent_heat_fusion(thp, ts) * (Tₐ(thp, ts) - mp.ps.T_freeze)
    @. Sᵖ_snow = ifelse(
        Tₐ(thp, ts) < mp.ps.T_freeze,
        Sᵖ,
        FT(-1) * min(Sᵖ * α(thp, ts), limit(qₚ(qₛ), dt, 5)),
    )
    @. Sqₛᵖ += Sᵖ_snow
    @. Sqₗᵖ -= Sᵖ
    @. Sqᵣᵖ += ifelse(Tₐ(thp, ts) < mp.ps.T_freeze, FT(0), Sᵖ - Sᵖ_snow)
    @. tmp_accr_sno_liq_sno_part = Sᵖ_snow
    @. tmp_accr_sno_liq_liq_part = Sᵖ

    # accretion: q_ice + q_rai -> q_sno
    @. Sᵖ = min(
        limit(qᵢ(thp, ts, qₚ(qₛ)), dt, 5),
        CM1.accretion(mp.ci, mp.pr, mp.tv.rain, mp.ce, qᵢ(thp, ts, qₚ(qₛ)), qₚ(qᵣ), ρ),
    )
    @. Sqᵢᵖ -= Sᵖ
    @. Sqₛᵖ += Sᵖ
    @. tmp_accr_rai_ice_sno_part = Sᵖ
    # sink of rain via accretion cloud ice - rain
    @. Sᵖ = min(
        limit(qₚ(qᵣ), dt, 5),
        CM1.accretion_rain_sink(mp.pr, mp.ci, mp.tv.rain, mp.ce, qᵢ(thp, ts, qₚ(qₛ)), qₚ(qᵣ), ρ),
    )
    @. Sqᵣᵖ -= Sᵖ
    @. Sqₛᵖ += Sᵖ
    @. tmp_accr_rai_ice_rai_part = Sᵖ

    # accretion: q_rai + q_sno -> q_rai or q_sno
    @. Sᵖ = ifelse(
        Tₐ(thp, ts) < mp.ps.T_freeze,
        min(
            limit(qₚ(qᵣ), dt, 5),
            CM1.accretion_snow_rain(mp.ps, mp.pr, mp.tv.rain, mp.tv.snow, mp.ce, qₚ(qₛ), qₚ(qᵣ), ρ),
        ),
        -min(
            limit(qₚ(qₛ), dt, 5),
            CM1.accretion_snow_rain(mp.pr, mp.ps, mp.tv.snow, mp.tv.rain, mp.ce, qₚ(qᵣ), qₚ(qₛ), ρ),
        ),
    )
    @. Sqₛᵖ += Sᵖ
    @. Sqᵣᵖ -= Sᵖ
    @. tmp_accr_rai_sno = Sᵖ
    #! format: on
end

"""
    compute_precipitation_sinks!(Sᵖ, Sqᵣᵖ, Sqₛᵖ, ρ, qᵣ, qₛ, ts, dt, mp, thp)

 - Sᵖ - a temporary containter to help compute precipitation source terms
 - Sqᵣᵖ, Sqₛᵖ - cached storage for precipitation source terms
 - ρ - air density
 - qᵣ, qₛ - precipitation (rain and snow) specific humidities
 - ts - thermodynamic state (see td package for details)
 - dt - model time step
 - thp, cmp - structs with thermodynamic and microphysics parameters

Returns the q source terms due to precipitation sinks from the 1-moment scheme.
The specific humidity source terms are defined as defined as Δmᵢ / (m_dry + m_tot)
where i stands for total, rain or snow.
Also returns the total energy source term due to the microphysics processes.
"""
function compute_precipitation_sinks!(
    Sᵖ,
    Sqᵣᵖ,
    Sqₛᵖ,
    ρ,
    qᵣ,
    qₛ,
    ts,
    dt,
    mp,
    thp,
    tmp_evap,
    tmp_melt,
    tmp_dep_sub, #3
)
    FT = eltype(thp)
    sps = (mp.ps, mp.tv.snow, mp.aps, thp)
    rps = (mp.pr, mp.tv.rain, mp.aps, thp)

    #! format: off
    # evaporation: q_rai -> q_vap
    @. Sᵖ = -min(
        limit(qₚ(qᵣ), dt, 5),
        -CM1.evaporation_sublimation(rps..., PP(thp, ts), qₚ(qᵣ), ρ, Tₐ(thp, ts)),
    )
    @. Sqᵣᵖ += Sᵖ
    @. tmp_evap = Sᵖ

    # melting: q_sno -> q_rai
    @. Sᵖ = min(
        limit(qₚ(qₛ), dt, 5),
        CM1.snow_melt(sps..., qₚ(qₛ), ρ, Tₐ(thp, ts)),
    )
    @. Sqᵣᵖ += Sᵖ
    @. Sqₛᵖ -= Sᵖ
    @. tmp_melt = Sᵖ

    # deposition/sublimation: q_vap <-> q_sno
    @. Sᵖ = CM1.evaporation_sublimation(sps..., PP(thp, ts), qₚ(qₛ), ρ, Tₐ(thp, ts))
    @. Sᵖ = ifelse(
        Sᵖ > FT(0),
        min(limit(qᵥ(thp, ts), dt, 5), Sᵖ),
        -min(limit(qₚ(qₛ), dt, 5), FT(-1) * Sᵖ),
    )
    @. Sqₛᵖ += Sᵖ
    @. tmp_dep_sub = Sᵖ
    #! format: on
end
