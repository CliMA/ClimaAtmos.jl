# A set of wrappers for using CloudMicrophysics.jl functions inside EDMFX loops

import Thermodynamics as TD
import CloudMicrophysics.Microphysics0M as CM0

"""
    q_tot_precipitation_sources(precip_model, t_prs, m_prs, dt, q_tot, ts)

 - precip_model - a type for precipitation scheme choice
 - t_prs, m_prs - stricts with thermodynamic and microphysics parameters
 - dt - model time step
 - q_tot - total water specific humidity
 - ts - thermodynamic state (see td package for details)

Returns the q_tot source term due to precipitation formation
defined as Δm_tot / (m_dry + m_tot)
"""
function q_tot_precipitation_sources(
    ::NoPrecipitation,
    t_prs,
    m_prs,
    dt,
    q_tot::FT,
    ts,
) where {FT <: Real}
    return FT(0)
end
function q_tot_precipitation_sources(
    ::Microphysics0Moment,
    t_prs,
    m_prs,
    dt,
    q_tot::FT,
    ts,
) where {FT <: Real}
    return -min(
        max(q_tot, 0) / dt,
        -CM0.remove_precipitation(
            m_prs,
            TD.PhasePartition(t_prs, ts),
            TD.q_vap_saturation(t_prs, ts),
        ),
    )
end

"""
    e_tot_0M_precipitation_sources_helper(t_prs, ts, Φ)

 - t_prs - set with thermodynamics parameters
 - ts - thermodynamic state (see td package for details)
 - Φ - geopotential

Returns the total energy source term multiplier from precipitation formation
"""
function e_tot_0M_precipitation_sources_helper(t_prs, ts, Φ)
    λ = TD.liquid_fraction(t_prs, ts)
    I_l = TD.internal_energy_liquid(t_prs, ts)
    I_i = TD.internal_energy_ice(t_prs, ts)
    return λ * I_l + (1 - λ) * I_i + Φ
end
