#=
0-moment microphysics: the equilibrium condensate (q_liq + q_ice) is removed as
precipitation over a relaxation time τ, adding sinks to ρq_tot and ρe. The
condensate partition is closed-form (condensate_partition on the dynamics T),
never a saturation-adjustment Newton solve — so it cannot throw on the implicit
solver's transient iterates. All explicit: moisture is not in the implicit
acoustic subsystem, so the HEVI split rhs = implicit + remaining is preserved.
=#

# Condensate specific humidity q_liq + q_ice at (T, ρ, q_tot).
@inline function q_condensate(thermo_params, T, ρ, q_tot)
    (q_liq, q_ice) = TD.condensate_partition(thermo_params, T, ρ, q_tot)
    return q_liq + q_ice
end

# Specific internal energy of the removed condensate (liquid/ice weighted).
@inline function condensate_energy(thermo_params, T, ρ, q_tot)
    (q_liq, q_ice) = TD.condensate_partition(thermo_params, T, ρ, q_tot)
    q_c = q_liq + q_ice
    f_liq = q_c > 0 ? q_liq / q_c : one(q_c)
    return f_liq * TD.internal_energy_liquid(thermo_params, T) +
           (1 - f_liq) * TD.internal_energy_ice(thermo_params, T)
end

"""
    microphysics_0m_tendency!(dYc, ρ, ᶜΦ, q_tot, T_air, m)

Add the 0-moment precipitation sinks to `dYc.ρq_tot` and `dYc.ρe`, using the
dynamics temperature `T_air` and total specific humidity `q_tot = ρq_tot/ρ`.
"""
function microphysics_0m_tendency!(dYc, ρ, ᶜΦ, q_tot, T_air, m::DGModel)
    tp = m.fields.thermo_params
    τ = m.prob.precip_timescale
    ᶜS_qt = @. -(q_condensate(tp, T_air, ρ, q_tot) / τ)      # ≤ 0 [1/s]
    ᶜI_c = @. condensate_energy(tp, T_air, ρ, q_tot)
    @. dYc.ρq_tot += ρ * ᶜS_qt
    @. dYc.ρe += ρ * ᶜS_qt * (ᶜI_c + ᶜΦ)
    return dYc
end
