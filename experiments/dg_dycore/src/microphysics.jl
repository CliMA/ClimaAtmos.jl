#=
0-moment ("bulk") microphysics for the moist FDDG core.

Semantics match CloudMicrophysics.jl `Microphysics0M` — the supersaturation
excess is removed as precipitation over a relaxation timescale τ — but the rate
is evaluated directly from Thermodynamics.jl saturation adjustment, so no
CloudMicrophysics dependency is added. The equilibrium condensate q_liq + q_ice
(from the SAME saturation adjustment the moist EOS uses) is relaxed to zero:

    S_qtot = −(q_liq + q_ice) / τ            [kg/kg/s], ≤ 0
    S_ρe   = ρ S_qtot · (Iₗ(T) + Φ)          removed water carries internal
                                             + potential energy (warm-rain
                                             approximation: all liquid)

Both are grid-mean sources added to the EXPLICIT tendency (moisture is never in
the implicit acoustic subsystem), so the HEVI split rhs = implicit + remaining
is preserved.
=#

# Equilibrium (T, condensate) from a density–internal-energy saturation
# adjustment. Split into two scalar accessors so each can be broadcast into a
# ClimaCore field (a struct-returning kernel is awkward under `@.`). NOTE: this
# calls saturation_adjustment twice per point per stage — a known perf item;
# the moist EOS already runs one adjustment and the two could be fused later.
@inline function sa_q_cond(thermo_params, ρ, e_int, q_tot)
    sa = TD.saturation_adjustment(thermo_params, TD.ρe(), ρ, e_int, q_tot)
    return sa.q_liq + sa.q_ice
end

@inline function sa_T(thermo_params, ρ, e_int, q_tot)
    sa = TD.saturation_adjustment(thermo_params, TD.ρe(), ρ, e_int, q_tot)
    return sa.T
end

"""
    microphysics_0m_tendency!(dYc, ρ, ρe, K, ᶜΦ, q_tot, m)

Add the 0-moment precipitation sinks to `dYc.ρq_tot` and `dYc.ρe`. `q_tot` is
the total specific humidity (ρq_tot/ρ); `K`, `ᶜΦ` the kinetic energy and
geopotential used to recover `e_int = ρe/ρ − K − Φ`.
"""
function microphysics_0m_tendency!(
    dYc,
    ρ,
    ρe,
    K,
    ᶜΦ,
    q_tot,
    m::DGModel{FT},
) where {FT}
    tp = m.fields.thermo_params
    τ = m.prob.precip_timescale
    e_int = @. ρe / ρ - K - ᶜΦ
    ᶜq_cond = @. sa_q_cond(tp, ρ, e_int, q_tot)
    ᶜT = @. sa_T(tp, ρ, e_int, q_tot)
    ᶜS_qt = @. -(ᶜq_cond / τ)                                # ≤ 0 [1/s]
    @. dYc.ρq_tot += ρ * ᶜS_qt
    @. dYc.ρe += ρ * ᶜS_qt * (TD.internal_energy_liquid(tp, ᶜT) + ᶜΦ)
    return dYc
end
