# ============================================================================
# Tendency Limiters for Microphysics
# ============================================================================
# Functions for limiting source/sink terms to prevent numerical instabilities
# and unphysical negative values in water species.

"""
    limit(q, dt, n::Int)

Compute the maximum safe tendency for depleting a quantity `q` over `n` timesteps.

Used to determine the maximum rate at which a source category can be depleted
without going negative, accounting for multiple simultaneous sinks.

# Arguments
- `q`: Available quantity (e.g., specific humidity) [kg/kg]
- `dt`: Model timestep [s]
- `n::Int`: Number of sinks splitting the available quantity

# Returns
`q / (dt × n)` — the maximum tendency [kg/kg/s] that can be applied.

# Example
```julia
# Rain has 3 sinks (evaporation, accretion, self-collection)
# Each sink can at most consume 1/3 of available rain per timestep
max_rate = limit(q_rai, dt, 3)
```
"""
@inline function limit(q, dt, n::Int)
    return max(zero(q), q) / dt / n
end

"""
    limit_sink(S, q, dt, n=3)

Limit a sink tendency to prevent species depletion.
Only applies limiting when `S < 0` (sink), sources (`S ≥ 0`) pass unchanged.

# Arguments
- `S`: Raw tendency (source or sink) [kg/kg/s]
- `q`: Available quantity [kg/kg]
- `dt`: Timestep [s]
- `n`: Number of competing sinks (default: 3)

# Returns
Limited tendency:
- If `S < 0`: `-min(-S, limit(q, dt, n))`
- If `S ≥ 0`: `S` (unchanged)

# Example
```julia
# Limit rain evaporation to available rain
S_evap_limited = limit_sink(S_evap, q_rain, dt, 3)
```
"""
@inline function limit_sink(S, q, dt, n = 3)
    return ifelse(S < zero(S), -min(-S, limit(q, dt, n)), S)
end

"""
    tendency_limiter(tendency, tend_bound_pos, tend_bound_neg)

Limits a `tendency` to be within `[tend_bound_neg, tend_bound_pos]`.

- If `tendency > 0`: limited by `min(tendency, tend_bound_pos)`
- If `tendency < 0`: limited by `-min(-tendency, tend_bound_neg)`

This ensures that sources do not exceed `tend_bound_pos` and sinks do not
exceed `tend_bound_neg` (magnitude-wise).

Used by `limit_1m_tendencies` and `limit_2m_tendencies` to prevent
negative tracer concentrations.
"""
@inline function tendency_limiter(
    tendency,
    tend_bound_pos,
    tend_bound_neg,
)

    # Ensure bounds are non-negative
    tend_bound_pos = max(zero(tend_bound_pos), tend_bound_pos)
    tend_bound_neg = max(zero(tend_bound_neg), tend_bound_neg)

    # Positive tendency (source): limit by tend_bound_pos
    limited_pos = min(tendency, tend_bound_pos)

    # Negative tendency (sink): limit by tend_bound_neg (which is positive scalar)
    limited_neg = -min(-tendency, tend_bound_neg)

    # Branchless selection
    return ifelse(tendency >= zero(tendency), limited_pos, limited_neg)
end

"""
    coupled_sink_limit_factor(S1, S2, q1, q2, dt, n=3)

Compute a uniform scaling factor for two coupled sink tendencies.

For processes where two species are simultaneously depleted (e.g., autoconversion
depletes both cloud liquid and number), compute a single scaling factor based on
the most restrictive constraint.

# Arguments
- `S1`, `S2`: Raw sink tendencies (must be ≤ 0) [kg/kg/s or #/kg/s]
- `q1`, `q2`: Available quantities [kg/kg or #/kg]
- `dt`: Timestep [s]
- `n`: Number of competing sinks (default: 3)

# Returns
Scaling factor `f ∈ [0, 1]` such that:
- `|S1 * f| ≤ q1/(dt*n)` and `|S2 * f| ≤ q2/(dt*n)`
- `f = min(M1/|S1|, M2/|S2|)` when both are sinks
- `f = 1` if either tendency is not a sink

# Example
```julia
# Autoconversion depletes both q_liq and n_liq
f = coupled_sink_limit_factor(
    dq_liq_auto, dn_liq_auto, q_liq, n_liq, dt,
)
dq_liq_auto *= f
dn_liq_auto *= f
```
"""
@inline function coupled_sink_limit_factor(S1, S2, q1, q2, dt, n = 3)
    M1 = limit(q1, dt, n)
    M2 = limit(q2, dt, n)

    # Compute individual scaling factors (only for sinks)
    f1 = ifelse(S1 < zero(S1) && -S1 > M1, M1 / (-S1), one(S1))
    f2 = ifelse(S2 < zero(S2) && -S2 > M2, M2 / (-S2), one(S2))

    # Take most restrictive
    return min(f1, f2)
end

# ============================================================================
# 0M Tendency Limiting
# ============================================================================
"""
    apply_0m_tendency_limit(dq_tot_dt, q_tot, dt)

Apply a limiter to 0M microphysics total water sink.
"""
@inline function apply_0m_tendency_limit(dq_tot_dt, q_tot, dt)
    return limit_sink(dq_tot_dt, q_tot, dt)
end

# ============================================================================
# 1M Tendency Limiting
# ============================================================================
# For 1M microphysics we use average bulk tendencies over dt from CloudMicrophysics
# which preserves positivity of specific humidities.

# ============================================================================
# 2M Tendency Limiting
# ============================================================================

"""
    apply_2m_tendency_limits!(ᶜmp_tendency, timestepping, ᶜq_lcl, ᶜn_lcl, ᶜq_rai, ᶜn_rai, dt)

Apply physical limiting to 2M microphysics tendencies in-place.

No-op for implicit timestepping as the Jacobian handles stability.
"""
@inline apply_2m_tendency_limits!(ᶜmp_tendency, ::Implicit, args...) = nothing
@inline function apply_2m_tendency_limits!(
    ᶜmp_tendency, ::Explicit, ᶜq_lcl, ᶜn_lcl, ᶜq_rai, ᶜn_rai, dt,
)
    @. ᶜmp_tendency = _explicit_2m_tendency_limits(
        ᶜmp_tendency, ᶜq_lcl, ᶜn_lcl, ᶜq_rai, ᶜn_rai, dt,
    )
end
@inline apply_2m_tendency_limits!(ᶜmp_tendency, ::Nothing, args...) = nothing

@inline function _explicit_2m_tendency_limits(mp_tendency, q_liq, n_liq, q_rai, n_rai, dt)
    f_liq = coupled_sink_limit_factor(
        mp_tendency.dq_lcl_dt, mp_tendency.dn_lcl_dt, q_liq, n_liq, dt,
    )
    f_rai = coupled_sink_limit_factor(
        mp_tendency.dq_rai_dt, mp_tendency.dn_rai_dt, q_rai, n_rai, dt,
    )

    return (
        dq_lcl_dt = mp_tendency.dq_lcl_dt * f_liq,
        dn_lcl_dt = mp_tendency.dn_lcl_dt * f_liq,
        dq_rai_dt = mp_tendency.dq_rai_dt * f_rai,
        dn_rai_dt = mp_tendency.dn_rai_dt * f_rai,
        dq_ice_dt = mp_tendency.dq_ice_dt,
        dn_ice_dt = mp_tendency.dn_ice_dt,
        dq_rim_dt = mp_tendency.dq_rim_dt,
        db_rim_dt = mp_tendency.db_rim_dt,
    )
end

# ============================================================================
# 2M+P3 saturation-adjustment limiter + substepping
# ============================================================================

"""
    apply_2m_satadj_limit(mp_tendency, thp, T, ρ, q_tot, q_lcl, q_rai, q_icl, dt)

Cap the net 2M+P3 condensation/deposition tendencies so a single timestep
cannot overshoot saturation. The liquid channel `(dq_lcl_dt + dq_rai_dt)` is
capped against the analytic condensation increment `qcon_satadj`; the ice
channel `(dq_ice_dt + dq_rim_dt)` is then capped against the deposition
increment `qdep_satadj` computed from the vapor remaining *after* the liquid
step. Each channel is scaled by a common, sign-preserving ratio in `[0, 1]`.

A method taking the microphysics timestepping (`::Explicit`/`::Implicit`/
`::Nothing`) dispatches to this cap only under `Explicit`; it is a no-op
otherwise (the implicit Jacobian handles stability).
"""
@inline function apply_2m_satadj_limit(
    mp_tendency, thp, T, ρ, q_tot, q_lcl, q_rai, q_icl, dt,
)
    (; dq_lcl_dt, dn_lcl_dt, dq_rai_dt, dn_rai_dt,
        dq_ice_dt, dn_ice_dt, dq_rim_dt, db_rim_dt) = mp_tendency
    FT = typeof(T)
    T_safe = max(FT(150), T)

    # Vapor diagnosed from total water minus condensate (no q_sno in P3 here).
    q_vap = max(zero(FT), q_tot - q_lcl - q_rai - q_icl)

    Rv = TD.Parameters.R_v(thp)
    cp_d = TD.Parameters.cp_d(thp)
    L_v = TD.latent_heat_vapor(thp, T_safe)
    L_s = TD.latent_heat_sublim(thp, T_safe)
    qv_sat_liq = TD.q_vap_saturation(thp, T_safe, ρ, TD.Liquid())
    qv_sat_ice = TD.q_vap_saturation(thp, T_safe, ρ, TD.Ice())

    qcon_satadj =
        (q_vap - qv_sat_liq) /
        (1 + L_v^2 * qv_sat_liq / (cp_d * Rv * T_safe^2)) / dt

    # Liquid: cap |dq_lcl_dt + dq_rai_dt| against qcon_satadj.
    net_liq = dq_lcl_dt + dq_rai_dt
    target_liq = if net_liq > 0
        min(net_liq, max(zero(FT), qcon_satadj))
    elseif net_liq < 0
        max(net_liq, min(zero(FT), qcon_satadj))
    else
        zero(FT)
    end
    ratio_liq = ifelse(abs(net_liq) > eps(FT), target_liq / net_liq, one(FT))
    ratio_liq = clamp(ratio_liq, zero(FT), one(FT))
    dq_lcl_dt *= ratio_liq
    dq_rai_dt *= ratio_liq
    dn_lcl_dt *= ratio_liq
    dn_rai_dt *= ratio_liq

    # Ice: same idea using the vapor remaining after the liquid step.
    qv_after_liq = max(zero(FT), q_vap - target_liq * dt)
    qdep_satadj =
        (qv_after_liq - qv_sat_ice) /
        (1 + L_s^2 * qv_sat_ice / (cp_d * Rv * T_safe^2)) / dt
    net_ice = dq_ice_dt + dq_rim_dt
    target_ice = if net_ice > 0
        min(net_ice, max(zero(FT), qdep_satadj))
    elseif net_ice < 0
        max(net_ice, min(zero(FT), qdep_satadj))
    else
        zero(FT)
    end
    ratio_ice = ifelse(abs(net_ice) > eps(FT), target_ice / net_ice, one(FT))
    ratio_ice = clamp(ratio_ice, zero(FT), one(FT))
    dq_ice_dt *= ratio_ice
    dq_rim_dt *= ratio_ice
    dn_ice_dt *= ratio_ice
    db_rim_dt *= ratio_ice

    return (;
        dq_lcl_dt, dn_lcl_dt, dq_rai_dt, dn_rai_dt,
        dq_ice_dt, dn_ice_dt, dq_rim_dt, db_rim_dt,
    )
end
@inline apply_2m_satadj_limit(t, thp, T, ρ, q_tot, q_lcl, q_rai, q_icl, dt, ::Implicit) = t
@inline apply_2m_satadj_limit(t, thp, T, ρ, q_tot, q_lcl, q_rai, q_icl, dt, ::Nothing) = t
@inline apply_2m_satadj_limit(t, thp, T, ρ, q_tot, q_lcl, q_rai, q_icl, dt, ::Explicit) =
    apply_2m_satadj_limit(t, thp, T, ρ, q_tot, q_lcl, q_rai, q_icl, dt)

# Project the full BMT 2M+P3 tendency NamedTuple onto the 8 prognostic fields
# we integrate (drops the activation/INP diagnostics; keeps dn_ice_dt). Inlined
# here to avoid coupling to the include order of microphysics_cache.jl.
@inline _project_mp23(t) = (;
    t.dq_lcl_dt, t.dn_lcl_dt, t.dq_rai_dt, t.dn_rai_dt,
    t.dq_ice_dt, t.dn_ice_dt, t.dq_rim_dt, t.db_rim_dt,
)

@inline function _limited_2m_tendency(
    cm2p, thp, ρ, T, q_tot, q_lcl, n_lcl, q_rai, n_rai,
    q_icl, n_ice, q_rim, b_rim, logλ, dt, timestepping,
)
    t = _project_mp23(
        BMT.bulk_microphysics_tendencies(
            BMT.Microphysics2Moment(), cm2p, thp, ρ, T, q_tot,
            q_lcl, n_lcl, q_rai, n_rai, q_icl, n_ice, q_rim, b_rim, logλ,
        ),
    )
    t = apply_2m_satadj_limit(t, thp, T, ρ, q_tot, q_lcl, q_rai, q_icl, dt, timestepping)
    f_liq = coupled_sink_limit_factor(t.dq_lcl_dt, t.dn_lcl_dt, q_lcl, n_lcl, dt)
    f_rai = coupled_sink_limit_factor(t.dq_rai_dt, t.dn_rai_dt, q_rai, n_rai, dt)
    return (;
        dq_lcl_dt = t.dq_lcl_dt * f_liq, dn_lcl_dt = t.dn_lcl_dt * f_liq,
        dq_rai_dt = t.dq_rai_dt * f_rai, dn_rai_dt = t.dn_rai_dt * f_rai,
        dq_ice_dt = t.dq_ice_dt, dn_ice_dt = t.dn_ice_dt,
        dq_rim_dt = t.dq_rim_dt, db_rim_dt = t.db_rim_dt,
    )
end

"""
    bulk_2m_tendencies_substepped(cm2p, thp, ρ, T, q_tot, q_lcl, n_lcl, q_rai,
        n_rai, q_icl, n_ice, q_rim, b_rim, logλ, dt, nsubs, timestepping)

Forward-Euler the 8-field 2M+P3 bulk tendency over `nsubs` substeps of
`dt/nsubs`, applying the saturation-adjustment cap and coupled-sink limiter at
each substep, and return the `dt`-averaged tendency `(state_after − state_before)/dt`.
`logλ` is held fixed across substeps; `q_tot` is conserved (mass-neutral
vapor↔condensate exchange); a local temperature `Tsub` evolves via latent
heating so the satadj cap sees the corrected saturation deficit. Reduces to the
single-shot limited tendency when `nsubs ≤ 1`.
"""
@inline function bulk_2m_tendencies_substepped(
    cm2p, thp, ρ, T, q_tot, q_lcl, n_lcl, q_rai, n_rai,
    q_icl, n_ice, q_rim, b_rim, logλ, dt, nsubs, timestepping,
)
    FT = typeof(T)
    if nsubs <= 1
        return _limited_2m_tendency(
            cm2p, thp, ρ, T, q_tot, q_lcl, n_lcl, q_rai, n_rai,
            q_icl, n_ice, q_rim, b_rim, logλ, dt, timestepping,
        )
    end
    dt_sub = dt / FT(nsubs)
    cp_d = TD.Parameters.cp_d(thp)
    qlcl, nlcl, qrai, nrai = q_lcl, n_lcl, q_rai, n_rai
    qicl, nice, qrim, brim = q_icl, n_ice, q_rim, b_rim
    Tsub = T
    for _ in 1:nsubs
        t = _limited_2m_tendency(
            cm2p, thp, ρ, Tsub, q_tot, qlcl, nlcl, qrai, nrai,
            qicl, nice, qrim, brim, logλ, dt_sub, timestepping,
        )
        qlcl += dt_sub * t.dq_lcl_dt
        nlcl += dt_sub * t.dn_lcl_dt
        qrai += dt_sub * t.dq_rai_dt
        nrai += dt_sub * t.dn_rai_dt
        qicl += dt_sub * t.dq_ice_dt
        nice += dt_sub * t.dn_ice_dt
        qrim += dt_sub * t.dq_rim_dt
        brim += dt_sub * t.db_rim_dt
        L_v = TD.latent_heat_vapor(thp, max(FT(150), Tsub))
        L_s = TD.latent_heat_sublim(thp, max(FT(150), Tsub))
        Tsub += dt_sub * (L_v * (t.dq_lcl_dt + t.dq_rai_dt) + L_s * t.dq_ice_dt) / cp_d
    end
    return (;
        dq_lcl_dt = (qlcl - q_lcl) / dt, dn_lcl_dt = (nlcl - n_lcl) / dt,
        dq_rai_dt = (qrai - q_rai) / dt, dn_rai_dt = (nrai - n_rai) / dt,
        dq_ice_dt = (qicl - q_icl) / dt, dn_ice_dt = (nice - n_ice) / dt,
        dq_rim_dt = (qrim - q_rim) / dt, db_rim_dt = (brim - b_rim) / dt,
    )
end
