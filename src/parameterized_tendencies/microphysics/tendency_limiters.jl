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

"""
    limit_sink(S, q, dt, n=3)

Limit a sink tendency to prevent species depletion using smooth limiting.

Only applies limiting when `S < 0` (sink). Source tendencies (`S ≥ 0`) pass through unchanged.

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
    return ifelse(
        S < zero(S),
        -min(-S, limit(q, dt, n)),
        S,
    )
end


"""
    apply_1m_tendency_limits(timestepping, mp_tendency, tps, q_tot, q_liq, q_ice, q_rai, q_sno, dt)

Apply physical limiting to 1M microphysics tendencies.

Applies mass-conservation and temperature-rate limiters to prevent
negative species concentrations from cross-species sinks (accretion,
autoconversion) that the diagonal-only Jacobian cannot stabilize.
"""
@inline function apply_1m_tendency_limits(
    ::AbstractTimesteppingMode,
    mp_tendency,
    tps,
    q_tot,
    q_liq,
    q_ice,
    q_rai,
    q_sno,
    dt,
)
    FT = typeof(q_tot)
    # Guard against negative q_vap from numerical errors (AD-safe via max)
    q_vap = max(zero(q_tot), q_tot - q_liq - q_ice - q_rai - q_sno)

    # Mass-conservation limits:
    # n_sink = 1: sink cannot exceed species content in one step
    # n_source = 10: source cannot exceed 10% of pool per step
    n_sink = 1
    n_source = 10

    dq_lcl_dt = tendency_limiter(
        mp_tendency.dq_lcl_dt,
        limit(q_vap + q_ice, dt, n_source),
        limit(q_liq, dt, n_sink),
    )
    dq_icl_dt = tendency_limiter(
        mp_tendency.dq_icl_dt,
        limit(q_vap + q_liq, dt, n_source),
        limit(q_ice, dt, n_sink),
    )
    dq_rai_dt = tendency_limiter(
        mp_tendency.dq_rai_dt,
        limit(q_liq + q_sno, dt, n_source),
        limit(q_rai, dt, n_sink),
    )
    dq_sno_dt = tendency_limiter(
        mp_tendency.dq_sno_dt,
        limit(q_ice, dt, n_source),
        limit(q_sno, dt, n_sink),
    )

    return (
        dq_lcl_dt = dq_lcl_dt,
        dq_icl_dt = dq_icl_dt,
        dq_rai_dt = dq_rai_dt,
        dq_sno_dt = dq_sno_dt,
    )
end

"""
    _apply_1m_limits!(ᶜmp_tendency, timestepping, tps, ᶜq_tot, ᶜq_liq, ᶜq_ice, ᶜq_rai, ᶜq_sno, dt)

Function barrier for `apply_1m_tendency_limits` in broadcast expressions.

Dispatches on `timestepping::AbstractTimesteppingMode` outside the broadcast,
then calls `apply_1m_tendency_limits` inside the broadcast with the concrete
singleton type.  This avoids passing the timestepping mode *through* the
broadcast (which would require a `Ref` wrapper and its 8-byte heap allocation).
"""
@inline function _apply_1m_limits!(
    ᶜmp_tendency, timestepping::Implicit, tps, ᶜq_tot, ᶜq_liq, ᶜq_ice, ᶜq_rai, ᶜq_sno, dt,
)
    @. ᶜmp_tendency = apply_1m_tendency_limits(
        Implicit(), ᶜmp_tendency, tps, ᶜq_tot, ᶜq_liq, ᶜq_ice, ᶜq_rai, ᶜq_sno, dt,
    )
end
@inline function _apply_1m_limits!(
    ᶜmp_tendency, timestepping::Explicit, tps, ᶜq_tot, ᶜq_liq, ᶜq_ice, ᶜq_rai, ᶜq_sno, dt,
)
    @. ᶜmp_tendency = apply_1m_tendency_limits(
        Explicit(), ᶜmp_tendency, tps, ᶜq_tot, ᶜq_liq, ᶜq_ice, ᶜq_rai, ᶜq_sno, dt,
    )
end

