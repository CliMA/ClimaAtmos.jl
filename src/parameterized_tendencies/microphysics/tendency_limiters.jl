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
    apply_2m_tendency_limits(mp_tendency, timestepping, q_lcl, n_lcl, q_rai, n_rai, q_ice, n_ice, dt) -> NamedTuple

Pointwise limiter for one cell's 2M microphysics tendency. Dispatches on
the timestepping kind:

- `Explicit`: scales `(dq_*, dn_*)` by the coupled-sink factor so that no
  reservoir empties below zero in `dt` — the explicit-Euler stability
  guard. Returns a new NamedTuple with scaled fields.
- `Implicit` / `Nothing`: no-op; the Jacobian (or the absence of a stiff
  step) handles stability. Returns `mp_tendency` unchanged.

Designed to be called inside a `@.` broadcast — the dispatch is on the
timestepping value, all other args are scalars per cell. Replaces the
older field-wise `apply_2m_tendency_limits!` (in-place mutation; deleted)
which was redundant with this + broadcast.
"""
@inline apply_2m_tendency_limits(mp_tendency, ::Implicit, args...) = mp_tendency
@inline apply_2m_tendency_limits(mp_tendency, ::Nothing,  args...) = mp_tendency
@inline function apply_2m_tendency_limits(
    mp_tendency, ::Explicit,
    q_lcl, n_lcl, q_rai, n_rai, q_ice, n_ice, dt,
)
    (; dq_lcl_dt, dn_lcl_dt, dq_rai_dt, dn_rai_dt) = mp_tendency
    (; dq_ice_dt, dn_ice_dt, dq_rim_dt, db_rim_dt) = mp_tendency
    f_liq = coupled_sink_limit_factor(dq_lcl_dt, dn_lcl_dt, q_lcl, n_lcl, dt)
    f_rai = coupled_sink_limit_factor(dq_rai_dt, dn_rai_dt, q_rai, n_rai, dt)
    f_ice = coupled_sink_limit_factor(dq_ice_dt, dn_ice_dt, q_ice, n_ice, dt)
    return (;
        dq_lcl_dt = dq_lcl_dt * f_liq,
        dn_lcl_dt = dn_lcl_dt * f_liq,
        dq_rai_dt = dq_rai_dt * f_rai,
        dn_rai_dt = dn_rai_dt * f_rai,
        dq_ice_dt = dq_ice_dt * f_ice,
        dn_ice_dt = dn_ice_dt * f_ice,
        dq_rim_dt = dq_rim_dt,
        db_rim_dt = db_rim_dt,
    )
end

"""
    microphysics_tendencies_quadrature_and_limits_2m(
        sgs_quad, cmp, thp, ρ, T, q_tot,
        q_lcl, n_lcl, q_rai, n_rai, q_ice, n_ice, q_rim, b_rim, logλ,
        inpc_log_shift, w, p,
        timestepping, dt,
    ) -> NamedTuple

Single pointwise function that combines:
1. SGS-quadrature integration of `bulk_microphysics_tendencies` (currently
   `GridMeanSGS` only — `SGSQuadrature` errors out).
2. Tendency limiting via [`apply_2m_tendency_limits`](@ref) — dispatches
   on `Explicit` / `Implicit` / `Nothing`.

This is the unified path for all 2M call sites in CA (grid-mean,
PrognosticEDMFX updrafts, PrognosticEDMFX environment), replacing the
older split between `microphysics_tendencies_quadrature_2m` +
`apply_2m_tendency_limits!` and the legacy `compute_2m_precipitation_tendencies!`
wrapper. Designed for `@.` broadcast use.
"""
@inline function microphysics_tendencies_quadrature_and_limits_2m(
    # arguments for bulk tendency
    sgs_quad, cmp, thp, ρ, T, q_tot_nonneg,
    q_lcl, n_lcl, q_rai, n_rai, q_ice, n_ice, q_rim, b_rim, logλ,
    # per-cell ambient inputs read by w/p-dependent activation schemes
    # (Twomey, FixedARG); ignored by DiagnosticNc and NoActivation.
    inpc_log_shift, w, p,
    # tendency-limiter dispatch
    timestepping, dt,
)
    F_rim = rime_mass_fraction(q_rim, q_ice)
    q_rim = F_rim * q_ice  # TODO: Should probably limit q_rim in one place
    mp_tendency = microphysics_tendencies_quadrature_2m(
        sgs_quad, cmp, thp, ρ, T, q_tot_nonneg,
        q_lcl, n_lcl, q_rai, n_rai, q_ice, n_ice, q_rim, b_rim, logλ,
        inpc_log_shift, w, p,
    )
    return apply_2m_tendency_limits(
        mp_tendency, timestepping,
        q_lcl, n_lcl, q_rai, n_rai, q_ice, n_ice, dt,
    )
end
