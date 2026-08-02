# ============================================================================
# Tendency Limiters for Microphysics
# ============================================================================
# Functions for limiting source/sink terms to prevent numerical instabilities
# and unphysical negative values in water species.

"""
    limit(q, dt, n::Int)

Compute the largest depletion rate of `q` that one of `n` competing sinks may apply.

Each sink is allowed at most a fraction `1/n` of the available quantity per
timestep, so that `n` simultaneous sinks cannot together drive `q` negative in one
step. Negative `q` is treated as zero.

# Arguments

  - `q`: Available quantity, e.g. a specific humidity [kg/kg].
  - `dt`: Model timestep [s].
  - `n::Int`: Number of sinks splitting the available quantity [-].

# Returns

`max(0, q) / (dt · n)`, a non-negative rate [kg/kg/s].

# Examples

```julia
# Rain has three sinks (evaporation, accretion, self-collection), so each may
# consume at most a third of the available rain per timestep.
max_rate = limit(q_rai, dt, 3)
```
"""
@inline function limit(q, dt, n::Int)
    return max(zero(q), q) / dt / n
end

"""
    limit_sink(S, q, dt, n = 3)

Clip a sink tendency so it cannot deplete more than its share of `q`.

Sign convention: `S < 0` is a sink and is clipped in magnitude at
[`limit`](@ref)`(q, dt, n)`; `S ≥ 0` is a source and passes through unchanged.

# Arguments

  - `S`: Raw tendency, source or sink [kg/kg/s].
  - `q`: Available quantity [kg/kg].
  - `dt`: Model timestep [s].
  - `n`: Number of competing sinks, default `3` [-].

# Returns

`-min(-S, limit(q, dt, n))` when `S < 0`, otherwise `S` [kg/kg/s].

# Examples

```julia
# Limit rain evaporation to the available rain.
S_evap_limited = limit_sink(S_evap, q_rai, dt, 3)
```
"""
@inline function limit_sink(S, q, dt, n = 3)
    return ifelse(S < zero(S), -min(-S, limit(q, dt, n)), S)
end

"""
    tendency_limiter(tendency, tend_bound_pos, tend_bound_neg)

Clip `tendency` to `[-tend_bound_neg, tend_bound_pos]`.

Both bounds are magnitudes and are floored at zero before use, so a source
(`tendency ≥ 0`) is capped at `tend_bound_pos` and a sink is capped in magnitude at
`tend_bound_neg`. The selection is branchless for GPU broadcast.

# Arguments

  - `tendency`: Raw tendency, positive for a source [kg/kg/s].
  - `tend_bound_pos`: Largest permitted source rate [kg/kg/s].
  - `tend_bound_neg`: Largest permitted sink magnitude [kg/kg/s].

# Returns

The clipped tendency, in the same units as `tendency`.
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
    coupled_sink_limit_factor(S1, S2, q1, q2, dt, n = 3)

Compute one scaling factor for a pair of tendencies that deplete two species together.

Mass and number are depleted by the same process (autoconversion removes both cloud
liquid mass and droplet number), so they must be scaled by a common factor to keep
the mean drop size consistent. The factor is the more restrictive of the two
individual limits; a component that is not a sink, or that already respects its
limit, contributes a factor of one.

# Arguments

  - `S1`, `S2`: Raw tendencies of the two coupled species [kg/kg/s and kg⁻¹/s];
    only components with `S < 0` are limited.
  - `q1`, `q2`: Corresponding available quantities [kg/kg and kg⁻¹].
  - `dt`: Model timestep [s].
  - `n`: Number of competing sinks, default `3` [-].

# Returns

A factor `f ∈ (0, 1]` [-] such that `|Sᵢ · f| ≤` [`limit`](@ref)`(qᵢ, dt, n)` for
every component that is a sink.

# Examples

```julia
# Autoconversion depletes both q_liq and n_liq.
f = coupled_sink_limit_factor(dq_liq_auto, dn_liq_auto, q_liq, n_liq, dt)
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

Limit the 0-moment total-water sink to the available `q_tot`.

Thin wrapper over [`limit_sink`](@ref) with the default three competing sinks.
Called from [`microphysics_tendencies_0m`](@ref).
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
    apply_2m_tendency_limits!(
        ᶜmp_tendency, timestepping, ᶜq_lcl, ᶜn_lcl, ᶜq_rai, ᶜn_rai, dt,
    )

Limit the cached 2-moment tendency field in place so explicit steps stay positive.

Dispatches on `timestepping`:

  - `Explicit`: scales the coupled liquid pair (`dq_lcl_dt`, `dn_lcl_dt`) and the
    coupled rain pair (`dq_rai_dt`, `dn_rai_dt`) by their respective
    [`coupled_sink_limit_factor`](@ref); the ice entries `dq_ice_dt`, `dq_rim_dt`,
    `db_rim_dt` pass through unchanged.
  - `Implicit`: no-op, since the Jacobian provides the stability.
  - `nothing`: no-op.

Mutates `ᶜmp_tendency`; the return value is unused.
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
        dq_rim_dt = mp_tendency.dq_rim_dt,
        db_rim_dt = mp_tendency.db_rim_dt,
    )
end
