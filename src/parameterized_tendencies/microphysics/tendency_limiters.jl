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
    smooth_min_limiter(tendency, tendency_bound, sharpness=1e-8)

Smooth approximation to `min(tendency, tendency_bound)` for GPU-friendly limiting.

Uses a differentiable formula that approaches `min(S, B)` as `sharpness → 0`:
```math
\\frac{S + B - \\sqrt{(S - B)^2 + ε^2}}{2}
```

# Arguments
- `tendency`: The raw tendency (source/sink term) [kg/kg/s]
- `tendency_bound`: Maximum allowed tendency [same units]
- `sharpness`: Smoothness parameter (default: 1e-10). Smaller = sharper transition.

# Example
```julia
# Limit rain accretion tendency by available cloud liquid
S_accr = accretion_rate(q_liq, q_rain)
S_limited = smooth_min_limiter(S_accr, limit(q_liq, dt, 3))
```

See also: [`smooth_tendency_limiter`](@ref)
"""
@inline function smooth_min_limiter(tendency, tendency_bound, sharpness = 1e-8)
    ε = oftype(tendency, sharpness)

    # Smooth minimum formula: approaches min(S, B) as ε → 0
    # When S = B: output = B - ε/2 ≈ B 
    S = tendency
    B = tendency_bound

    return (S + B - sqrt((S - B)^2 + ε^2)) / 2
end

"""
    smooth_tendency_limiter(tendency, tend_bound_pos, tend_bound_neg)

Bidirectional smooth tendency limiter for processes with both positive and negative tendencies.

# Arguments
- `tendency`: Raw tendency (source or sink) [kg/kg/s]
- `tend_bound_pos`: Maximum allowed magnitude for positive tendencies [kg/kg/s]
- `tend_bound_neg`: Maximum allowed magnitude for negative tendencies [kg/kg/s]

# Returns
Limited tendency satisfying:
- `0 ≤ S_limited ≤ tend_bound_pos` when `tendency > 0`
- `-tend_bound_neg ≤ S_limited ≤ 0` when `tendency < 0`

# Example
```julia
# Limit condensation/evaporation by available vapor and liquid
S_cond = condensation_rate(q_vap, q_liq)
S_limited = smooth_tendency_limiter(
    S_cond,
    limit(q_vap, dt, 5),     # bound for condensation (depletes q_vap)
    limit(q_liq, dt, 5),     # bound for evaporation (depletes q_liq)
)
```
"""
@inline function smooth_tendency_limiter(
    tendency,
    tend_bound_pos,
    tend_bound_neg,
)

    # Ensure bounds are non-negative
    tend_bound_pos = max(zero(tend_bound_pos), tend_bound_pos)
    tend_bound_neg = max(zero(tend_bound_neg), tend_bound_neg)

    # Positive tendency (source): limit by tend_bound
    limited_pos = smooth_min_limiter(tendency, tend_bound_pos)

    # Negative tendency (sink): limit by tend_bound_neg (which becomes a negative limit)
    limited_neg = -smooth_min_limiter(-tendency, tend_bound_neg)

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
- If `S < 0`: `-smooth_min_limiter(-S, limit(q, dt, n))`
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
        -smooth_min_limiter(-S, limit(q, dt, n)),
        S,
    )
end


"""
    apply_1m_tendency_limits(mp_tendency, tps, q_tot, q_liq, q_ice, q_rai, q_sno, dt)

Apply physical limiting to 1M microphysics tendencies.

Two layers of limiting are applied:
1. **Mass conservation**: prevents unphysical depletion of each species beyond
   available sources (via `smooth_tendency_limiter`). Each species is limited
   bidirectionally against its maximal cross-species source pool:
   - `dq_lcl_dt`: source = `q_vap + q_ice`, sink = `q_liq`
   - `dq_icl_dt`: source = `q_vap + q_liq`, sink = `q_ice`
   - `dq_rai_dt`: source = `q_liq + q_sno`, sink = `q_rai`
   - `dq_sno_dt`: source = `q_ice + q_rai`, sink = `q_sno`
2. **Combined temperature rate**: caps the combined tendency to a maximum equivalent 
   temperature change `dT/dt = (L_v/c_p)·dq_lcl + (L_s/c_p)·dq_icl`
   and rescales `dq_lcl_dt` and `dq_icl_dt` uniformly. (Temperature is used as a convenient 
   metric for the magnitude of tendencies, although not all tendencies are actually associated 
   with temperature changes.)

# Arguments
- `mp_tendency`: NamedTuple from BMT with raw tendencies
- `tps`: Thermodynamic parameters (for `L_v`, `L_s`, `c_p`)
- `q_tot`: Total water specific humidity [kg/kg]
- `q_liq`: Cloud liquid [kg/kg]
- `q_ice`: Cloud ice [kg/kg]
- `q_rai`: Rain [kg/kg]
- `q_sno`: Snow [kg/kg]
- `dt`: Timestep [s]

# Returns
NamedTuple with limited tendencies: `(dq_lcl_dt, dq_icl_dt, dq_rai_dt, dq_sno_dt)`
"""
@inline function apply_1m_tendency_limits(
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
    q_vap = q_tot - q_liq - q_ice - q_rai - q_sno

    # Mass-conservation limits using cross-species source pools
    n_sink = 5
    n_source = 20
    dq_lcl_dt = smooth_tendency_limiter(
        mp_tendency.dq_lcl_dt,
        limit(q_vap + q_ice, dt, n_source),
        limit(q_liq, dt, n_sink),
    )
    dq_icl_dt = smooth_tendency_limiter(
        mp_tendency.dq_icl_dt,
        limit(q_vap + q_liq, dt, n_source),
        limit(q_ice, dt, n_sink),
    )
    dq_rai_dt = smooth_tendency_limiter(
        mp_tendency.dq_rai_dt,
        limit(q_liq + q_sno, dt, n_source),
        limit(q_rai, dt, n_sink),
    )
    dq_sno_dt = smooth_tendency_limiter(
        mp_tendency.dq_sno_dt,
        limit(q_ice + q_rai, dt, n_source),
        limit(q_sno, dt, n_sink),
    )

    # Combined temperature-rate limiter:
    # Condensate tendencies are expressed as possible temperature changes (although they 
    # may not be realized).
    # A single combined scale factor preserves the ratio between condensate species,
    # preventing mass-energy decoupling that can drive temperatures negative.
    Lv_over_cp = TD.Parameters.LH_v0(tps) / TD.Parameters.cp_d(tps)
    Ls_over_cp = TD.Parameters.LH_s0(tps) / TD.Parameters.cp_d(tps)

    # Max 5 K temperature change per timestep 
    # TODO: arbitrary choice; remove or make very large once microphysics is implicit
    dT_dt_max = FT(5) / dt

    dT_dt = Lv_over_cp * dq_lcl_dt + Ls_over_cp * dq_icl_dt
    scale = min(FT(1), dT_dt_max / max(abs(dT_dt), eps(FT)))

    return (
        dq_lcl_dt = dq_lcl_dt * scale,
        dq_icl_dt = dq_icl_dt * scale,
        dq_rai_dt = dq_rai_dt,
        dq_sno_dt = dq_sno_dt,
    )
end
