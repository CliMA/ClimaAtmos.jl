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
    triangle_inequality_limiter(tendency, tendency_bound, tendency_bound_neg=0)

Smoothly limit a `tendency` to not exceed `tendency_bound` using a C∞ differentiable formula.

This limiter ensures that source/sink terms do not deplete their source species faster
than physically available, while remaining smooth for GPU kernels and automatic differentiation.

# Mathematical Formula
Uses the Horn (2012) formula derived from the triangle inequality:
```math
S_{\\text{limited}} = S + B - \\sqrt{S^2 + B^2}
```
where `S` is the raw `tendency` and `B` is the `tendency_bound`.

# Properties
For `S ≥ 0` and `B ≥ 0`:
- **Bounded output:** `0 ≤ S_limited ≤ min(S, B)`
- **Limiting regime:** When `S >> B`, output approaches `B` (capped at bound)
- **Crossover point:** When `S = B`, output is `(2 - √2)S ≈ 0.59 S`
- **Pass-through regime:** When `S << B`, output approaches `S` (minimal limiting)
- **C∞ differentiable:** Smooth everywhere, suitable for GPU and AD

# TODO: The `S = B` behavior is arbitrary and odd. We could instead use a limiter 
# that smoothly transitions from `S` to `B` over the entire range of `S`.

# Arguments
- `tendency`: The raw tendency (source/sink term) to limit [kg/kg/s or similar]
- `tendency_bound`: Maximum allowed magnitude for positive tendencies [same units].
  Typically computed as `limit(q_source, dt, n)` where `n` is the number of
  competing sinks for the source species.
- `tendency_bound_neg`: Maximum allowed magnitude when `tendency < 0` [same units].
  Used for bidirectional processes (e.g., condensation/evaporation).

# Returns
Limited tendency `S_limited` satisfying `0 ≤ S_limited ≤ tendency_bound` when
`tendency ≥ 0` and `tendency_bound ≥ 0`.

# Negative Tendency Handling
When `tendency < 0`, the function recursively calls itself with swapped bounds:
```julia
-triangle_inequality_limiter(-tendency, tendency_bound_neg, tendency_bound)
```
This ensures symmetric limiting for bidirectional processes.

# Edge Cases
When `tendency_bound < 0` (due to numerical errors in the source quantity):
- If `tendency_bound_neg > 0`: Attempts reverse limiting
- Otherwise: Returns zero (conservative fallback)

# Example
```julia
# Limit rain accretion tendency by available cloud liquid and rain
S_accr = accretion_rate(q_liq, q_rain)
S_limited = triangle_inequality_limiter(
    S_accr,
    limit(q_liq, dt, 3),  # bound for positive tendency (depletes q_liq)
    limit(q_rain, dt, 3), # bound for negative tendency (depletes q_rain)
)

!!! note
    This is deprecated. Use `smooth_tendency_limiter` instead.
```

# Reference
Horn, M. (2012). "ASAMgpu V1.0 – a moist fully compressible atmospheric model
    using graphics processing units (GPUs)". *Geosci. Model Dev.*, 5, 345–353.
    https://doi.org/10.5194/gmd-5-345-2012
"""
@inline function triangle_inequality_limiter(
    tendency,
    tendency_bound,
    tendency_bound_neg = 0,
)
    FT = eltype(tendency)

    # Handle negative tendency via recursion with swapped bounds
    # NOTE: Must use `if` here, not `ifelse`, because ifelse evaluates both branches
    # which would cause infinite recursion
    if tendency < FT(0)
        return -triangle_inequality_limiter(-tendency, tendency_bound_neg, tendency_bound)
    end

    # Apply Horn (2012) limiter formula: S + B - sqrt(S² + B²)
    # This smoothly transitions from S (when S << B) to B (when S >> B)
    tendency_limited =
        tendency + tendency_bound - sqrt(tendency^2 + tendency_bound^2)

    # Edge case: tendency_bound < 0 (source quantity went negative due to numerics)
    # If tendency_bound_neg > 0, attempt reverse limiting; otherwise return zero
    reverse_step1 = tendency_limited
    reverse_step2 =
        -reverse_step1 + tendency_bound_neg -
        sqrt(reverse_step1^2 + tendency_bound_neg^2)
    tendency_reverse_limited = -reverse_step2

    # Branchless selection (GPU-friendly)
    has_valid_bound = tendency_bound >= FT(0)
    can_reverse_limit = tendency_bound_neg > FT(0)

    return ifelse(
        has_valid_bound,
        tendency_limited,           # Standard case: valid bound
        ifelse(
            can_reverse_limit,
            tendency_reverse_limited,  # Fallback: try reverse limiting
            FT(0),                     # No valid limiting possible
        ),
    )
end

"""
    smooth_min_limiter(tendency, tendency_bound, sharpness=1e-10)

Smoothly limit `tendency` to not exceed `tendency_bound` using a smooth minimum approximation.

This is a C∞ differentiable approximation to `min(tendency, tendency_bound)` that:
- Returns `≈ tendency` when `tendency << tendency_bound` (pass-through)
- Returns `≈ tendency_bound` when `tendency >> tendency_bound` (capped)
- Returns `≈ tendency_bound` when `tendency = tendency_bound` 

# Mathematical Formula
```math
S_{\\text{limited}} = \\frac{1}{2}\\left(S + B - \\sqrt{(S - B)^2 + \\epsilon^2}\\right)
```
where `S` is the raw `tendency`, `B` is the `tendency_bound`, and `ε` is the `sharpness` parameter.

# Properties
For `S ≥ 0` and `B ≥ 0`:
- **Bounded output:** `0 ≤ S_limited ≤ min(S, B)`
- **Limiting regime:** When `S >> B`, output approaches `B`
- **Crossover point:** When `S = B`, output is `B - ε/2 ≈ B` (much better than triangle's 0.59B)
- **Pass-through regime:** When `S << B`, output approaches `S`
- **Symmetric:** `smooth_min_limiter(S, B) = smooth_min_limiter(B, S)`
- **C∞ differentiable:** Smooth everywhere, suitable for GPU and AD

# Arguments
- `tendency`: The raw tendency to limit [kg/kg/s or similar]
- `tendency_bound`: Maximum allowed value [same units]
- `sharpness`: Smoothing parameter controlling transition width (default: 1e-10).
  Smaller values give sharper transitions closer to `min(S, B)`.

# Returns
Limited tendency `S_limited` satisfying `0 ≤ S_limited ≤ min(S, B)` for `S, B ≥ 0`.

# Example
```julia
# Limit rain accretion tendency by available cloud liquid
S_accr = accretion_rate(q_liq, q_rain)
S_limited = smooth_min_limiter(S_accr, limit(q_liq, dt, 3))
```

See also: [`triangle_inequality_limiter`](@ref), [`smooth_tendency_limiter`](@ref)
"""
@inline function smooth_min_limiter(tendency, tendency_bound, sharpness = 1e-10)
    FT = eltype(tendency)
    ε = FT(sharpness)

    # Smooth minimum formula: approaches min(S, B) as ε → 0
    # When S = B: output = B - ε/2 ≈ B 
    S = tendency
    B = tendency_bound

    return (S + B - sqrt((S - B)^2 + ε^2)) / 2
end

"""
    smooth_tendency_limiter(S, q_source, q_sink, dt, n_sink, n_source)

Branchless bidirectional limiter for fused microphysics tendencies.

Limits both sinks and sources using [`smooth_min_limiter`](@ref):
- Sources (S > 0): capped by `q_source / (dt × n_source)`
- Sinks (S < 0): capped by `q_sink / (dt × n_sink)` (prevent species depletion)

This is needed because BMT aggregates all micro-processes (including cloud
condensation and phase conversions) into a single tendency.

Source limits should be physically motivated per-species bounds:
- liquid cloud: `saturation_excess + q_ice` (condensation + ice melting)
- ice cloud: `saturation_excess + q_liq` (deposition + liquid freezing)
- rain: `q_liq + q_sno` (autoconversion/accretion + snow melting)
- snow: `q_ice + q_rai` (autoconversion/accretion + rain freezing)

# Arguments
- `S`: Tendency [kg/kg/s], can be positive (source) or negative (sink)
- `q_source`: Available source quantity for this species [kg/kg]
- `q_sink`: Available sink quantity (species being depleted) [kg/kg]
- `dt`: Timestep [s]
- `n_sink`: Number of timesteps for sink depletion (default: 5)
- `n_source`: Number of timesteps for source depletion (default: 10)

# Example
```julia
# Limit BMT tendency: sources bounded by sat_excess + partner, sinks by species
Sqₗᵐ = smooth_tendency_limiter(dq_lcl_dt, sat_excess + q_ice, q_liq, dt)
```

See also: [`smooth_min_limiter`](@ref), [`limit`](@ref)
"""
@inline function smooth_tendency_limiter(
    S, q_source, q_sink, dt, n_sink = 5, n_source = 10,
)
    FT = eltype(S)
    bound_pos = limit(q_source, dt, n_source)
    bound_neg = limit(q_sink, dt, n_sink)
    pos_limited = max(smooth_min_limiter(S, bound_pos), FT(0))
    neg_limited = -max(smooth_min_limiter(-S, bound_neg), FT(0))
    return ifelse(S >= FT(0), pos_limited, neg_limited)
end

"""
    coupled_sink_limit_factor(Sq, Sn, q, n, dt, n_sinks=3)

Compute a single limiting factor for coupled mass/number tendencies.

Uses the MORE restrictive limit to preserve the mass/number ratio (mean particle size).
Both mass and number tendencies should be multiplied by this factor.

# Arguments
- `Sq`: Mass tendency [kg/kg/s]
- `Sn`: Number tendency [1/kg/s]
- `q`: Available mass [kg/kg]
- `n`: Available number [1/kg]
- `dt`: Timestep [s]
- `n_sinks`: Number of competing sink processes (default: 3)

# Returns
Limiting factor `f ∈ [0, 1]` to multiply both tendencies by.
Returns 1 (no limiting) if both tendencies are sources (≥ 0).
"""
@inline function coupled_sink_limit_factor(Sq, Sn, q, n, dt, n_sinks = 3)
    FT = eltype(Sq)

    # Both sources → no limiting needed
    if Sq >= FT(0) && Sn >= FT(0)
        return FT(1)
    end

    # Compute bounds for each quantity
    bound_q = limit(q, dt, n_sinks)
    bound_n = limit(n, dt, n_sinks)

    # Compute limiting factors: how much must we scale to stay within bounds?
    # Only compute for sinks (negative tendencies)
    f_q = ifelse(Sq < FT(0) && -Sq > bound_q, bound_q / -Sq, FT(1))
    f_n = ifelse(Sn < FT(0) && -Sn > bound_n, bound_n / -Sn, FT(1))

    # Use the more restrictive factor to preserve mass/number ratio
    return min(f_q, f_n)
end


"""
    limit_sink(S, q, dt, n=3)

Limit a sink tendency to prevent overdrawing the source quantity.

Only limits negative tendencies (sinks); positive tendencies (sources) pass through.
Uses `smooth_min_limiter` for C∞ differentiability.

# Arguments
- `S`: Tendency [kg/kg/s or similar], negative for sinks
- `q`: Available source quantity [kg/kg or similar]
- `dt`: Timestep [s]
- `n`: Number of competing sink processes (default: 3)

# Returns
Limited tendency with same sign as input, magnitude bounded by `q / (dt × n)`.

# Example
```julia
# Limit ice depletion tendency
Sqᵢᵐ_lim = limit_sink(Sqᵢᵐ, q_ice, dt)
```
"""
@inline function limit_sink(S, q, dt, n = 3)
    FT = eltype(S)
    return ifelse(
        S < FT(0),
        -smooth_min_limiter(-S, limit(q, dt, n)),
        S,
    )
end
