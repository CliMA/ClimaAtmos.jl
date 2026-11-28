# ============================================================================
# Conditional-moment node rule for the 1M SGS microphysics
# ============================================================================
#
# With the Gaussian closure every quantity the 1M evaluator reads at a node is
# a function of one random variable, the centred saturation excess
# S′ ~ N(0, σ_S²), and of the temperature regressed on it. Three thresholds
# partition S′:
#
#     t_a = −λ / α        actual cloud edge:       q_c(S′)    = max(0, λ + α S′)
#     t_e = −λ_eq         equilibrium cloud edge:  q_c_eq(S′) = max(0, λ_eq + S′)
#     t_p = S_star        precipitation shaft edge (the moistest a_p of the PDF)
#
# Sorting them gives at most four intervals, on each of which both condensates
# are linear in S′ and the shaft flag is constant. The Gaussian interval
# moments give each interval its probability weight and the conditional mean
# and variance of S′ exactly, so one node at the conditional mean integrates
# every linear function exactly (mass conservation and the mean relaxation
# toward q_c_eq hold for any cloud fraction), and a two-point Gauss rule of the
# truncated distribution, matched to the conditional mean, variance and third
# moment, carries the in-cloud spread the power-law collision rates see. The
# fixed Gauss–Hermite tensor rule instead places all of the condensate of a
# cell with a small cloud fraction on its moistest corner node of weight 1/36
# and the rates with it.
#
# The temperature also spreads at fixed S′. That spread, σ⊥, is the part of the
# temperature variance the regression on S′ does not carry (q′ moves with
# ∂q_sat/∂T T′ so that S′ stays fixed). The rates respond to it mainly through
# the kink at the freezing point (melting of snow, rain–snow collisions), so a
# node whose spread straddles T_freeze is split there; the temperature points
# are generated in the sum over the nodes, so the rule itself keeps its eight
# slots.

"""
    _mills_continued_fraction(x)

Inner convergents `(d, e)` of Laplace's continued fraction
`r(−x) = φ(x) / Q(x) = x + 1 / (x + d)`, `d = 2 / (x + e)`,
`e = 3 / (x + 4 / (x + ⋯))` for the inverse Mills ratio of the standard
normal tail at `x > 0`, evaluated to 20 levels in `Float32` and 32 levels in `Float64`.
"""
@inline function _mills_continued_fraction(x)
    FT = typeof(x)
    e = zero(FT)
    n_levels = FT === Float32 ? 20 : 32
    for k in n_levels:-1:3
        e = FT(k) / (x + e)
    end
    d = FT(2) / (x + e)
    return d, e
end

@inline function _normal_abs_tail_and_pdf(z)
    FT = typeof(z)
    if !isfinite(z)
        return (zero(FT), zero(FT))
    end
    z_abs = abs(z)
    z_max = FT === Float32 ? FT(15) : FT(40)
    if !(z_abs < z_max)
        return (zero(FT), zero(FT))
    end
    inv_sqrt2π = one(FT) / sqrt(FT(2) * FT(π))
    φ = exp(-FT(0.5) * z_abs * z_abs) * inv_sqrt2π
    if z_abs >= FT(2)
        d, _ = _mills_continued_fraction(z_abs)
        t = z_abs + inv(z_abs + d)
        return (φ / t, φ)
    end
    t_as = one(FT) / (one(FT) + FT(0.2316419) * z_abs)
    poly =
        t_as * (
            FT(0.319381530) +
            t_as * (
                FT(-0.356563782) +
                t_as * (
                    FT(1.781477937) +
                    t_as * (FT(-1.821255978) + t_as * FT(1.330274429))
                )
            )
        )
    return (φ * poly, φ)
end

"""
    normal_tail_prob(z)

Upper tail probability `Q(z) = 1 − Φ(z)` of the standard normal with
relative accuracy in the tail: for `z ≥ 2` the Laplace continued fraction
`Q(z) = φ(z) / (z + 1/(z + 2/(z + 3/(z + ⋯))))` evaluated to 20 levels in
`Float32` (relative error below `4 × 10⁻⁷`) and 32 levels in `Float64`
(`_mills_continued_fraction`, relative error below `4 × 10⁻⁹`), otherwise
`normal_cdf(−z)`, whose absolute error of `7.5 × 10⁻⁸` is a relative error
below `4 × 10⁻⁶` there. The conditional moments of a tail interval are
differences of such probabilities, which the absolute accuracy of `normal_cdf`
alone does not resolve beyond `z ≈ 2`.
"""
@inline function normal_tail_prob(z)
    FT = typeof(z)
    if z >= zero(FT)
        Q, _ = _normal_abs_tail_and_pdf(z)
        return Q
    end
    return normal_cdf(-z)
end

"""
    _gaussian_interval_moments4(a, b, Qa = normal_tail_prob(|a|), Qb = normal_tail_prob(|b|))

Moments `(I₀, I₁, I₂, I₃) = ∫ₐᵇ (1, x, x², x³) φ(x) dx` of the standard normal
density over `[a, b]`, `a ≤ b`, with `I₃ = (a² + 2) φ(a) − (b² + 2) φ(b)`;
the products with `φ` are taken as zero at infinite limits. `I₀` is the
difference of the tail probabilities of the tail nearer the interval, so that
the weight and moments of an interval far out in either tail keep their
relative accuracy: `Qa − Qb` above zero, `Qb − Qa` below it and `1 − Qa − Qb`
across it, from the upper tail probabilities `Qa = Q(|a|)`, `Qb = Q(|b|)`
(`normal_tail_prob`), which `conditional_rule` evaluates once per edge.
"""
@inline _gaussian_interval_moments4(a, b) = _gaussian_interval_moments4(
    a, b, normal_tail_prob(abs(a)), normal_tail_prob(abs(b)),
)
@inline function _gaussian_interval_moments4(a, b, Qa, Qb)
    FT = typeof(a)
    inv_sqrt2π = one(FT) / sqrt(FT(2) * FT(π))
    φa = ifelse(isfinite(a), exp(-a^2 / 2) * inv_sqrt2π, zero(FT))
    φb = ifelse(isfinite(b), exp(-b^2 / 2) * inv_sqrt2π, zero(FT))
    aφa = ifelse(isfinite(a), a * φa, zero(FT))
    bφb = ifelse(isfinite(b), b * φb, zero(FT))
    a²φa = ifelse(isfinite(a), a * aφa, zero(FT))
    b²φb = ifelse(isfinite(b), b * bφb, zero(FT))
    I₀ = ifelse(
        a >= zero(FT),
        Qa - Qb,
        ifelse(b <= zero(FT), Qb - Qa, one(FT) - Qb - Qa),
    )
    I₀ = max(I₀, zero(FT))
    I₁ = φa - φb
    I₂ = I₀ + aφa - bφb
    I₃ = 2 * I₁ + a²φa - b²φb
    return (I₀, I₁, I₂, I₃)
end

"""
    _truncated_normal_gauss2(a, b, Qa = normal_tail_prob(|a|), Qb = normal_tail_prob(|b|))

Two-point Gauss rule `(w, x₁, p₁, x₂, p₂)` of the standard normal truncated to
`[a, b]`: `w = Φ(b) − Φ(a)` is the probability of the interval and the nodes
`x₁ ≤ x₂` with probabilities `p₁ + p₂ = 1` reproduce the conditional mean `m`,
variance `v` and third central moment `c₃` of `x` on the interval. With
`s = c₃ / v` the node offsets `d = x − m` solve `d² − s d − v = 0`, and
`p₁ = d₂ / (d₂ − d₁)`, `p₂ = −d₁ / (d₂ − d₁)`; both nodes lie inside `[a, b]`.
Where the variance is below the rounding level of the moment differences
(`v ≤ 64 eps · max(m², 1)`), or the interval is empty, both nodes sit at the
mean with `p₁ = p₂ = ½`. `Qa`, `Qb` are the edge tail probabilities of
`_gaussian_interval_moments4`.
"""
@inline _truncated_normal_gauss2(a, b) = _truncated_normal_gauss2(
    a, b, normal_tail_prob(abs(a)), normal_tail_prob(abs(b)),
)
@inline function _truncated_normal_gauss2(a, b, Qa, Qb)
    FT = typeof(a)
    (I₀, I₁, I₂, I₃) = _gaussian_interval_moments4(a, b, Qa, Qb)
    empty = I₀ <= zero(FT)
    inv_I₀ = ifelse(empty, zero(FT), one(FT) / max(I₀, floatmin(FT)))
    m = I₁ * inv_I₀
    m2 = I₂ * inv_I₀
    m3 = I₃ * inv_I₀
    v = m2 - m^2
    c₃ = m3 - 3 * m * m2 + 2 * m^3
    degenerate = empty | (v <= 64 * eps(FT) * max(m^2, one(FT)))
    v_safe = ifelse(degenerate, one(FT), v)
    s = ifelse(degenerate, zero(FT), c₃ / v_safe)
    r = sqrt(s^2 + 4 * v_safe)
    d₁ = (s - r) / 2
    d₂ = (s + r) / 2
    p₂ = ifelse(degenerate, FT(0.5), -d₁ / r)
    p₁ = one(FT) - p₂
    x₁ = ifelse(degenerate, m, max(m + d₁, a))
    x₂ = ifelse(degenerate, m, min(m + d₂, b))
    return (I₀, x₁, p₁, x₂, p₂)
end

"""
    ConditionalNode{FT}

One node of the conditional-moment rule: its temperature `T` [K], the centred
saturation excess `S′` [kg/kg] the evaluator recovers at it, the probability
weight `w`, and the shaft flag (`1` inside the precipitation shaft, `0`
outside). The total water the evaluator receives is set per temperature point
by `_point_value`, which is also where the node's `S′` is recovered from.
"""
struct ConditionalNode{FT}
    T::FT
    S′::FT
    w::FT
    shaft::FT
end

"""
    ConditionalRule{N, FT}

The `N` nodes of the conditional-moment rule of one cell, in order of
increasing saturation excess; nodes of empty intervals carry zero weight and
the mean state. Built by `conditional_rule`, integrated by
`sum_over_conditional_nodes`.
"""
struct ConditionalRule{N, FT}
    nodes::SA.SVector{N, ConditionalNode{FT}}
end

"""
    sgs_temperature_regression(thp, T, ρ, σ_S, T′T′, q′q′, corr_Tq)

Regression slope `β_T = Cov(T′, S′) / σ_S² = r_TS σ_T / σ_S` [K / (kg/kg)] of
the temperature on the centred saturation excess `S′ = q′ − γ T′` of the linear
closure, with `γ = ∂q_sat/∂T`, `σ_S` the linearized width
(`_sgs_linear_sigma_S`), and `r_TS = (corr_Tq σ_q − γ σ_T) / σ_S` the
correlation of `T′` and `S′`. The correlation is clamped to `[−1, 1]`, so the
regressed temperature spread `|β_T| σ_S` stays within `σ_T` (Cauchy–Schwarz)
where rounding or the floor on `σ_S` puts the quotient outside that range.
"""
@inline function sgs_temperature_regression(thp, T, ρ, σ_S, T′T′, q′q′, corr_Tq)
    FT = typeof(σ_S)
    T′T′ <= zero(FT) && return zero(FT)
    (σ_q, σ_T, corr) = sgs_stddevs_and_correlation(q′q′, T′T′, corr_Tq)
    γ = FT(TD.∂q_vap_sat_∂T(_unwrapped(thp), T, ρ))
    σ_S_safe = max(σ_S, ϵ_numerics(FT))
    r_TS = clamp((corr * σ_q - γ * σ_T) / σ_S_safe, -one(FT), one(FT))
    return r_TS * (σ_T / σ_S_safe)
end

"""
    _sort3(a, b, c)

The three values in nondecreasing order.
"""
@inline function _sort3(a, b, c)
    lo = min(a, b)
    hi = max(a, b)
    return (min(lo, c), clamp(c, lo, hi), max(hi, c))
end

"""
    _conditional_interval_nodes(T, σ_S, β_T, lo, hi, Qlo, Qhi, cloudy, shaft, T_min)

The two nodes of the rule on the interval `[lo, hi]` of the centred saturation
excess, with the upper tail probabilities `Qlo`, `Qhi` of its standardized
edges: the two-point Gauss rule where the interval carries actual cloud
(`cloudy`), otherwise one node at the conditional mean with the full weight
and a second node of zero weight. The node temperature is the regression
`T + β_T S′` floored at `T_min`.
"""
@inline function _conditional_interval_nodes(
    T, σ_S, β_T, lo, hi, Qlo, Qhi, cloudy, shaft, T_min,
)
    FT = typeof(σ_S)
    (w, x₁, p₁, x₂, p₂) = _truncated_normal_gauss2(lo / σ_S, hi / σ_S, Qlo, Qhi)
    # the conditional mean is the probability-weighted node position
    m = p₁ * x₁ + p₂ * x₂
    S′₁ = σ_S * ifelse(cloudy, x₁, m)
    S′₂ = σ_S * ifelse(cloudy, x₂, m)
    w₁ = w * ifelse(cloudy, p₁, one(FT))
    w₂ = w * ifelse(cloudy, p₂, zero(FT))
    return (
        _conditional_node(T, β_T, S′₁, w₁, shaft, T_min),
        _conditional_node(T, β_T, S′₂, w₂, shaft, T_min),
    )
end

"""
    _conditional_node(T, β_T, S′, w, shaft, T_min)

The `ConditionalNode` at centred saturation excess `S′` [kg/kg] with
probability weight `w`: its temperature is the regression on `S′`,
`T + β_T S′` floored at `T_min` [K]. The Boolean `shaft` becomes the node's
shaft flag, `1` or `0`.
"""
@inline function _conditional_node(T, β_T, S′, w, shaft, T_min)
    FT = typeof(S′)
    T_node = max(T_min, T + β_T * S′)
    return ConditionalNode(T_node, S′, w, ifelse(shaft, one(FT), zero(FT)))
end

"""
    conditional_rule(T, σ_S, β_T, t_a, t_e, t_p, T_min)

The `ConditionalRule` of a cell with mean temperature `T` [K], linearized width
`σ_S` [kg/kg] of the centred excess `S′`, temperature regression slope `β_T`
(`sgs_temperature_regression`) and the thresholds on `S′` [kg/kg], each of
which may be infinite:

  - `t_a`: actual cloud edge `−λ/α`; `S′ ≥ t_a` holds condensate,
  - `t_e`: equilibrium cloud edge `−λ_eq`,
  - `t_p`: precipitation shaft edge (`sgs_precip_shaft_threshold`); `S′ ≥ t_p`
    is the shaft.

The thresholds, sorted, bound four intervals; each gets the nodes of
`_conditional_interval_nodes`, two Gauss nodes where it carries actual cloud and
one node at its conditional mean otherwise, for eight node slots in all. The
weights sum to one; an interval of zero width contributes zero weight.
"""
@inline function conditional_rule(T, σ_S, β_T, t_a, t_e, t_p, T_min)
    FT = typeof(σ_S)
    # A sliver interval between nearly coincident actual and equilibrium
    # cloud edges, |t_e − t_a| < 10⁻³ σ_S, collapses onto the actual edge. Its
    # moments are differences of nearly equal tail integrals, so its
    # conditional mean is rounding noise and its node could land far outside
    # the interval with a small but nonzero weight.
    t_e_guarded = ifelse(
        isfinite(t_e) & isfinite(t_a) &
        (abs(FT(t_e) - FT(t_a)) < FT(1e-3) * FT(σ_S)),
        FT(t_a),
        FT(t_e),
    )
    (b₁, b₂, b₃) = _sort3(FT(t_a), t_e_guarded, FT(t_p))
    bounds = (FT(-Inf), b₁, b₂, b₃, FT(Inf))
    # the upper tail probability Q(|b|/σ_S) of each edge, shared by the two
    # intervals it bounds (zero at the infinite ends)
    Qs = (
        zero(FT),
        normal_tail_prob(abs(b₁ / σ_S)),
        normal_tail_prob(abs(b₂ / σ_S)),
        normal_tail_prob(abs(b₃ / σ_S)),
        zero(FT),
    )
    nodes = ntuple(Val(4)) do i
        lo = bounds[i]
        hi = bounds[i + 1]
        # an interval between sorted thresholds lies entirely on one side of
        # each threshold, so its lower bound decides every flag
        _conditional_interval_nodes(
            T, σ_S, β_T, lo, hi, Qs[i], Qs[i + 1],
            lo >= t_a, lo >= t_p, T_min,
        )
    end
    return ConditionalRule(
        SA.SVector{8}(
            nodes[1][1], nodes[1][2], nodes[2][1], nodes[2][2],
            nodes[3][1], nodes[3][2], nodes[4][1], nodes[4][2],
        ),
    )
end

# Register-compact 8-node rule storing only `(S′, w)` and a 1-byte shaft
# bitmask. Temperature is reconstructed on demand from `T + β_T * S′`, and
# saturation specific humidity is deferred to positive-weight wet points.
struct _CompactConditionalRule{FT}
    S′::NTuple{8, FT}
    w::NTuple{8, FT}
    shaft_mask::UInt8
end

@inline function _compact_conditional_interval_nodes(
    σ_S::FT,
    a::FT,
    b::FT,
    Qa::FT,
    φa::FT,
    Qb::FT,
    φb::FT,
    cloudy::Bool,
) where {FT}
    z = zero(FT)
    if !(a < b)
        return (z, z, z, z)
    end
    I₀ = max(
        z,
        ifelse(a >= z, Qa - Qb, ifelse(b <= z, Qb - Qa, one(FT) - Qb - Qa)),
    )
    if I₀ <= z
        return (z, z, z, z)
    end
    inv_I₀ = one(FT) / max(I₀, floatmin(FT))
    I₁ = φa - φb
    m = I₁ * inv_I₀
    if !cloudy
        S′_m = σ_S * m
        return (S′_m, I₀, S′_m, z)
    end
    aφa = ifelse(isfinite(a), a * φa, z)
    bφb = ifelse(isfinite(b), b * φb, z)
    a²φa = ifelse(isfinite(a), a * aφa, z)
    b²φb = ifelse(isfinite(b), b * bφb, z)
    I₂ = I₀ + aφa - bφb
    I₃ = FT(2) * I₁ + a²φa - b²φb
    m2 = I₂ * inv_I₀
    m3 = I₃ * inv_I₀
    v = m2 - m^2
    c₃ = m3 - FT(3) * m * m2 + FT(2) * m^3
    degenerate = v <= 64 * eps(FT) * max(m^2, one(FT))
    v_safe = ifelse(degenerate, one(FT), v)
    s = ifelse(degenerate, z, c₃ / v_safe)
    r = sqrt(s^2 + FT(4) * v_safe)
    d₁ = (s - r) * FT(0.5)
    d₂ = (s + r) * FT(0.5)
    p₂ = ifelse(degenerate, FT(0.5), -d₁ / r)
    p₁ = one(FT) - p₂
    x₁ = ifelse(degenerate, m, max(m + d₁, a))
    x₂ = ifelse(degenerate, m, min(m + d₂, b))
    return (σ_S * x₁, I₀ * p₁, σ_S * x₂, I₀ * p₂)
end

@inline function _compact_conditional_rule(
    σ_S::FT,
    t_a::FT,
    t_e::FT,
    t_p::FT,
) where {FT}
    t_e_guarded = ifelse(
        isfinite(t_e) & isfinite(t_a) & (abs(t_e - t_a) < FT(1e-3) * σ_S),
        t_a,
        t_e,
    )
    (b₁, b₂, b₃) = _sort3(t_a, t_e_guarded, t_p)
    z = zero(FT)
    inv_σ = one(FT) / σ_S
    z₁ = b₁ * inv_σ
    z₂ = b₂ * inv_σ
    z₃ = b₃ * inv_σ
    (Q₁, φ₁) = _normal_abs_tail_and_pdf(z₁)
    (Q₂, φ₂) = if !isfinite(b₂)
        (z, z)
    elseif b₂ == b₁
        (Q₁, φ₁)
    else
        _normal_abs_tail_and_pdf(z₂)
    end
    (Q₃, φ₃) = if !isfinite(b₃)
        (z, z)
    elseif b₃ == b₂
        (Q₂, φ₂)
    else
        _normal_abs_tail_and_pdf(z₃)
    end
    (S1, w1, S2, w2) = _compact_conditional_interval_nodes(
        σ_S,
        FT(-Inf),
        z₁,
        z,
        z,
        Q₁,
        φ₁,
        FT(-Inf) >= t_a,
    )
    (S3, w3, S4, w4) = _compact_conditional_interval_nodes(
        σ_S,
        z₁,
        z₂,
        Q₁,
        φ₁,
        Q₂,
        φ₂,
        b₁ >= t_a,
    )
    (S5, w5, S6, w6) = _compact_conditional_interval_nodes(
        σ_S,
        z₂,
        z₃,
        Q₂,
        φ₂,
        Q₃,
        φ₃,
        b₂ >= t_a,
    )
    (S7, w7, S8, w8) = _compact_conditional_interval_nodes(
        σ_S,
        z₃,
        FT(Inf),
        Q₃,
        φ₃,
        z,
        z,
        b₃ >= t_a,
    )
    sh1 = ifelse(FT(-Inf) >= t_p, UInt8(0x03), UInt8(0x00))
    sh2 = ifelse(b₁ >= t_p, UInt8(0x0c), UInt8(0x00))
    sh3 = ifelse(b₂ >= t_p, UInt8(0x30), UInt8(0x00))
    sh4 = ifelse(b₃ >= t_p, UInt8(0xc0), UInt8(0x00))
    return _CompactConditionalRule{FT}(
        (S1, S2, S3, S4, S5, S6, S7, S8),
        (w1, w2, w3, w4, w5, w6, w7, w8),
        sh1 | sh2 | sh3 | sh4,
    )
end

@inline function _compact_single_node_rule(::Type{FT}) where {FT}
    z = zero(FT)
    return _CompactConditionalRule{FT}(
        (z, z, z, z, z, z, z, z),
        (one(FT), z, z, z, z, z, z, z),
        UInt8(0x01),
    )
end

# Register-only binary selection over an 8-tuple avoids NVPTX `.local` spills.
@inline function _select_tuple8(k::Int32, t::NTuple{8, FT}) where {FT}
    v12 = ifelse(k == Int32(1), t[1], t[2])
    v34 = ifelse(k == Int32(3), t[3], t[4])
    v56 = ifelse(k == Int32(5), t[5], t[6])
    v78 = ifelse(k == Int32(7), t[7], t[8])
    v14 = ifelse(k <= Int32(2), v12, v34)
    v58 = ifelse(k <= Int32(6), v56, v78)
    return ifelse(k <= Int32(4), v14, v58)
end

@inline function _compact_rule_mean(
    g,
    rule::_CompactConditionalRule{FT},
) where {FT}
    S = rule.S′
    w = rule.w
    return muladd(
        w[1],
        g(S[1]),
        muladd(
            w[2],
            g(S[2]),
            muladd(
                w[3],
                g(S[3]),
                muladd(
                    w[4],
                    g(S[4]),
                    muladd(
                        w[5],
                        g(S[5]),
                        muladd(
                            w[6],
                            g(S[6]),
                            muladd(w[7], g(S[7]), w[8] * g(S[8])),
                        ),
                    ),
                ),
            ),
        ),
    )
end

@inline function _compact_rule_shaft_fraction(
    rule::_CompactConditionalRule{FT},
) where {FT}
    w = rule.w
    m = rule.shaft_mask
    z = zero(FT)
    return ifelse((m & UInt8(0x01)) != UInt8(0), w[1], z) +
           ifelse((m & UInt8(0x02)) != UInt8(0), w[2], z) +
           ifelse((m & UInt8(0x04)) != UInt8(0), w[3], z) +
           ifelse((m & UInt8(0x08)) != UInt8(0), w[4], z) +
           ifelse((m & UInt8(0x10)) != UInt8(0), w[5], z) +
           ifelse((m & UInt8(0x20)) != UInt8(0), w[6], z) +
           ifelse((m & UInt8(0x40)) != UInt8(0), w[7], z) +
           ifelse((m & UInt8(0x80)) != UInt8(0), w[8], z)
end

"""
    sgs_temperature_spread_at_fixed_excess(T′T′, β_T, σ_S)

Standard deviation `σ⊥ = √(Var(T′) − β_T² σ_S²)` [K] of the temperature at
fixed centred saturation excess: the part of the temperature spread the
regression on `S′` (`sgs_temperature_regression`) does not carry.
"""
@inline function sgs_temperature_spread_at_fixed_excess(T′T′, β_T, σ_S)
    FT = typeof(σ_S)
    return sqrt(max(FT(T′T′) - (β_T * σ_S)^2, zero(FT)))
end

"""
    TemperatureSpread(thp, ρ, mu_S, σ⊥, T_freeze, T_min, q_max)

What `sum_over_conditional_nodes` needs to place temperature points at fixed
saturation excess: the spread `σ⊥` [K], from
`sgs_temperature_spread_at_fixed_excess`; the freezing point; and the density,
mean excess and limits with which the point's total water is set so that the
evaluator recovers the node's `S′`.
"""
struct TemperatureSpread{FT, P}
    thp::P
    ρ::FT
    mu_S::FT
    σ⊥::FT
    T_freeze::FT
    T_min::FT
    q_max::FT
end

"""
    _freezing_split(T_node, σ⊥, T_freeze)

Split `(p₁, t₁, p₂, t₂)` of the temperature spread at fixed saturation excess,
`T′ ~ N(0, σ⊥²)` about `T_node`, at the freezing point: where both sides of
`T_freeze` hold more than 2 % of the spread, the probabilities `p₁ = Φ(t★)`,
`p₂ = 1 − p₁` of the two sides, `t★ = (T_freeze − T_node)/σ⊥`, and the
conditional means `t₁ = −φ(t★)/p₁`, `t₂ = φ(t★)/p₂` of the half-normals, in
units of `σ⊥`; otherwise a single point, `(1, 0, 0, 0)`. Both forms keep the
mean temperature.
"""
@inline function _freezing_split(T_node, σ⊥, T_freeze)
    FT = typeof(T_node)
    z = zero(FT)
    if !(σ⊥ > ϵ_numerics(FT)) || !(abs(T_freeze - T_node) < FT(2.06) * σ⊥)
        return (one(FT), z, z, z)
    end
    t★ = (T_freeze - T_node) / σ⊥
    inv_sqrt2π = one(FT) / sqrt(FT(2) * FT(π))
    t_abs = abs(t★)
    φ★ = exp(-FT(0.5) * t_abs * t_abs) * inv_sqrt2π
    t_as = one(FT) / (one(FT) + FT(0.2316419) * t_abs)
    poly =
        t_as * (
            FT(0.319381530) +
            t_as * (
                FT(-0.356563782) +
                t_as * (
                    FT(1.781477937) +
                    t_as * (FT(-1.821255978) + t_as * FT(1.330274429))
                )
            )
        )
    q = φ★ * poly
    p₁ = ifelse(t★ >= z, one(FT) - q, q)
    p₂ = one(FT) - p₁
    split = (p₁ > FT(0.02)) & (p₂ > FT(0.02))
    return (
        ifelse(split, p₁, one(FT)),
        ifelse(split, -φ★ / max(p₁, eps(FT)), z),
        ifelse(split, p₂, z),
        ifelse(split, φ★ / max(p₂, eps(FT)), z),
    )
end

"""
    _temperature_points(T_node, spread)

Points `((p₁, t₁), (p₂, t₂))` in temperature at fixed saturation excess, with
probabilities `p` and offsets `t` in units of `σ⊥`, for a node at `T_node`
(`_freezing_split`). Away from the freezing point the first point carries the
whole probability at `t = 0` and the second has zero weight.
"""
@inline function _temperature_points(T_node, spread)
    (p₁, t₁, p₂, t₂) = _freezing_split(T_node, spread.σ⊥, spread.T_freeze)
    return ((p₁, t₁), (p₂, t₂))
end

"""
    _point_value(f, n, t, spread)

`f(T, q, shaft, q_sat)` at the temperature point of the node `n` offset by `t`
[-], in units of the spread `σ⊥` at fixed saturation excess
(`_temperature_points`): the point's temperature is `n.T + t σ⊥` floored at
`spread.T_min` [K], and its total water is set from the `TemperatureSpread`
`spread` so that the evaluator recovers the node's `S′`, clamped to
`[0, spread.q_max]` [kg/kg]. The saturation specific humidity the point's total
water was built from is passed to `f`, which needs the same value to recover
`S′`. The value is wrapped for accumulation (`add_auto_broadcasters`), so `f`
may return a scalar or a `NamedTuple` of scalars.
"""
@inline function _point_value(f, n, t, spread)
    FT = typeof(n.w)
    T_pt = max(spread.T_min, n.T + t * spread.σ⊥)
    q_sat_pt = TD.q_vap_saturation(_unwrapped(spread.thp), T_pt, spread.ρ)
    q_pt = clamp(q_sat_pt + spread.mu_S + n.S′, zero(FT), spread.q_max)
    return add_auto_broadcasters(f(T_pt, q_pt, n.shaft, q_sat_pt))
end

@inline function _sum_node_points(point_fn::PF, f, n, spread, acc) where {PF}
    if n.w > zero(n.w)
        (p₁, t₁), (p₂, t₂) = _temperature_points(n.T, spread)
        w₁ = n.w * p₁
        if w₁ > zero(w₁)
            acc = acc + point_fn(f, n, t₁, spread) * w₁
        end
        w₂ = n.w * p₂
        if w₂ > zero(w₂)
            acc = acc + point_fn(f, n, t₂, spread) * w₂
        end
    end
    return acc
end

# Compile-time unrolling keeps `ConditionalRule.nodes` in NVPTX registers.
@inline function _sum_over_conditional_nodes_unrolled(
    point_fn::PF,
    f,
    rule::ConditionalRule{N},
    spread::TemperatureSpread,
) where {PF, N}
    nodes = rule.nodes
    @inbounds begin
        n₁ = nodes[1]
        ((p₁, t₁), (p₂, t₂)) = _temperature_points(n₁.T, spread)
        acc = point_fn(f, n₁, t₁, spread) * (n₁.w * p₁)
        w₁₂ = n₁.w * p₂
        if w₁₂ > zero(w₁₂)
            acc = acc + point_fn(f, n₁, t₂, spread) * w₁₂
        end
        if N == 8
            acc = _sum_node_points(point_fn, f, nodes[2], spread, acc)
            acc = _sum_node_points(point_fn, f, nodes[3], spread, acc)
            acc = _sum_node_points(point_fn, f, nodes[4], spread, acc)
            acc = _sum_node_points(point_fn, f, nodes[5], spread, acc)
            acc = _sum_node_points(point_fn, f, nodes[6], spread, acc)
            acc = _sum_node_points(point_fn, f, nodes[7], spread, acc)
            acc = _sum_node_points(point_fn, f, nodes[8], spread, acc)
        else
            for k in 2:N
                acc = _sum_node_points(point_fn, f, nodes[k], spread, acc)
            end
        end
    end
    return drop_auto_broadcasters(acc)
end

"""
    sum_over_conditional_nodes(f, rule, spread)

Weighted sum of `f(T, q, shaft, q_sat)` over the nodes of a `ConditionalRule` and,
for each node, over its points in temperature at fixed saturation excess
(`_temperature_points`, with the `TemperatureSpread` `spread`); a point's
total water is set so that the evaluator recovers the node's `S′`. The first
point of the first node is always evaluated (it seeds the accumulator); every
other point only where its weight is positive.
"""
@inline function sum_over_conditional_nodes(
    f,
    rule::ConditionalRule{N},
    spread::TemperatureSpread,
) where {N}
    return _sum_over_conditional_nodes_unrolled(_point_value, f, rule, spread)
end

"""
    conditional_rule_shaft_fraction(rule)

`Σᵣ wᵣ shaftᵣ`, the weight of the precipitation shaft over the nodes of a
`ConditionalRule`, from the nodes' own shaft flags, which the evaluator
receives. A node of the Gauss pair that sits on the shaft edge belongs to the
shaft by its interval, whatever the rounding of its `S′`, so the placement
normalized with this fraction conserves the cell-mean precipitation exactly.
"""
@inline function conditional_rule_shaft_fraction(rule::ConditionalRule{N}) where {N}
    nodes = rule.nodes
    acc = zero(nodes[1].w)
    @inbounds for k in 1:N
        acc += nodes[k].w * nodes[k].shaft
    end
    return acc
end

"""
    conditional_rule_mean(g, rule)

`Σᵣ wᵣ g(S′ᵣ)` over the nodes of a `ConditionalRule`, for the scalar function
`g` of the node saturation excess, such as the condensate mean the driver
normalizes with (`NodeCondensate`).
"""
@inline function conditional_rule_mean(g, rule::ConditionalRule{N}) where {N}
    nodes = rule.nodes
    acc = zero(nodes[1].w)
    @inbounds for k in 1:N
        acc += nodes[k].w * g(nodes[k].S′)
    end
    return acc
end

"""
    NodeCondensate(λ, α)

Functor `S′ ↦ max(0, λ + α S′)`, the reconstructed condensate of a node, for
`conditional_rule_mean`.
"""
struct NodeCondensate{FT}
    λ::FT
    α::FT
end
@inline (f::NodeCondensate)(S′) = max(zero(f.λ), f.λ + f.α * S′)

"""
    _continuous_lagrange(q_c, σ)

Lagrange multiplier `λ` of the continuous truncated-Gaussian constraint
`E[max(0, λ + S′)] = q_c`, `S′ ~ N(0, σ²)`: the seed of `_compute_z` refined
by two further Newton steps, which bring the residual of the seed (up to a few
percent of `q_c` at small `q_c / σ`) to `≲ 1e-8 σ`, a few `1e-6` of `q_c` in the
deep tail.
"""
@inline function _continuous_lagrange(q_c, σ)
    FT = typeof(q_c)
    inv_sqrt2π = one(FT) / sqrt(FT(2) * FT(π))
    C = q_c / σ
    z = _compute_z(C)
    for _ in 1:2
        Φz = normal_cdf(z)
        φz = exp(-z^2 / 2) * inv_sqrt2π
        z_new = (C - φz) / max(Φz, ϵ_numerics(FT))
        z = ifelse(Φz > ϵ_numerics(FT), z_new, z)
    end
    return z * σ
end
