# ============================================================================
# Conditional-moment node rule for the 1M SGS microphysics
# ============================================================================
#
# With the Gaussian closure every quantity the 1M evaluator reads at a node is
# a function of one random variable, the centred saturation excess
# S′ ~ N(0, σ_S²), and of the temperature regressed on it. Three interior
# thresholds partition S′ in the default 8-node rule:
#
#     t_a = −λ / α        actual cloud edge:       q_c(S′) = max(0, λ + α S′)
#     t_e = −λ_eq         equilibrium cloud edge:  q_c,eq(S′) = max(0, λ_eq + S′)
#     t_p = S_star        precipitation shaft edge (the moistest a_p of the PDF)
#
# Sorting them gives four intervals on (-Inf, Inf), on each of which actual
# condensate and equilibrium excess are linear in S′ and the shaft flag is
# constant. The Gaussian interval moments give each interval its probability weight
# and the conditional moments of S′; a two-point Gauss rule on each interval
# yields eight nodes in all. A 6-node variant with two thresholds (t_a, t_p) is
# also available for two-threshold ablation comparisons.
#
# The temperature also spreads at fixed S′. That spread, σ⊥, is the part of the
# temperature variance the regression on S′ does not carry (q′ moves with
# ∂q_sat/∂T T′ so that S′ stays fixed). The rates respond to it mainly through
# the kink at the freezing point (melting of snow, rain–snow collisions), so a
# node whose spread straddles T_freeze is split there; the temperature points
# are generated in the sum over the nodes, so the rule itself keeps its slots.

"""
    normal_tail_prob(z)

Upper tail probability `Q(z) = 1 − Φ(z)` of the standard normal with
relative accuracy in the tail: for `z ≥ 2` the Laplace continued fraction
`Q(z) = φ(z) / (z + 1/(z + 2/(z + 3/(z + ⋯))))` to 32 terms (relative error
below `4 × 10⁻⁹`), otherwise `normal_cdf(−z)`, whose absolute error of
`7.5 × 10⁻⁸` is a relative error below `4 × 10⁻⁶` there. The conditional
moments of a tail interval are differences of such probabilities, which the
absolute accuracy of `normal_cdf` alone does not resolve beyond `z ≈ 2`.
"""
@inline function normal_tail_prob(z)
    FT = typeof(z)
    inv_sqrt2π = one(FT) / sqrt(FT(2) * FT(π))
    z_cf = max(z, FT(2))
    t = z_cf
    for k in 32:-1:1
        t = z_cf + FT(k) / t
    end
    q_cf = exp(-z_cf^2 / 2) * inv_sqrt2π / t
    return ifelse(z >= FT(2), q_cf, normal_cdf(-z))
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

One node of the conditional-moment rule: the state `(T, q)` [K, kg/kg] the
evaluator receives, the centred saturation excess `S′` [kg/kg] it recovers
from that state, the probability weight `w`, and the shaft flag (`1` inside
the precipitation shaft, `0` outside).
"""
struct ConditionalNode{FT}
    T::FT
    q::FT
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

Regression slope `β_T = Cov(T′, S′) / σ_S²` [K / (kg/kg)] of the temperature
on the centred saturation excess `S′ = q′ − γ T′` of the linear closure,
`γ = ∂q_sat/∂T`, with `Cov(T′, S′) = Cov(T′, q′) − γ Var(T′)` and `σ_S` the
linearized width (`_sgs_linear_sigma_S`). Since `|Cov(T′, S′)| ≤ σ_T σ_S`, the
regressed temperature spread `|β_T| σ_S` never exceeds `σ_T`.
"""
@inline function sgs_temperature_regression(thp, T, ρ, σ_S, T′T′, q′q′, corr_Tq)
    FT = typeof(σ_S)
    (σ_q, σ_T, corr) = sgs_stddevs_and_correlation(q′q′, T′T′, corr_Tq)
    γ = FT(TD.∂q_vap_sat_∂T(_unwrapped(thp), T, ρ))
    return (corr * σ_T * σ_q - γ * σ_T^2) / max(FT(σ_S), ϵ_numerics(FT))^2
end

"""
    _sort2(a, b)

The two values in nondecreasing order.
"""
@inline _sort2(a, b) = (min(a, b), max(a, b))

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
    _conditional_interval_nodes(thp, ρ, T, mu_S, σ_S, β_T, lo, hi, Qlo, Qhi, cloudy, shaft, T_min, q_max)

The two nodes of the rule on the interval `[lo, hi]` of the centred saturation
excess, with the upper tail probabilities `Qlo`, `Qhi` of its standardized
edges: the two-point Gauss rule where the interval carries actual cloud
(`cloudy`), otherwise one node at the conditional mean with the full weight
and a second node of zero weight. The node temperature is the regression
`T + β_T S′` floored at `T_min`, and the node total water is set so that the
evaluator, which recovers `S′ = q − q_sat(T_node, ρ) − μ_S`, lands on the
node's `S′` exactly; `q` is clamped to `[0, q_max]`.
"""
@inline function _conditional_interval_nodes(
    thp, ρ, T, mu_S, σ_S, β_T, lo, hi, Qlo, Qhi, cloudy, shaft, T_min, q_max,
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
        _conditional_node(thp, ρ, T, mu_S, β_T, S′₁, w₁, shaft, T_min, q_max),
        _conditional_node(thp, ρ, T, mu_S, β_T, S′₂, w₂, shaft, T_min, q_max),
    )
end

"""
    _conditional_node(thp, ρ, T, mu_S, β_T, S′, w, shaft, T_min, q_max)

The `ConditionalNode` at centred saturation excess `S′` [kg/kg] with
probability weight `w`: its temperature is the regression on `S′`,
`T + β_T S′` floored at `T_min` [K], and its total water is set so that the
evaluator, which recovers `S′ = q − q_sat(T_node, ρ) − μ_S`, lands on the
node's `S′` exactly, clamped to `[0, q_max]` [kg/kg]. The Boolean `shaft` becomes
the node's shaft flag, `1` or `0`.
"""
@inline function _conditional_node(thp, ρ, T, mu_S, β_T, S′, w, shaft, T_min, q_max)
    FT = typeof(S′)
    T_node = max(T_min, T + β_T * S′)
    q_node = clamp(
        TD.q_vap_saturation(_unwrapped(thp), T_node, ρ) + mu_S + S′, zero(FT), q_max,
    )
    return ConditionalNode(T_node, q_node, S′, w, ifelse(shaft, one(FT), zero(FT)))
end

"""
    conditional_rule(
        thp, ρ, T, mu_S, σ_S, β_T, t_a, t_p, T_min, q_max,
    )

The 6-node `ConditionalRule` of a cell with mean temperature `T` [K], density `ρ`,
linearized mean saturation excess `mu_S` [kg/kg], linearized width `σ_S` [kg/kg]
of the centred excess `S′`, temperature regression slope `β_T`
(`sgs_temperature_regression`) and the two interior thresholds on `S′` [kg/kg]:

  - `t_a`: actual cloud edge `−λ/α`; `S′ ≥ t_a` holds condensate,
  - `t_p`: precipitation shaft edge (`sgs_precip_shaft_threshold`); `S′ ≥ t_p`
    is the shaft.

The thresholds, sorted, bound three intervals on `(-Inf, Inf)`; each gets the
nodes of `_conditional_interval_nodes`, two Gauss nodes where it carries actual
cloud and one node at its conditional mean otherwise, for six node slots in all.
The weights sum to one; an interval of zero width contributes zero weight.
"""
@inline function conditional_rule(
    thp, ρ, T, mu_S, σ_S, β_T, t_a, t_p, T_min, q_max,
)
    FT = typeof(σ_S)
    (b₁, b₂) = _sort2(FT(t_a), FT(t_p))
    bounds = (FT(-Inf), b₁, b₂, FT(Inf))
    Qs = (
        zero(FT),
        normal_tail_prob(abs(b₁ / σ_S)),
        normal_tail_prob(abs(b₂ / σ_S)),
        zero(FT),
    )
    nodes = ntuple(Val(3)) do i
        lo = bounds[i]
        hi = bounds[i + 1]
        _conditional_interval_nodes(
            thp, ρ, T, mu_S, σ_S, β_T, lo, hi, Qs[i], Qs[i + 1],
            lo >= t_a, lo >= t_p, T_min, q_max,
        )
    end
    return ConditionalRule(
        SA.SVector{6}(
            nodes[1][1], nodes[1][2],
            nodes[2][1], nodes[2][2],
            nodes[3][1], nodes[3][2],
        ),
    )
end

"""
    conditional_rule(
        thp, ρ, T, mu_S, σ_S, β_T, t_a, t_e, t_p, T_min, q_max,
    )

The `ConditionalRule` of a cell with mean temperature `T` [K], density `ρ`,
linearized mean saturation excess `mu_S` [kg/kg], linearized width `σ_S` [kg/kg]
of the centred excess `S′`, temperature regression slope `β_T`
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
@inline function conditional_rule(
    thp, ρ, T, mu_S, σ_S, β_T, t_a, t_e, t_p, T_min, q_max,
)
    FT = typeof(σ_S)
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
            thp, ρ, T, mu_S, σ_S, β_T, lo, hi, Qs[i], Qs[i + 1],
            lo >= t_a, lo >= t_p, T_min, q_max,
        )
    end
    return ConditionalRule(
        SA.SVector{8}(
            nodes[1][1], nodes[1][2], nodes[2][1], nodes[2][2],
            nodes[3][1], nodes[3][2], nodes[4][1], nodes[4][2],
        ),
    )
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
    t★ = (T_freeze - T_node) / max(σ⊥, ϵ_numerics(FT))
    p₁ = normal_cdf(t★)
    p₂ = one(FT) - p₁
    φ★ = exp(-t★ * t★ / 2) / sqrt(2 * FT(π))
    split = (σ⊥ > ϵ_numerics(FT)) & (p₁ > FT(0.02)) & (p₂ > FT(0.02))
    return (
        ifelse(split, p₁, one(FT)),
        ifelse(split, -φ★ / max(p₁, eps(FT)), zero(FT)),
        ifelse(split, p₂, zero(FT)),
        ifelse(split, φ★ / max(p₂, eps(FT)), zero(FT)),
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

`f(T, q, shaft)` at the temperature point of the node `n` offset by `t` [-],
in units of the spread `σ⊥` at fixed saturation excess (`_temperature_points`):
the point's temperature is `n.T + t σ⊥` floored at `spread.T_min` [K], and its
total water is set from the `TemperatureSpread` `spread` so that the evaluator
recovers the node's `S′`, clamped to `[0, spread.q_max]` [kg/kg]. The value is
wrapped for accumulation (`add_auto_broadcasters`), so `f` may return a
scalar or a `NamedTuple` of scalars.
"""
@inline function _point_value(f, n, t, spread)
    FT = typeof(n.w)
    T_pt = max(spread.T_min, n.T + t * spread.σ⊥)
    q_pt = clamp(
        TD.q_vap_saturation(_unwrapped(spread.thp), T_pt, spread.ρ) + spread.mu_S + n.S′,
        zero(FT),
        spread.q_max,
    )
    return add_auto_broadcasters(f(T_pt, q_pt, n.shaft))
end

"""
    sum_over_conditional_nodes(f, rule, spread)

Weighted sum of `f(T, q, shaft)` over the nodes of a `ConditionalRule` and,
for each node, over its points in temperature at fixed saturation excess
(`_temperature_points`, with the `TemperatureSpread` `spread`); a point's
total water is set so that the evaluator recovers the node's `S′`. The first
point of the first node is always evaluated (it seeds the accumulator); every
other point only where its weight is positive.
"""
function sum_over_conditional_nodes(
    f, rule::ConditionalRule{N}, spread::TemperatureSpread,
) where {N}
    nodes = rule.nodes
    @inbounds begin
        n₁ = nodes[1]
        pts = _temperature_points(n₁.T, spread)
        (p, t) = pts[1]
        acc = _point_value(f, n₁, t, spread) * (n₁.w * p)
        for j in 2:length(pts)
            (p, t) = pts[j]
            if n₁.w * p > zero(p)
                acc += _point_value(f, n₁, t, spread) * (n₁.w * p)
            end
        end
        for k in 2:N
            nₖ = nodes[k]
            if nₖ.w > zero(nₖ.w)
                pts = _temperature_points(nₖ.T, spread)
                for j in 1:length(pts)
                    (p, t) = pts[j]
                    if p > zero(p)
                        acc += _point_value(f, nₖ, t, spread) * (nₖ.w * p)
                    end
                end
            end
        end
    end
    return drop_auto_broadcasters(acc)
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
