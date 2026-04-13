#####
##### Subgrid / subcell variance statistics (pure scalar kernels)
#####
##### Used by SGS quadrature and precomputed quantities.  See
##### `variance_statistics_reference.md` for PDF / moment-matched Gaussian discussion.

import SpecialFunctions: erf

"""
    ϵ_variance_statistics(FT)

Numerical floor for variance–correlation algebra (same idea as `ϵ_numerics` in
`microphysics_cache.jl`).
"""
@inline ϵ_variance_statistics(FT) = cbrt(floatmin(FT))

"""
    subcell_geometric_variance_increment(Δz, dqdz_sq, dTdz_sq)

Return `(Δq′q′, ΔT′T′)` from subcell linear reconstruction over thickness `Δz`
(``(1/12) Δz^2 (∂q/∂z)^2`` and ``(1/12) Δz^2 (∂T/∂z)^2``). Arguments `dqdz_sq`
and `dTdz_sq` are **squared** vertical derivatives in quadrature variables.
"""
@inline function subcell_geometric_variance_increment(Δz, dqdz_sq, dTdz_sq)
    geom = (one(typeof(Δz)) / 12) * Δz^2
    return geom * dqdz_sq, geom * dTdz_sq
end

"""
    subcell_geometric_covariance_Tq(Δz, ∂T∂θ_li, dot_wq_wθ)

Subcell geometric contribution to ``\\mathrm{Cov}(T', q')`` for a linear-in-`z` profile:
``(1/12) Δz^2 (∂T/∂z)(∂q/∂z)`` with ``∂T/∂z ≈ (∂T/∂θ_{li})(∂θ_{li}/∂z)``.
Here `dot_wq_wθ` is `dot(WVector(∇q), WVector(∇θ_{li}))` (column geometry).
"""
@inline function subcell_geometric_covariance_Tq(Δz, ∂T∂θ_li, dot_wq_wθ)
    return (one(typeof(Δz)) / 12) * Δz^2 * ∂T∂θ_li * dot_wq_wθ
end

"""
    effective_sgs_quadrature_moments_matched_gaussian(
        q′q′, T′T′, ρ_param, ε, Δz, w_grad_q_sq, w_grad_θ_sq, ∂T∂θ_li, wq_dot_wθ,
    )

Moment-matched **Gaussian** effective variances and correlation for SGS quadrature
when subcell linear-in-`z` geometry is folded into the moments passed to
Gauss–Hermite quadrature (same algebra previously implemented in
`materialize_sgs_quadrature_moments!`).

Returns `(q_var_eff, T_var_eff, ρ_eff)` with `ρ_eff ∈ [-1, 1]`.
"""
@inline function effective_sgs_quadrature_moments_matched_gaussian(
    q′q′,
    T′T′,
    ρ_param,
    ε,
    Δz,
    w_grad_q_sq,
    w_grad_θ_sq,
    ∂T∂θ_li,
    wq_dot_wθ,
)
    FT = typeof(q′q′)
    twelfth = one(FT) / 12
    q_var =
        q′q′ +
        twelfth * Δz^2 * w_grad_q_sq
    T_var =
        T′T′ +
        twelfth * Δz^2 * ∂T∂θ_li^2 * w_grad_θ_sq
    num =
        ρ_param *
        sqrt(max(zero(FT), q′q′)) *
        sqrt(max(zero(FT), T′T′)) +
        twelfth * Δz^2 * ∂T∂θ_li * wq_dot_wθ
    den = max(
        oftype(q_var, ε),
        sqrt(max(zero(FT), q_var)) * sqrt(max(zero(FT), T_var)),
    )
    ρ_eff = clamp(num / den, -one(FT), one(FT))
    return q_var, T_var, ρ_eff
end

"""
    uniform_normal_convolution_pdf(x, a, b, σ)

PDF at `x` of the convolution of `Uniform(a,b)` with `Normal(0,σ)` (univariate).
For `a == b`, reduces to `Normal(a, σ)`.
"""
function uniform_normal_convolution_pdf(x, a, b, σ)
    FT = typeof(x)
    πf = FT(π)
    σp = max(σ, ϵ_variance_statistics(FT))
    s2 = σp * sqrt(FT(2))
    if a == b
        return (one(FT) / (σp * sqrt(FT(2) * πf))) *
               exp(-((x - a) / σp)^2 / 2)
    end
    return (one(FT) / (2 * (b - a))) * (erf((x - a) / s2) - erf((x - b) / s2))
end

"""
    bivariate_uniform_normal_isotropic_pdf(q, T, μ_q, μ_T, δq_half, δT_half, σ)

Bivariate PDF at `(q, T)` in **fully factorized form** ``f(q,T) = f_q(q) f_T(T)``: each
marginal is a univariate uniform–normal convolution on
`[μ_q − δq_half, μ_q + δq_half]` and `[μ_T − δT_half, μ_T + δT_half]` respectively, with
**the same** Gaussian noise scale `σ` on both axes. That is exactly **`Σ_turb = σ² I`**
(diagonal turbulent covariance — this kernel assigns **no** turbulent `q`–`T` cross-covariance).

That is **not** a statement that moisture and temperature are uncoupled in the real SGS
closure: operationally we still carry **`q′q′`**, **`T′T′`**, and **`corr_Tq`**
(`correlation_Tq`), with geometry in `subcell_geometric_covariance_Tq` and
`effective_sgs_quadrature_moments_matched_gaussian`, and quadrature uses
`sgs_stddevs_and_correlation` in `sgs_quadrature.jl`. This function is only the
**diagonal-`Σ_turb`, product-density** special case for tests and pencil-and-paper checks;
it does **not** accept `corr_Tq` or distinct `σ_q`, `σ_T`.

For the segment + full 2×2 turbulent covariance model (idealized PDF), see
`variance_statistics_reference.md` §4–6.
"""
function bivariate_uniform_normal_isotropic_pdf(
    q,
    T,
    μ_q,
    μ_T,
    δq_half,
    δT_half,
    σ,
)
    FT = typeof(q)
    σp = max(σ, ϵ_variance_statistics(FT))
    return uniform_normal_convolution_pdf(q, μ_q - δq_half, μ_q + δq_half, σp) *
           uniform_normal_convolution_pdf(T, μ_T - δT_half, μ_T + δT_half, σp)
end
