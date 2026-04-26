# Layer-mean profile quadrature for vertically resolved SGS distributions
# (column-tensor and profile–Rosenblatt). Two-slope half-center reconstruction
# from layer-mean and half-cell slopes passed in as scalars (see `microphysics_cache.jl`
# broadcasts). No persistent scratch fields.
#
# Profile–Rosenblatt (`SubgridProfileRosenblatt`): outer Hermite in `v` × inner
# Gauss–Legendre in `p` on the **two-component** uniform–Gaussian layer marginal.
# Inner marginal is **always** the composite split: ½ mass on each shifted
# half-cell `uniform⊛Gaussian` law, with per-leg inverse chosen by
# `ConvolutionQuantiles{Halley,Bracketed,ChebyshevLogEta}` (no scalar `F_mix^{-1}`).
# Each half uses **half-center-anchored** conditional std `σ_{u|v}`
# (constant on that half’s segment in `u`). There is **no** implemented closed form for a
# linear-in-`z` conditional variance inside the half’s inner marginal; variance
# slopes from the cache still enter the **half-center** reconstruction that sets
# `s_u_cond_dn` / `s_u_cond_up`. Outer `v` uses **per-half** marginal scales `s_v_fdn` / `s_v_fup`
# and shifts `r_fdn` / `r_fup` from the same half-center `Σ` as each inner `u` leg (same `χ[j]`,
# same Gauss–Hermite weights). The `uniform_gaussian_convolution_*` primitives
# below are the analytic core (`erf` / Mills integral algebra only).

import ClimaCore.Geometry as Geometry
import RootSolvers as RS
import SpecialFunctions as SF
import StaticArrays as SA

# ----------------------------------------------------------------------------
# Uniform–Gaussian convolution primitives: uniform[-L/2, L/2] ⊛ N(0, s²) on ℝ.
# (The density of `U + Z` with `U ~ Uniform(-L/2, L/2)` and `Z ~ N(0, s²)`
# independent; a rectangular "tophat" smoothed by a Gaussian of width s.)
# ----------------------------------------------------------------------------

@inline function _std_normal_cdf(y::FT) where {FT}
    return FT(0.5) * (one(FT) + SF.erf(y / sqrt(FT(2))))
end

@inline function _std_normal_pdf(y::FT) where {FT}
    return exp(-y^2 / FT(2)) / sqrt(FT(2) * FT(π))
end

"""Antiderivative `I(y) = y Φ(y) + φ(y)` for the standard normal."""
@inline function _normal_mills_integral(y::FT) where {FT}
    return y * _std_normal_cdf(y) + _std_normal_pdf(y)
end

"""CDF of `uniform[-L/2, L/2] ⊛ N(0, s²)` on `ℝ`."""
@inline function uniform_gaussian_convolution_cdf(x::FT, L::FT, s::FT) where {FT}
    ε = ϵ_numerics(FT)
    sp = max(s, ε)
    Lp = max(L, ε)
    u1 = (x + Lp / FT(2)) / sp
    u2 = (x - Lp / FT(2)) / sp
    return (sp / Lp) * (_normal_mills_integral(u1) - _normal_mills_integral(u2))
end

@inline function uniform_gaussian_convolution_pdf(x::FT, L::FT, s::FT) where {FT}
    ε = ϵ_numerics(FT)
    sp = max(s, ε)
    Lp = max(L, ε)
    s2 = sp * sqrt(FT(2))
    u1 = (x + Lp / FT(2)) / s2
    u2 = (x - Lp / FT(2)) / s2
    return (one(FT) / (FT(2) * Lp)) * (SF.erf(u1) - SF.erf(u2))
end

@inline function uniform_gaussian_convolution_pdf_prime(x::FT, L::FT, s::FT) where {FT}
    ε = ϵ_numerics(FT)
    sp = max(s, ε)
    Lp = max(L, ε)
    u1 = (x + Lp / FT(2)) / sp
    u2 = (x - Lp / FT(2)) / sp
    c = one(FT) / (Lp * sp * sqrt(FT(2) * FT(π)))
    return c * (exp(-u1^2 / FT(2)) - exp(-u2^2 / FT(2)))
end

"""Brent inverse of `uniform[-L/2,L/2] ⊛ N(0,s²)` at quantile level `p` (reference for Chebyshev tables)."""
function centered_uniform_gaussian_convolution_quantile_brent(
    p::FT, L::FT, s::FT,
) where {FT}
    ε = ϵ_numerics(FT)
    Lp = max(L, ε)
    sp = max(s, ε)
    smax = sp
    lo = -Lp / FT(2) - FT(6) * smax
    hi = Lp / FT(2) + FT(6) * smax
    f = x -> uniform_gaussian_convolution_cdf(x, Lp, sp) - p
    sol = RS.find_zero(
        f,
        RS.BrentsMethod{FT}(lo, hi),
        RS.CompactSolution(),
    )
    return sol.root
end

"""
    dn_half_uniform_gaussian_convolution_quantile_brent(p, L_dn, s_dn)

`p`-quantile of `uniform[-L_dn, 0] ⊛ N(0, s_dn²)` (lower half-cell shifted law;
same as the `F_{dn}` term inside [`mixture_uniform_gaussian_convolution_cdf`](@ref)).
"""
function dn_half_uniform_gaussian_convolution_quantile_brent(
    p::FT, L_dn::FT, s_dn::FT,
) where {FT}
    w = centered_uniform_gaussian_convolution_quantile_brent(p, L_dn, s_dn)
    return w - L_dn / FT(2)
end

"""
    up_half_uniform_gaussian_convolution_quantile_brent(p, L_up, s_up)

`p`-quantile of `uniform[0, L_up] ⊛ N(0, s_up²)` (upper half-cell shifted law).
"""
function up_half_uniform_gaussian_convolution_quantile_brent(
    p::FT, L_up::FT, s_up::FT,
) where {FT}
    w = centered_uniform_gaussian_convolution_quantile_brent(p, L_up, s_up)
    return w + L_up / FT(2)
end

"""
    centered_uniform_gaussian_convolution_quantile_halley(p, L, s)

One Halley step toward the `p`-quantile of **centered**
`uniform[-L/2,L/2] ⊛ N(0,s²)` (single-component law; used per leg in Rosenblatt).
"""
function centered_uniform_gaussian_convolution_quantile_halley(
    p::FT, L::FT, s::FT,
) where {FT}
    ε = ϵ_numerics(FT)
    Lp = max(L, ε)
    sp = max(s, ε)
    mix_ref = max(oftype(p, 1.0e-9), FT(256) * ε)
    t = FT(2) * p - one(FT)
    t = clamp(t, -one(FT) + FT(100) * mix_ref, one(FT) - FT(100) * mix_ref)
    z = sqrt(FT(2)) * SF.erfinv(t)
    σ_lin = sqrt(max(Lp^2 / FT(12) + sp^2, mix_ref^2))
    u = σ_lin * z
    g = uniform_gaussian_convolution_cdf(u, Lp, sp) - p
    fv = uniform_gaussian_convolution_pdf(u, Lp, sp)
    abs(fv) < mix_ref && return u
    fpv = uniform_gaussian_convolution_pdf_prime(u, Lp, sp)
    denom = FT(2) * fv^2 - g * fpv
    denom = copysign(
        max(abs(denom), FT(1.0e-15)),
        denom == zero(FT) ? one(FT) : denom,
    )
    return u - FT(2) * fv * g / denom
end

"""Lower half-cell `u` from one Halley step on the shifted single law."""
function dn_half_uniform_gaussian_convolution_quantile_halley(
    p::FT, L_dn::FT, s_dn::FT,
) where {FT}
    w = centered_uniform_gaussian_convolution_quantile_halley(p, L_dn, s_dn)
    return w - L_dn / FT(2)
end

"""Upper half-cell `u` from one Halley step on the shifted single law."""
function up_half_uniform_gaussian_convolution_quantile_halley(
    p::FT, L_up::FT, s_up::FT,
) where {FT}
    w = centered_uniform_gaussian_convolution_quantile_halley(p, L_up, s_up)
    return w + L_up / FT(2)
end

"""Lower half-cell `u` from Chebyshev row `(N_gl, i_node)` on the centered single law."""
function dn_half_uniform_gaussian_convolution_quantile_chebyshev(
    L_dn::FT, s_dn::FT, N_gl::Int, i_node::Int,
) where {FT}
    w = centered_uniform_gaussian_convolution_quantile_chebyshev(
        L_dn, s_dn, N_gl, i_node,
    )
    return w - L_dn / FT(2)
end

"""Upper half-cell `u` from Chebyshev row `(N_gl, i_node)` on the centered single law."""
function up_half_uniform_gaussian_convolution_quantile_chebyshev(
    L_up::FT, s_up::FT, N_gl::Int, i_node::Int,
) where {FT}
    w = centered_uniform_gaussian_convolution_quantile_chebyshev(
        L_up, s_up, N_gl, i_node,
    )
    return w + L_up / FT(2)
end

@inline function _chebyshev_tau_from_eta_single_conv(η::FT) where {FT}
    ε = ϵ_numerics(FT)
    lo = FT(CHEB_CONV_ETA_LOG10_RANGE[1])
    hi = FT(CHEB_CONV_ETA_LOG10_RANGE[2])
    ηmin = FT(10)^lo
    ηmax = FT(10)^hi
    ηp = clamp(max(η, ε), ηmin * (one(FT) + FT(100) * ε), ηmax)
    ℓ = log10(ηp)
    return clamp(FT(2) * (ℓ - lo) / (hi - lo) - one(FT), -one(FT), one(FT))
end

"""
    centered_uniform_gaussian_convolution_quantile_chebyshev(L, s, N_gl, i_node)

Chebyshev surrogate for the **`p`-quantile** of the **single** law
`uniform[-L/2,L/2] ⊛ N(0,s²)` at `η = s/L`, using the offline table row
`(N_gl, i_node)` fit for `p = i_node` Gauss–Legendre abscissa on order `N_gl`
(see `gen_convolution_chebyshev_tables.jl`).

Profile–Rosenblatt applies this **per leg** via
[`dn_half_uniform_gaussian_convolution_quantile_chebyshev`](@ref) /
[`up_half_uniform_gaussian_convolution_quantile_chebyshev`](@ref).
Callers must pair `i_node` with the same `p` used in Brent (the `i_node`th
node from [`gauss_legendre_01`](@ref)).
"""
function centered_uniform_gaussian_convolution_quantile_chebyshev(
    L::FT, s::FT, N_gl::Int, i_node::Int,
) where {FT}
    ε = ϵ_numerics(FT)
    Lp = max(L, ε)
    sp = max(s, ε)
    η = sp / Lp
    τ = _chebyshev_tau_from_eta_single_conv(η)
    coeffs = chebyshev_convolution_coeffs(GaussianSGS(), FT, N_gl, i_node)
    return chebyshev_evaluate(coeffs, τ) * Lp
end

# ----------------------------------------------------------------------------
# Two-component uniform–Gaussian convolution mixture (inner `u` for profile–Rosenblatt).
#
# The u-marginal along the mean-gradient direction is a length-weighted
# (½, ½) mixture of two uniform–Gaussian convolutions, one per half-cell:
#     lower half-cell component: uniform[-L_-, 0] ⊛ N(0, s_-²)   (mean -L_-/2)
#     upper half-cell component: uniform[ 0, L_+] ⊛ N(0, s_+²)   (mean +L_+/2)
# Each half-cell σ is a single scalar (face-anchored evaluation of the
# conditional std), so σ² is held constant within each half-cell for the inner
# `u` marginal. Deeper `z`–`(T,q)` coupling is handled by `SubgridColumnTensor`
# and the other `AbstractSubgridLayerProfileQuadrature` layouts (LHS–`z`, Voronoi,
# barycentric, principal axis), not by a second analytic branch here.
# ----------------------------------------------------------------------------

@inline function _uniform_gaussian_convolution_cdf_shifted(
    x::FT, a::FT, b::FT, s::FT,
) where {FT}
    mid = (a + b) / FT(2)
    return uniform_gaussian_convolution_cdf(x - mid, b - a, s)
end

@inline function _uniform_gaussian_convolution_pdf_shifted(
    x::FT, a::FT, b::FT, s::FT,
) where {FT}
    mid = (a + b) / FT(2)
    return uniform_gaussian_convolution_pdf(x - mid, b - a, s)
end

@inline function _uniform_gaussian_convolution_pdf_prime_shifted(
    x::FT, a::FT, b::FT, s::FT,
) where {FT}
    mid = (a + b) / FT(2)
    return uniform_gaussian_convolution_pdf_prime(x - mid, b - a, s)
end

"""
    mixture_uniform_gaussian_convolution_cdf(x, L_dn, s_dn, L_up, s_up)

CDF of the length-weighted two-component mixture
`½·(uniform[-L_dn, 0] ⊛ N(0, s_dn²)) + ½·(uniform[0, L_up] ⊛ N(0, s_up²))`.
This is the exact u-marginal of the two-slope face-anchored layer joint.
"""
@inline function mixture_uniform_gaussian_convolution_cdf(
    x::FT, L_dn::FT, s_dn::FT, L_up::FT, s_up::FT,
) where {FT}
    F_dn =
        _uniform_gaussian_convolution_cdf_shifted(x, -L_dn, zero(FT), s_dn)
    F_up =
        _uniform_gaussian_convolution_cdf_shifted(x, zero(FT), L_up, s_up)
    return (F_dn + F_up) / FT(2)
end

"""
    mixture_uniform_gaussian_convolution_pdf(x, L_dn, s_dn, L_up, s_up)

PDF of the two-component uniform–Gaussian mixture; see
[`mixture_uniform_gaussian_convolution_cdf`](@ref).
"""
@inline function mixture_uniform_gaussian_convolution_pdf(
    x::FT, L_dn::FT, s_dn::FT, L_up::FT, s_up::FT,
) where {FT}
    f_dn =
        _uniform_gaussian_convolution_pdf_shifted(x, -L_dn, zero(FT), s_dn)
    f_up =
        _uniform_gaussian_convolution_pdf_shifted(x, zero(FT), L_up, s_up)
    return (f_dn + f_up) / FT(2)
end

@inline function mixture_uniform_gaussian_convolution_pdf_prime(
    x::FT, L_dn::FT, s_dn::FT, L_up::FT, s_up::FT,
) where {FT}
    fp_dn =
        _uniform_gaussian_convolution_pdf_prime_shifted(x, -L_dn, zero(FT), s_dn)
    fp_up =
        _uniform_gaussian_convolution_pdf_prime_shifted(x, zero(FT), L_up, s_up)
    return (fp_dn + fp_up) / FT(2)
end

"""
    mixture_uniform_gaussian_convolution_mean_var(L_dn, s_dn, L_up, s_up)

Closed-form mean and variance of the two-component uniform–Gaussian mixture.
Variance decomposition:
`var = (5L_+² + 6 L_− L_+ + 5L_−²)/48 + (s_−² + s_+²)/2`
(the first term is the length-weighted mixture of the two shifted uniforms;
the second is the mean of the two local Gaussian variances).
"""
@inline function mixture_uniform_gaussian_convolution_mean_var(
    L_dn::FT, s_dn::FT, L_up::FT, s_up::FT,
) where {FT}
    μ = (L_up - L_dn) / FT(4)
    var =
        (FT(5) * L_up^2 + FT(6) * L_up * L_dn + FT(5) * L_dn^2) / FT(48) +
        (s_dn^2 + s_up^2) / FT(2)
    return μ, var
end

"""
    mixture_uniform_gaussian_convolution_quantile_brent(p, L_dn, s_dn, L_up, s_up)

Bracketed Brent inverse of the analytic mixture CDF
[`mixture_uniform_gaussian_convolution_cdf`](@ref): returns `u` such that
`F_mix(u) = p`.

This helper is provided for robustness checks and analysis. The production
profile integrator uses composite per-leg inverses, not a single `F_mix^{-1}`.
"""
function mixture_uniform_gaussian_convolution_quantile_brent(
    p::FT, L_dn::FT, s_dn::FT, L_up::FT, s_up::FT,
) where {FT}
    ε = ϵ_numerics(FT)
    s_max = max(s_dn, s_up, ε)
    lo = -L_dn - FT(6) * s_max
    hi = L_up + FT(6) * s_max
    f = x -> mixture_uniform_gaussian_convolution_cdf(x, L_dn, s_dn, L_up, s_up) - p
    sol = RS.find_zero(
        f,
        RS.BrentsMethod{FT}(lo, hi),
        RS.CompactSolution(),
    )
    return sol.root
end

"""
    _rosenblatt_half_w_dn(method, p, L_dn, s_dn, N_gl, i_node)
    _rosenblatt_half_w_up(method, p, L_up, s_up, N_gl, i_node)

Dispatch layer-profile half-leg quantile solves in transformed `w`-space
(`w = q` for Gaussian, `w = ln(q)` for LogNormal wrappers). These helpers are
pure uniform-normal half-convolution solvers and do not perform any outer
mapping to physical `(T, q)` coordinates.
"""
@inline function _rosenblatt_half_w_dn(
    ::ConvolutionQuantilesBracketed,
    p::FT,
    L_dn::FT,
    s_dn::FT,
    ::Int,
    ::Int,
) where {FT}
    return dn_half_uniform_gaussian_convolution_quantile_brent(p, L_dn, s_dn)
end

@inline function _rosenblatt_half_w_dn(
    ::ConvolutionQuantilesHalley,
    p::FT,
    L_dn::FT,
    s_dn::FT,
    ::Int,
    ::Int,
) where {FT}
    return dn_half_uniform_gaussian_convolution_quantile_halley(p, L_dn, s_dn)
end

@inline function _rosenblatt_half_w_dn(
    ::ConvolutionQuantilesChebyshevLogEta,
    ::FT,
    L_dn::FT,
    s_dn::FT,
    N_gl::Int,
    i_node::Int,
) where {FT}
    return dn_half_uniform_gaussian_convolution_quantile_chebyshev(
        L_dn, s_dn, N_gl, i_node,
    )
end

@inline function _rosenblatt_half_w_up(
    ::ConvolutionQuantilesBracketed,
    p::FT,
    L_up::FT,
    s_up::FT,
    ::Int,
    ::Int,
) where {FT}
    return up_half_uniform_gaussian_convolution_quantile_brent(p, L_up, s_up)
end

@inline function _rosenblatt_half_w_up(
    ::ConvolutionQuantilesHalley,
    p::FT,
    L_up::FT,
    s_up::FT,
    ::Int,
    ::Int,
) where {FT}
    return up_half_uniform_gaussian_convolution_quantile_halley(p, L_up, s_up)
end

@inline function _rosenblatt_half_w_up(
    ::ConvolutionQuantilesChebyshevLogEta,
    ::FT,
    L_up::FT,
    s_up::FT,
    N_gl::Int,
    i_node::Int,
) where {FT}
    return up_half_uniform_gaussian_convolution_quantile_chebyshev(
        L_up, s_up, N_gl, i_node,
    )
end

"""
    _rosenblatt_lognormal_half_q_dn(method, p, μ_q, L_dn, σ_ln, q_max, N_gl, i_node)
    _rosenblatt_lognormal_half_q_up(method, p, μ_q, L_up, σ_ln, q_max, N_gl, i_node)

LogNormal half-leg quantile helpers for `SubgridProfileRosenblatt`.

These functions operate in transformed space (`ln(q)`) and then map back to
physical `q` with `exp(·)`. They are separate from the Gaussian `u`-helpers
(`_rosenblatt_{dn,up}_half_u`) because LogNormal quantiles include moisture
state parameters (`μ_q`, `σ_ln`) and positive-domain clamping via `q_max`.

Method variants:
- `ConvolutionQuantilesBracketed`: Brent solve in transformed uniform-normal law
- `ConvolutionQuantilesHalley`: one-step Halley in transformed law
- `ConvolutionQuantilesChebyshevLogEta`: transformed-space Chebyshev surrogate
"""
@inline function _rosenblatt_lognormal_half_q_dn(
    method::ConvolutionQuantilesChebyshevLogEta,
    p,
    μ_q,
    L_dn,
    σ_ln,
    q_max,
    N_gl,
    i_node,
)
    FT = promote_type(typeof(p), typeof(μ_q), typeof(L_dn), typeof(σ_ln), typeof(q_max))
    return _rosenblatt_lognormal_half_q_dn_impl(
        method, FT(p), FT(μ_q), FT(L_dn), FT(σ_ln), FT(q_max), N_gl, i_node
    )
end

@inline function _rosenblatt_lognormal_half_q_dn(
    method,
    p,
    μ_q,
    L_dn,
    σ_ln,
    q_max,
    N_gl,
    i_node,
)
    FT = promote_type(typeof(p), typeof(μ_q), typeof(L_dn), typeof(σ_ln), typeof(q_max))
    return _rosenblatt_lognormal_half_q_dn_impl(
        method, FT(p), FT(μ_q), FT(L_dn), FT(σ_ln), FT(q_max)
    )
end

@inline function _rosenblatt_lognormal_half_q_up(
    method::ConvolutionQuantilesChebyshevLogEta,
    p,
    μ_q,
    L_up,
    σ_ln,
    q_max,
    N_gl,
    i_node,
)
    FT = promote_type(typeof(p), typeof(μ_q), typeof(L_up), typeof(σ_ln), typeof(q_max))
    return _rosenblatt_lognormal_half_q_up_impl(
        method, FT(p), FT(μ_q), FT(L_up), FT(σ_ln), FT(q_max), N_gl, i_node
    )
end

@inline function _rosenblatt_lognormal_half_q_up(
    method,
    p,
    μ_q,
    L_up,
    σ_ln,
    q_max,
    N_gl,
    i_node,
)
    FT = promote_type(typeof(p), typeof(μ_q), typeof(L_up), typeof(σ_ln), typeof(q_max))
    return _rosenblatt_lognormal_half_q_up_impl(
        method, FT(p), FT(μ_q), FT(L_up), FT(σ_ln), FT(q_max)
    )
end

@inline function _rosenblatt_lognormal_half_q_dn_impl(
    ::ConvolutionQuantilesBracketed,
    p::FT,
    μ_q::FT,
    L_dn::FT,
    σ_ln::FT,
    q_max::FT,
) where {FT}
    ε = ϵ_numerics(FT)
    μ_ln = log(max(μ_q, ε))
    w = _rosenblatt_half_w_dn(
        ConvolutionQuantilesBracketed(), p, L_dn, σ_ln, 0, 0
    )
    return clamp(exp(μ_ln + w), ε, q_max)
end

@inline function _rosenblatt_lognormal_half_q_dn_impl(
    ::ConvolutionQuantilesHalley,
    p::FT,
    μ_q::FT,
    L_dn::FT,
    σ_ln::FT,
    q_max::FT,
) where {FT}
    ε = ϵ_numerics(FT)
    μ_ln = log(max(μ_q, ε))
    w = _rosenblatt_half_w_dn(ConvolutionQuantilesHalley(), p, L_dn, σ_ln, 0, 0)
    return clamp(exp(μ_ln + w), ε, q_max)
end

@inline function _rosenblatt_lognormal_half_q_dn_impl(
    ::ConvolutionQuantilesChebyshevLogEta,
    p::FT,
    μ_q::FT,
    L_dn::FT,
    σ_ln::FT,
    q_max::FT,
    N_gl::Int,
    i_node::Int,
) where {FT}
    # Evaluate the 1D uniform-normal Chebyshev surrogate in ln(q)-space
    # on the lower half interval [ln(μ_q)-L_dn, ln(μ_q)].
    ε = ϵ_numerics(FT)
    μ_ln = log(max(μ_q, ε))
    w = _rosenblatt_half_w_dn(
        ConvolutionQuantilesChebyshevLogEta(), p, L_dn, σ_ln, N_gl, i_node,
    )
    return clamp(exp(μ_ln + w), ε, q_max)
end

@inline function _rosenblatt_lognormal_half_q_up_impl(
    ::ConvolutionQuantilesChebyshevLogEta,
    p::FT,
    μ_q::FT,
    L_up::FT,
    σ_ln::FT,
    q_max::FT,
    N_gl::Int,
    i_node::Int,
) where {FT}
    # Evaluate the 1D uniform-normal Chebyshev surrogate in ln(q)-space
    # on the upper half interval [ln(μ_q), ln(μ_q)+L_up].
    ε = ϵ_numerics(FT)
    μ_ln = log(max(μ_q, ε))
    w = _rosenblatt_half_w_up(
        ConvolutionQuantilesChebyshevLogEta(), p, L_up, σ_ln, N_gl, i_node,
    )
    return clamp(exp(μ_ln + w), ε, q_max)
end

@inline function _rosenblatt_lognormal_half_q_up_impl(
    ::ConvolutionQuantilesBracketed,
    p::FT,
    μ_q::FT,
    L_up::FT,
    σ_ln::FT,
    q_max::FT,
) where {FT}
    ε = ϵ_numerics(FT)
    μ_ln = log(max(μ_q, ε))
    w = _rosenblatt_half_w_up(ConvolutionQuantilesBracketed(), p, L_up, σ_ln, 0, 0)
    return clamp(exp(μ_ln + w), ε, q_max)
end

@inline function _rosenblatt_lognormal_half_q_up_impl(
    ::ConvolutionQuantilesHalley,
    p::FT,
    μ_q::FT,
    L_up::FT,
    σ_ln::FT,
    q_max::FT,
) where {FT}
    ε = ϵ_numerics(FT)
    μ_ln = log(max(μ_q, ε))
    w = _rosenblatt_half_w_up(ConvolutionQuantilesHalley(), p, L_up, σ_ln, 0, 0)
    return clamp(exp(μ_ln + w), ε, q_max)
end



@inline _profile_rosenblatt_method(
    ::VerticallyResolvedSGS{<:SubgridProfileRosenblatt{B}},
) where {B <: AbstractConvolutionQuantileMethod} = B()

function _integrate_over_sgs_profile_rosenblatt(
    f,
    quad::SGSQuadrature,
    dist::VerticallyResolvedSGS{<:SubgridProfileRosenblatt, I},
    innerD::I,
    μ_q,
    μ_T,
    q′q′,
    T′T′,
    ρ_c,
    H,
    dq_dz_dn,
    dq_dz_up,
    dT_dz_dn,
    dT_dz_up,
    dqq_dz_dn,
    dqq_dz_up,
    dTT_dz_dn,
    dTT_dz_up,
    seed,
) where {I <: Union{GaussianSGS, LogNormalSGS}}
    # Core layer-profile Rosenblatt integration:
    # 1) build two half-cell parameter packs (`:dn`, `:up`) from mean and variance
    #    slopes via `_two_slope_rosenblatt_params`;
    # 2) outer Gauss-Hermite loop samples `v`;
    # 3) inner Gauss-Legendre loop inverts half-leg marginals (u for Gaussian,
    #    transformed ln(q) for LogNormal) using Bracketed/Halley/Chebyshev;
    # 4) map samples back to `(T, q)` and accumulate weighted tendencies.
    #
    # For LogNormal Profile–Rosenblatt, this path treats the q-axis in ln-space
    # consistently: the incoming `dq_dz_*` are already converted to d(ln q)/dz at
    # the `integrate_over_sgs` entry point, `_two_slope_rosenblatt_params` builds
    # the rotated map with that axis, and `_profile_rosenblatt_emit_inner_sample`
    # converts `(δT, δln q)` to physical `q` only at the final local map.
    FT = typeof(H)
    μ_T_p = convert(FT, μ_T)
    μ_q_p = convert(FT, μ_q)
    ε = ϵ_numerics(FT)
    σ_q_c, σ_T_c, _ = sgs_stddevs_and_correlation(
        convert(FT, q′q′),
        convert(FT, T′T′),
        convert(FT, ρ_c),
    )
    method = _profile_rosenblatt_method(dist)
    function _profile_rosenblatt_accumulate(params)
        M_inv = params.M_inv
        s_v_fdn = params.s_v_fdn
        s_v_fup = params.s_v_fup
        L_dn = params.L_dn
        L_up = params.L_up
        s_u_cond_c = params.s_u_cond_c
        s_u_cond_dn = params.s_u_cond_dn
        s_u_cond_up = params.s_u_cond_up
        r_c = params.r_c
        r_fdn = params.r_fdn
        r_fup = params.r_fup
        σ_T_fdn = params.σ_T_fdn
        σ_q_fdn = params.σ_q_fdn
        σ_T_fup = params.σ_T_fup
        σ_q_fup = params.σ_q_fup
        ε = ϵ_numerics(FT)
        N = quadrature_order(quad)
        p_nodes, p_w = gauss_legendre_01(FT, N)
        χ = quad.a
        wgh = quad.w
        inv_sqrt_pi = one(FT) / sqrt(FT(π))
        acc = rzero(seed)
        sqrt2 = sqrt(FT(2))
        @inbounds for j in 1:N
            χj = χ[j]
            wvj = wgh[j] * inv_sqrt_pi
            vj_fdn = sqrt2 * s_v_fdn * χj
            vj_fup = sqrt2 * s_v_fup * χj
            μ_0_fdn = r_fdn * vj_fdn
            μ_0_fup = r_fup * vj_fup
            # Fully degenerate inner marginal: fall back to one outer draw using the larger
            # half-center `v` scale so `v_j` remains resolved when one half has zero spread.
            s_v_deg = max(s_v_fdn, s_v_fup, ε)
            vj_deg = sqrt2 * s_v_deg * χj
            μ_0_deg = r_c * vj_deg
            degenerate =
                (L_dn + L_up) <= ε &&
                max(s_u_cond_c, s_u_cond_dn, s_u_cond_up) <= ε
            if degenerate
                @inbounds for i in 1:N
                    wi = p_w[i]
                    ui = μ_0_deg
                    acc = _profile_rosenblatt_emit_inner_sample(
                        acc, f, innerD, μ_q_p, μ_T_p, σ_q_c, σ_T_c, ρ_c,
                        quad, M_inv, ui, μ_0_deg, vj_deg, wvj, wi,
                        method,
                    )
                end
            else
                @inbounds for i in 1:N
                    pi = p_nodes[i]
                    wi = p_w[i]
                    # When `L_up` or `L_dn` vanishes on this mean-gradient axis, that half's
                    # `uniform ⊛ Gaussian` leg has **zero width** in `u`. The CDF helpers clamp
                    # `L → max(L, ε)`, which invents a spurious inner law and (with huge
                    # `s_u_cond_*` from the other half's rotation) can make Brent vs Halley disagree
                    # badly. Skip zero-width legs; use full `wi` on each surviving leg so inner
                    # mass still sums like Gauss–Legendre on `[0,1]`.
                    use_dn = L_dn > ε
                    use_up = L_up > ε
                    if use_dn && use_up
                        wi_half = wi * FT(0.5)
                        ui =
                            _rosenblatt_half_w_dn(
                                method, pi, L_dn, s_u_cond_dn, N, i,
                            ) + μ_0_fdn
                        acc = _profile_rosenblatt_emit_inner_sample(
                            acc, f, innerD, μ_q_p, μ_T_p, σ_q_fdn, σ_T_fdn, ρ_c,
                            quad, M_inv, ui, μ_0_fdn, vj_fdn, wvj, wi_half,
                            method,
                        )
                        ui_up =
                            _rosenblatt_half_w_up(
                                method, pi, L_up, s_u_cond_up, N, i,
                            ) + μ_0_fup
                        acc = _profile_rosenblatt_emit_inner_sample(
                            acc, f, innerD, μ_q_p, μ_T_p, σ_q_fup, σ_T_fup, ρ_c,
                            quad, M_inv, ui_up, μ_0_fup, vj_fup, wvj, wi_half,
                            method,
                        )
                    elseif use_dn
                        ui =
                            _rosenblatt_half_w_dn(
                                method, pi, L_dn, s_u_cond_dn, N, i,
                            ) + μ_0_fdn
                        acc = _profile_rosenblatt_emit_inner_sample(
                            acc, f, innerD, μ_q_p, μ_T_p, σ_q_fdn, σ_T_fdn, ρ_c,
                            quad, M_inv, ui, μ_0_fdn, vj_fdn, wvj, wi,
                            method,
                        )
                    elseif use_up
                        ui_up =
                            _rosenblatt_half_w_up(
                                method, pi, L_up, s_u_cond_up, N, i,
                            ) + μ_0_fup
                        acc = _profile_rosenblatt_emit_inner_sample(
                            acc, f, innerD, μ_q_p, μ_T_p, σ_q_fup, σ_T_fup, ρ_c,
                            quad, M_inv, ui_up, μ_0_fup, vj_fup, wvj, wi,
                            method,
                        )
                    else
                        # Both half-widths vanish: inner `p` nodes collapse to `μ_0` (same as the
                        # `degenerate` branch above, but reachable when `L_dn + L_up > ε` in raw
                        # storage while each leg is still below `ε` after projection).
                        ui = μ_0_deg
                        acc = _profile_rosenblatt_emit_inner_sample(
                            acc, f, innerD, μ_q_p, μ_T_p, σ_q_c, σ_T_c, ρ_c,
                            quad, M_inv, ui, μ_0_deg, vj_deg, wvj, wi,
                            method,
                        )
                    end
                end
            end
        end
        return acc
    end
    σT²p = oftype(μ_T_p, T′T′)
    σq²p = oftype(μ_T_p, q′q′)
    ρp = oftype(μ_T_p, ρ_c)
    Hp = oftype(μ_T_p, H)
    has_dn = _two_slope_rosenblatt_has_valid_axis(
        μ_T_p, μ_q_p, σT²p, σq²p, ρp,
        dT_dz_dn, dT_dz_up, dq_dz_dn, dq_dz_up,
        dTT_dz_dn, dTT_dz_up, dqq_dz_dn, dqq_dz_up, Hp;
        mean_gradient_axis = Val(:dn),
    )
    has_up = _two_slope_rosenblatt_has_valid_axis(
        μ_T_p, μ_q_p, σT²p, σq²p, ρp,
        dT_dz_dn, dT_dz_up, dq_dz_dn, dq_dz_up,
        dTT_dz_dn, dTT_dz_up, dqq_dz_dn, dqq_dz_up, Hp;
        mean_gradient_axis = Val(:up),
    )
    if !has_dn && !has_up
        # ‖(d_T, d_q)‖ = 0 on both half-gradient packs: the Rosenblatt half-cell
        # `uniform ⊛ Gaussian` construction has no axis. Fall back to the **same**
        # cell-center bivariate Gauss–Hermite rule used by the 2D SGS overload of
        # `integrate_over_sgs` for [`VerticallyResolvedSGS`](@ref) (see
        # `sgs_quadrature.jl`), not a single grid-mean sample `f(μ_T, μ_q)` (which
        # would kill SGS variance for every `f`).
        σ_q_0, σ_T_0, ρ_0 = sgs_stddevs_and_correlation(
            convert(FT, q′q′),
            convert(FT, T′T′),
            convert(FT, ρ_c),
        )
        χ = quad.a
        weights = quad.w
        inv_sqrt_pi = one(FT) / sqrt(FT(π))
        acc0 = rzero(seed)
        @inbounds for i in eachindex(χ)
            inner_sum = rzero(acc0)
            @inbounds for j in eachindex(χ)
                T_hat, q_hat = get_physical_point(
                    innerD,
                    χ[i],
                    χ[j],
                    μ_q_p,
                    μ_T_p,
                    oftype(μ_T_p, σ_q_0),
                    oftype(μ_T_p, σ_T_0),
                    oftype(μ_T_p, ρ_0),
                    oftype(μ_T_p, quad.T_min),
                    oftype(μ_T_p, quad.q_max),
                )
                inner_sum = inner_sum ⊞ (f(T_hat, q_hat) ⊠ (weights[j] * inv_sqrt_pi))
            end
            acc0 = acc0 ⊞ (inner_sum ⊠ (weights[i] * inv_sqrt_pi))
        end
        return acc0
    elseif has_dn && has_up
        p_dn = _two_slope_rosenblatt_params(
            μ_T_p, μ_q_p, σT²p, σq²p, ρp,
            dT_dz_dn, dT_dz_up, dq_dz_dn, dq_dz_up,
            dTT_dz_dn, dTT_dz_up, dqq_dz_dn, dqq_dz_up, Hp;
            mean_gradient_axis = Val(:dn),
        )
        p_up = _two_slope_rosenblatt_params(
            μ_T_p, μ_q_p, σT²p, σq²p, ρp,
            dT_dz_dn, dT_dz_up, dq_dz_dn, dq_dz_up,
            dTT_dz_dn, dTT_dz_up, dqq_dz_dn, dqq_dz_up, Hp;
            mean_gradient_axis = Val(:up),
        )
        a_dn = _profile_rosenblatt_accumulate(p_dn)
        a_up = _profile_rosenblatt_accumulate(p_up)
        return a_dn ⊠ FT(0.5) ⊞ a_up ⊠ FT(0.5)
    elseif has_dn
        p_dn = _two_slope_rosenblatt_params(
            μ_T_p, μ_q_p, σT²p, σq²p, ρp,
            dT_dz_dn, dT_dz_up, dq_dz_dn, dq_dz_up,
            dTT_dz_dn, dTT_dz_up, dqq_dz_dn, dqq_dz_up, Hp;
            mean_gradient_axis = Val(:dn),
        )
        return _profile_rosenblatt_accumulate(p_dn)
    else
        p_up = _two_slope_rosenblatt_params(
            μ_T_p, μ_q_p, σT²p, σq²p, ρp,
            dT_dz_dn, dT_dz_up, dq_dz_dn, dq_dz_up,
            dTT_dz_dn, dTT_dz_up, dqq_dz_dn, dqq_dz_up, Hp;
            mean_gradient_axis = Val(:up),
        )
        return _profile_rosenblatt_accumulate(p_up)
    end
end

# ----------------------------------------------------------------------------
# Inner-distribution mapping and physical-point recovery.
# ----------------------------------------------------------------------------

# _inner_dist is now defined in sgs_quadrature.jl and dispatches on the I type parameter.

"""Recover Hermite nodes `(χ1, χ2)` yielding fluctuations `(δT, δq)` with correlation `ρ`."""
@inline function _hermite_from_gaussian_fluctuations(δq, δT, σ_q, σ_T, ρ)
    FT = typeof(δT)
    sqrt2 = sqrt(FT(2))
    εf = ϵ_numerics(FT)
    χ1 = δq / (sqrt2 * max(σ_q, εf))
    σ_c = sqrt(max(one(FT) - ρ^2, zero(FT))) * max(σ_T, εf)
    χ2 = (δT - sqrt2 * ρ * max(σ_T, εf) * χ1) / (sqrt2 * σ_c)
    return χ1, χ2
end

@inline function _physical_Tq_from_fluctuations(
    innerD, μ_q, μ_T, δq, δT, σ_q, σ_T, ρ, T_min, q_max,
)
    χ1, χ2 = _hermite_from_gaussian_fluctuations(δq, δT, σ_q, σ_T, ρ)
    return get_physical_point(
        innerD, χ1, χ2, μ_q, μ_T, σ_q, σ_T, ρ, T_min, q_max,
    )
end

@inline function _physical_Tq_from_lnq_fluctuations(
    μ_q::FT,
    μ_T::FT,
    δlnq::FT,
    δT::FT,
    T_min::FT,
    q_max::FT,
) where {FT}
    ε = ϵ_numerics(FT)
    q_hat = clamp(exp(log(max(μ_q, ε)) + δlnq), ε, q_max)
    T_hat = max(T_min, μ_T + δT)
    return (T_hat, q_hat)
end

# ----------------------------------------------------------------------------
# Two-slope helpers (piecewise-linear reconstruction; no persistent fields).
# ----------------------------------------------------------------------------

@inline function _pick_half_slope(ζ::FT, s_dn::FT, s_up::FT) where {FT}
    return ζ >= zero(FT) ? s_up : s_dn
end

"""
    _vertical_layer_mean_q(innerD, μ_q, ζ, dq_dz_dn_eff, dq_dz_up_eff, FT)

Piecewise-linear **layer-center** mean of `q` at vertical offset `ζ` from the
layer midpoint (same half-slope convention as [`_pick_half_slope`](@ref)).

For [`LogNormalSGS`](@ref), `dq_dz_*_eff` are **ln(q)** slopes along `z`
(see the `dq_dz_*_eff` projection in [`integrate_over_sgs`](@ref) for vertically
resolved distributions); the conditional mean follows
``μ_q(ζ) = \\exp(\\ln μ_q + ζ\\, s_{\\ln q})``. For [`GaussianSGS`](@ref),
`dq_dz_*_eff` are physical `q` slopes and ``μ_q(ζ) = μ_q + ζ\\, s_q``.
"""
@inline function _vertical_layer_mean_q(
    innerD,
    μ_q::FT,
    ζ::FT,
    dq_dz_dn_eff::FT,
    dq_dz_up_eff::FT,
    ::Type{FT},
) where {FT}
    sμ_q = _pick_half_slope(ζ, dq_dz_dn_eff, dq_dz_up_eff)
    if innerD isa LogNormalSGS
        ε = ϵ_numerics(FT)
        ln_μ = log(max(μ_q, ε))
        return exp(ln_μ + ζ * sμ_q)
    end
    return μ_q + ζ * sμ_q
end

"""
    _largest_eigval_unit_evec_Tq(σT², ρ_σT_σq, σq²) -> (λ₁, u₁, u₂)

Largest eigenvalue and **unit** eigenvector of the 2×2 symmetric covariance
``[[σT², ρ σT σq], [ρ σT σq, σq²]]`` in physical ``(T, q)`` space (principal axis
for [`SubgridPrincipalAxisLayer`](@ref)).
"""
@inline function _largest_eigval_unit_evec_Tq(
    σT²::FT, ρ_σT_σq::FT, σq²::FT,
) where {FT}
    ε = ϵ_numerics(FT)
    a, b, c = σT², ρ_σT_σq, σq²
    tr = a + c
    det2 = a * c - b * b
    disc = max(tr * tr - 4 * det2, zero(FT))
    λ1 = (tr + sqrt(disc)) / FT(2)
    u1, u2 = b, λ1 - a
    nrm = hypot(u1, u2)
    if nrm < ε
        u1, u2, nrm = one(FT), zero(FT), one(FT)
    else
        u1 = u1 / nrm
        u2 = u2 / nrm
    end
    return (max(λ1, ε * ε), u1, u2)
end

"""0-based linear index `t0` in ``[0, N^3)`` to 1-based `(k, i, j)` (lexicographic in ``k, i, j``)."""
@inline function _lex_index0_to_ktij(t0::Int, N::Int)
    t0c = clamp(t0, 0, N^3 - 1)
    k = t0c ÷ N^2 + 1
    r = t0c - (k - 1) * N^2
    i = r ÷ N + 1
    j = r % N + 1
    return k, i, j
end

@inline function _normalized_kij_coord(k::Int, i::Int, j::Int, ::Type{FT}, N::Int) where {FT}
    den = FT(max(N - 1, 1))
    return SA.SVector(
        FT(k - 1) / den,
        FT(i - 1) / den,
        FT(j - 1) / den,
    )
end

"""Rotated Σ_turb component at a point given untransformed `(σT², σq², sTq)` and axes `(u_row_T, u_row_q, d_T, d_q)`."""
@inline function _rotated_sigma_uv(
    σT²::FT, σq²::FT, sTq::FT,
    u_row_T::FT, u_row_q::FT, d_T::FT, d_q::FT,
) where {FT}
    s1sq =
        u_row_T^2 * σT² + FT(2) * u_row_T * u_row_q * sTq + u_row_q^2 * σq²
    s12 =
        -d_q * u_row_T * σT² +
        (d_T * u_row_T - d_q * u_row_q) * sTq +
        d_T * u_row_q * σq²
    s2sq = d_q^2 * σT² - FT(2) * d_q * d_T * sTq + d_T^2 * σq²
    return s1sq, s12, s2sq
end

"""
    _two_slope_rosenblatt_params(...; mean_gradient_axis = Val(:avg))

Assemble **profile–Rosenblatt** inner-marginal parameters: half-widths `L_±` on
`u`, conditional standard deviations `σ_{u|v}` at the **cell center and each
half-center** (piecewise-linear turbulent covariances in `(T,q)` then
[`_rotated_sigma_uv`](@ref) into the `(u,v)` frame built from the chosen **mean
slopes**), and ratios `r = s12/s2²` at center and half-centers (`r_c`, `r_fdn`, `r_fup`).
`SubgridProfileRosenblatt` integration applies **half-local** shifts on `u` for each
inner leg: `μ0_dn = r_fdn * vj_dn` and `μ0_up = r_fup * vj_up` with
`vj_leg = sqrt(2) * s_v_leg * χ_j`, using half-center `r_fdn` / `r_fup` from the same
`Σ` slice as that leg’s `s_u_cond_*`. The **fully degenerate** inner marginal (all `L` and
conditional inner spreads below `ε`) still uses `μ0 = r_c * v` with `r_c` from **cell-center**
`Σ` and a resolved outer scale `max(s_v_fdn, s_v_fup, ε)` so the outer Hermite axis does
not vanish.

The inner `uniform ⊛ N` inversion uses **half-center-anchored** `σ_{u|v}` on each half:
`s_u_cond_dn` / `s_u_cond_up` come from the rotated turbulent covariance at the **lower** /
**upper** half-center after the piecewise-linear `Σ_{turb}(z)` reconstruction, not from
linearly interpolating `σ²_{u|v}` in `z` along the half. There is **no** linear-in-`z`
conditional variance inside each half’s closed-form inner marginal.

Center/half-center/face convention:
- means and mean-gradients follow the two half-cell mean reconstructions;
- turbulent covariance used for `s_u_cond_dn/up` is sampled at half-centers (`±H/4`);
- full faces (`±H/2`) are not used for `σ_{u|v}` in this implementation.

Abscissas `quadrature_order(quad) → (ξ_01, w_01)` drive the **composite** split
(DN half-law + UP half-law at the same `p` nodes, each weighted `1/2` in the
two-half integration when combining the separate `:dn` and `:up` parameter packs).

**Keyword** `mean_gradient_axis` selects which **vertical mean-gradient** (slopes
of the piecewise-linear cell-mean fields in each half) defines
the inner `u` direction and `M_inv` for *this* parameter pack:

  - `:avg` — mean of the below- and above-center half slopes.
  - `:dn` / `:up` — the below-center or above-center **half** only. The
    production `SubgridProfileRosenblatt` path builds **two** such packs
    (`:dn` and `:up`) and **averages** their cubature contributions
    (factor `1/2` each) so both axes are represented when DN and UP means differ.

Returns `(; M_inv, s_v_fdn, s_v_fup, s2sq_v_fdn, s2sq_v_fup, L_dn, L_up, ...)` with
`s2sq_v_fdn` / `s2sq_v_fup` the marginal `v` variances (third output of [`_rotated_sigma_uv`](@ref))
at lower/upper half-center `Σ`, and `s_v_fdn = sqrt(max(s2sq_v_fdn, 0))` (and similarly `fup`).
Half-center `σ_T`, `σ_q` match the `Σ_turb(z)` reconstruction
used for `s_u_cond_dn/up` (correlation in [`_profile_rosenblatt_emit_inner_sample`](@ref) remains `ρ_Tq`).
Degenerate/non-finite axis checks are handled by
`_two_slope_rosenblatt_has_valid_axis` before calling this constructor.
"""
@inline _mean_gradient_slopes(
    ::Val{:avg}, dT_dz_dn::FT, dT_dz_up::FT, dq_dz_dn::FT, dq_dz_up::FT,
) where {FT} = ((dT_dz_dn + dT_dz_up) / FT(2), (dq_dz_dn + dq_dz_up) / FT(2))

@inline _mean_gradient_slopes(
    ::Val{:dn}, dT_dz_dn::FT, dT_dz_up::FT, dq_dz_dn::FT, dq_dz_up::FT,
) where {FT} = (dT_dz_dn, dq_dz_dn)

@inline _mean_gradient_slopes(
    ::Val{:up}, dT_dz_dn::FT, dT_dz_up::FT, dq_dz_dn::FT, dq_dz_up::FT,
) where {FT} = (dT_dz_up, dq_dz_up)

function _two_slope_rosenblatt_params(
    μ_T::FT, μ_q::FT,
    σ_T²_c::FT, σ_q²_c::FT, ρ_Tq::FT,
    dT_dz_dn::FT, dT_dz_up::FT, dq_dz_dn::FT, dq_dz_up::FT,
    sTT_dn::FT, sTT_up::FT, sqq_dn::FT, sqq_up::FT,
    H::FT;
    mean_gradient_axis::Union{Val{:avg}, Val{:dn}, Val{:up}} = Val(:avg),
) where {FT}
    ε = ϵ_numerics(FT)
    # Mean-gradient direction for Rosenblatt / M_inv (must match Python `mean_gradient`).
    d_T, d_q = _mean_gradient_slopes(
        mean_gradient_axis, dT_dz_dn, dT_dz_up, dq_dz_dn, dq_dz_up,
    )
    α_sq = d_T^2 + d_q^2
    @assert α_sq > ε^2 "_two_slope_rosenblatt_params called with degenerate axis"
    inv_α = one(FT) / α_sq
    u_row_T = d_T * inv_α
    u_row_q = d_q * inv_α
    # Half-cell projections on u direction (clamped to zero for rare anti-alignment)
    α_dn_raw = u_row_T * dT_dz_dn + u_row_q * dq_dz_dn
    α_up_raw = u_row_T * dT_dz_up + u_row_q * dq_dz_up
    α_dn = max(α_dn_raw, zero(FT))
    α_up = max(α_up_raw, zero(FT))
    L_dn = α_dn * (H / FT(2))
    L_up = α_up * (H / FT(2))
    # Half-center Σ_turb from piecewise-linear reconstruction of σ²
    σT²_fdn = max(σ_T²_c - (H / FT(4)) * sTT_dn, zero(FT))
    σT²_fup = max(σ_T²_c + (H / FT(4)) * sTT_up, zero(FT))
    σq²_fdn = max(σ_q²_c - (H / FT(4)) * sqq_dn, zero(FT))
    σq²_fup = max(σ_q²_c + (H / FT(4)) * sqq_up, zero(FT))
    # ρ_Tq is a ClimaParams constant ⇒ half-center s_Tq = ρ √(σ_T² · σ_q²)
    ρc = clamp(ρ_Tq, -one(FT), one(FT))
    sTq_c = ρc * sqrt(max(σ_T²_c, zero(FT)) * max(σ_q²_c, zero(FT)))
    sTq_fdn = ρc * sqrt(σT²_fdn * σq²_fdn)
    sTq_fup = ρc * sqrt(σT²_fup * σq²_fup)
    # Std devs at each half-center (same Σ reconstruction as `s_u_cond_dn/up`).
    σ_T_fdn = sqrt(σT²_fdn)
    σ_q_fdn = sqrt(σq²_fdn)
    σ_T_fup = sqrt(σT²_fup)
    σ_q_fup = sqrt(σq²_fup)
    # Rotations (center → axes; half-centers → conditional std)
    s1sq_c, s12_c, s2sq_c = _rotated_sigma_uv(
        max(σ_T²_c, zero(FT)), max(σ_q²_c, zero(FT)), sTq_c,
        u_row_T, u_row_q, d_T, d_q,
    )
    s1sq_fdn, s12_fdn, s2sq_fdn = _rotated_sigma_uv(
        σT²_fdn, σq²_fdn, sTq_fdn, u_row_T, u_row_q, d_T, d_q,
    )
    s1sq_fup, s12_fup, s2sq_fup = _rotated_sigma_uv(
        σT²_fup, σq²_fup, sTq_fup, u_row_T, u_row_q, d_T, d_q,
    )
    # Marginal `v` variance at each half-center `Σ` (same frame as `s_u_cond_dn/up`); no averaging
    # of the two halves — each leg pairs its own `s_v_f*` / `r_f*` with the same Hermite node `χ[j]`.
    s2sq_v_fdn = s2sq_fdn
    s2sq_v_fup = s2sq_fup
    s_v_fdn = sqrt(max(s2sq_fdn, zero(FT)))
    s_v_fup = sqrt(max(s2sq_fup, zero(FT)))
    s2_c_eff = max(s2sq_c, ε)
    # Conditional σ_{u|v} at center and each half-center (piecewise-linear Σ_turb(z)).
    s_u_cond_c = sqrt(max(s1sq_c - s12_c^2 / s2_c_eff, zero(FT)))
    s2_fdn_eff = max(s2sq_fdn, ε)
    s2_fup_eff = max(s2sq_fup, ε)
    s_u_cond_dn = sqrt(max(s1sq_fdn - s12_fdn^2 / s2_fdn_eff, zero(FT)))
    s_u_cond_up = sqrt(max(s1sq_fup - s12_fup^2 / s2_fup_eff, zero(FT)))
    # Linear-in-z conditional mean μ_{u|v}(z) ≈ (s12/s2²)(z) · v  → slopes from center to half-center.
    r_c = s12_c / s2_c_eff
    r_fdn = s12_fdn / s2_fdn_eff
    r_fup = s12_fup / s2_fup_eff
    # M_inv maps (u, v) → (δT, δq). Columns: [d; d*inv_α ⊥]
    M_inv = SA.@SMatrix [
        d_T  -d_q*inv_α
        d_q   d_T*inv_α
    ]
    return (;
        M_inv, s_v_fdn, s_v_fup, s2sq_v_fdn, s2sq_v_fup,
        L_dn, L_up,
        s_u_cond_c, s_u_cond_dn, s_u_cond_up,
        r_c, r_fdn, r_fup,
        σ_T_fdn, σ_q_fdn, σ_T_fup, σ_q_fup,
    )
end

@inline function _two_slope_rosenblatt_has_valid_axis(
    μ_T::FT, μ_q::FT,
    σ_T²_c::FT, σ_q²_c::FT, ρ_Tq::FT,
    dT_dz_dn::FT, dT_dz_up::FT, dq_dz_dn::FT, dq_dz_up::FT,
    sTT_dn::FT, sTT_up::FT, sqq_dn::FT, sqq_up::FT,
    H::FT;
    mean_gradient_axis::Union{Val{:avg}, Val{:dn}, Val{:up}} = Val(:avg),
) where {FT}
    ε = ϵ_numerics(FT)
    if !(
        isfinite(μ_T) & isfinite(μ_q) &
        isfinite(σ_T²_c) & isfinite(σ_q²_c) & isfinite(ρ_Tq) &
        isfinite(dT_dz_dn) & isfinite(dT_dz_up) &
        isfinite(dq_dz_dn) & isfinite(dq_dz_up) &
        isfinite(sTT_dn) & isfinite(sTT_up) & isfinite(sqq_dn) & isfinite(sqq_up) &
        isfinite(H)
    )
        return false
    end
    d_T, d_q = _mean_gradient_slopes(
        mean_gradient_axis, dT_dz_dn, dT_dz_up, dq_dz_dn, dq_dz_up,
    )
    α_sq = d_T^2 + d_q^2
    return α_sq > ε^2
end

"""
    _profile_rosenblatt_emit_inner_sample(...)

Evaluate one Profile–Rosenblatt inner sample for a given inner quantile `ui`
and outer Gaussian deviate `vj`.

For `LogNormalSGS` in Profile–Rosenblatt, the local map is evaluated in
`(T, ln(q))`: the rotated state provides `(δT, δln(q))`, then `q = exp(ln(μ_q)+δln(q))`
is applied at the end. This avoids mixing a ln-space Rosenblatt construction with
the bivariate `get_physical_point(LogNormalSGS, ...)` mapping that expects
Gaussian `q` fluctuations in physical `q`.

For [`GaussianSGS`](@ref), `σ_q` / `σ_T` must match the turbulent second moments used to build the
inner `u` draw: [`_profile_rosenblatt_accumulate`](@ref) passes **half-center** `σ_T_fdn`/`σ_q_fdn`
or `σ_T_fup`/`σ_q_fup` from [`_two_slope_rosenblatt_params`](@ref) per inner leg (cell-center
values only for degenerate / collapsed-`u` branches).
"""
function _profile_rosenblatt_emit_inner_sample(
    acc,
    f,
    innerD,
    μ_q_p,
    μ_T_p,
    σ_q_c,
    σ_T_c,
    ρ_c,
    quad,
    M_inv,
    ui,
    μ_0,
    vj,
    wvj,
    wi,
    method,
)
    FT = typeof(μ_T_p)

    vec = SA.SVector(ui - μ_0, vj)
    δvec = M_inv * vec

    T_hat, q_hat = if innerD isa LogNormalSGS
        _physical_Tq_from_lnq_fluctuations(
            oftype(μ_T_p, μ_q_p),
            μ_T_p,
            oftype(μ_T_p, δvec[2]),
            oftype(μ_T_p, δvec[1]),
            oftype(μ_T_p, quad.T_min),
            oftype(μ_T_p, quad.q_max),
        )
    else
        _physical_Tq_from_fluctuations(
            innerD, μ_q_p, μ_T_p,
            δvec[2], δvec[1],
            σ_q_c, σ_T_c, oftype(μ_T_p, ρ_c),
            quad.T_min, quad.q_max,
        )
    end
    return acc ⊞ f(T_hat, q_hat) ⊠ wvj ⊠ wi
end

# ----------------------------------------------------------------------------
# Main kernel
# ----------------------------------------------------------------------------

"""
    integrate_over_sgs(
        f, quad, μ_q, μ_T, q′q′, T′T′, ρ_Tq,
        H, local_geometry,
        grad_q_dn, grad_q_up, grad_θ_dn, grad_θ_up, ∂T∂θ_li,
        grad_qq_dn, grad_qq_up, grad_TT_dn, grad_TT_up,
    )

Two-slope half-center-anchored layer-mean quadrature for
[`VerticallyResolvedSGS{S,I}`](@ref) with `S <: AbstractSubgridLayerProfileQuadrature`.

# Inputs
  - `μ_q, μ_T, q′q′, T′T′, ρ_Tq`: cell-center values (`ρ_Tq` is the ClimaParams
    correlation scalar, treated as constant across the cell).
  - `H, local_geometry`: cell thickness and `LocalGeometry` for the C3 → W
    projection.
  - `grad_{q,θ}_{dn,up}`: C3 vectors at center representing the half-cell
    mean-gradient fluxes of `q_tot` / `θ_li`. At the call site these are
    produced by `ᶜleft_bias(ᶠgradᵥ(·))` / `ᶜright_bias(ᶠgradᵥ(·))`;
    `θ_li` is rotated to T-space via `∂T∂θ_li` inside the kernel. No
    persistent face scratch fields are introduced.
  - `grad_{qq,TT}_{dn,up}`: C3 vectors at center for the half-cell gradients
    of the turbulent *variances* `q′q′`, `T′T′`. Enables the F2
    variance-gradient correction to `Σ_turb(z)`; zeros reduce to the
    classical frozen-Σ limit.

`f` is called as `f(T_hat, q_hat)`.

Dispatch on the layer quadrature type `S`:
  - `SubgridColumnTensor`: **Outer rule = depth integral** with **Gauss–Legendre**
    nodes `quad.z_t` / `quad.z_w`; inner **Gauss–Hermite** in `(T, q)` at each depth
    with piecewise-linear `μ(ζ), Σ(ζ)` → `N³` evaluations of `f`.
  - `SubgridLatinHypercubeZ`: same inner nodes and weights as the tensor, but pairs
    each `(χ_i, χ_j)` with depth index `k = 1 + mod(i + j - 2, N)` (cyclic LHS-style
    staggering) → `N²` evaluations.
  - `SubgridPrincipalAxisLayer`: at each of `N` depth nodes, one **1D** Gauss–Hermite
    step along the largest-eigenvector direction of the local `(T, q)` covariance
    (see [`_largest_eigval_unit_evec_Tq`](@ref)) → `N²` evaluations.
  - `SubgridVoronoiRepresentatives`: build the full `N³` column-tensor pool in
    `(k,i,j)`/`(z,χ₁,χ₂)`, cluster to `N²` Voronoi cells (deterministic weighted
    Lloyd iterations in normalized index space), and evaluate `f` at one pooled
    representative per cell with pooled mass.
  - `SubgridBarycentricSeeds`: initialize `N²` seeds over `(k,i)` and stream all
    `N³` candidates into nearest seeds with weighted centroid updates
    (deterministic order); evaluate `f` at seed centroids with pooled mass.
  - `SubgridProfileRosenblatt{M}`: Gauss–Hermite outer `v` × **composite** inner `p` rule
    (½ weight per shifted half-cell law; each GL abscissa used on DN and UP legs).
    `M` selects per-leg inversion: Brent, one-step Halley, or Chebyshev on the **single**
    centered `uniform⊛Gaussian` for each half ([`_rosenblatt_dn_half_u`](@ref) /
    [`_rosenblatt_up_half_u`](@ref)).
"""
function integrate_over_sgs(
    f,
    quad::SGSQuadrature{NQ, A, W, D, QFT, ZZ, ZW},
    μ_q, μ_T, q′q′, T′T′, ρ_Tq,
    H, local_geometry,
    grad_q_dn, grad_q_up,
    grad_θ_dn, grad_θ_up,
    ∂T∂θ_li,
    grad_qq_dn, grad_qq_up,
    grad_TT_dn, grad_TT_up,
) where {
    NQ, A, W,
    S <: SubgridProfileRosenblatt,
    I <: Union{GaussianSGS, LogNormalSGS},
    D <: VerticallyResolvedSGS{S, I},
    QFT, ZZ, ZW,
}
    dist = quad.dist
    μ_T_p, μ_q_p = promote(μ_T, μ_q)
    FT = typeof(μ_T_p)
    innerD = _inner_dist(dist)
    dq_dz_dn = Geometry.WVector(grad_q_dn, local_geometry)[1]
    dq_dz_up = Geometry.WVector(grad_q_up, local_geometry)[1]
    dθ_dz_dn = Geometry.WVector(grad_θ_dn, local_geometry)[1]
    dθ_dz_up = Geometry.WVector(grad_θ_up, local_geometry)[1]
    dT_dz_dn = ∂T∂θ_li * dθ_dz_dn
    dT_dz_up = ∂T∂θ_li * dθ_dz_up
    dqq_dz_dn = Geometry.WVector(grad_qq_dn, local_geometry)[1]
    dqq_dz_up = Geometry.WVector(grad_qq_up, local_geometry)[1]
    dTT_dz_dn = Geometry.WVector(grad_TT_dn, local_geometry)[1]
    dTT_dz_up = Geometry.WVector(grad_TT_up, local_geometry)[1]
    dq_dz_dn_eff = dq_dz_dn
    dq_dz_up_eff = dq_dz_up
    if innerD isa LogNormalSGS
        ε = ϵ_numerics(FT)
        q_face_dn = μ_q_p - dq_dz_dn * (H / FT(2))
        q_face_up = μ_q_p + dq_dz_up * (H / FT(2))
        ln_q_mean = log(max(μ_q_p, ε))
        ln_q_face_dn = log(max(q_face_dn, ε))
        ln_q_face_up = log(max(q_face_up, ε))
        dq_dz_dn_eff = (ln_q_mean - ln_q_face_dn) / (H / FT(2))
        dq_dz_up_eff = (ln_q_face_up - ln_q_mean) / (H / FT(2))
    end
    if !(
        isfinite(μ_T_p) & isfinite(μ_q_p) &
        isfinite(T′T′) & isfinite(q′q′) & isfinite(ρ_Tq) &
        isfinite(H) & isfinite(∂T∂θ_li) &
        isfinite(dq_dz_dn) & isfinite(dq_dz_up) &
        isfinite(dθ_dz_dn) & isfinite(dθ_dz_up) &
        isfinite(dT_dz_dn) & isfinite(dT_dz_up) &
        isfinite(dqq_dz_dn) & isfinite(dqq_dz_up) &
        isfinite(dTT_dz_dn) & isfinite(dTT_dz_up)
    )
        error(
            "integrate_over_sgs (layer-profile): non-finite SGS layer-mean inputs. " *
            "μ_T=$(μ_T_p), μ_q=$(μ_q_p), T′T′=$(T′T′), q′q′=$(q′q′), ρ_Tq=$(ρ_Tq), " *
            "H=$(H), ∂T∂θ=$(∂T∂θ_li), slopes: " *
            "dq=($(dq_dz_dn),$(dq_dz_up)), dθ=($(dθ_dz_dn),$(dθ_dz_up)), " *
            "dT=($(dT_dz_dn),$(dT_dz_up)), dqq=($(dqq_dz_dn),$(dqq_dz_up)), " *
            "dTT=($(dTT_dz_dn),$(dTT_dz_up)).",
        )
    end
    seed = f(μ_T_p, μ_q_p)
    return _integrate_over_sgs_profile_rosenblatt(
        f, quad, dist, innerD, μ_q_p, μ_T_p, q′q′, T′T′, ρ_Tq, H,
        dq_dz_dn_eff, dq_dz_up_eff, dT_dz_dn, dT_dz_up,
        dqq_dz_dn, dqq_dz_up, dTT_dz_dn, dTT_dz_up, seed,
    )::typeof(seed)
end

function integrate_over_sgs(
    f, quad::SGSQuadrature{NQ, A, W, D, QFT, ZZ, ZW},
    μ_q, μ_T, q′q′, T′T′, ρ_Tq,
    H, local_geometry,
    grad_q_dn, grad_q_up,
    grad_θ_dn, grad_θ_up,
    ∂T∂θ_li,
    grad_qq_dn, grad_qq_up,
    grad_TT_dn, grad_TT_up,
) where {NQ, A, W, D <: AbstractSGSDistribution, QFT, ZZ, ZW}
    dist = quad.dist
    μ_T_p, μ_q_p = promote(μ_T, μ_q)
    FT = typeof(μ_T_p)
    innerD = _inner_dist(dist)
    # Project C3 half-slopes to scalar ∂/∂z (no persistent fields; single inline projection).
    dq_dz_dn = Geometry.WVector(grad_q_dn, local_geometry)[1]
    dq_dz_up = Geometry.WVector(grad_q_up, local_geometry)[1]
    dθ_dz_dn = Geometry.WVector(grad_θ_dn, local_geometry)[1]
    dθ_dz_up = Geometry.WVector(grad_θ_up, local_geometry)[1]
    dT_dz_dn = ∂T∂θ_li * dθ_dz_dn
    dT_dz_up = ∂T∂θ_li * dθ_dz_up
    dqq_dz_dn = Geometry.WVector(grad_qq_dn, local_geometry)[1]
    dqq_dz_up = Geometry.WVector(grad_qq_up, local_geometry)[1]
    dTT_dz_dn = Geometry.WVector(grad_TT_dn, local_geometry)[1]
    dTT_dz_up = Geometry.WVector(grad_TT_up, local_geometry)[1]
    dq_dz_dn_eff = dq_dz_dn
    dq_dz_up_eff = dq_dz_up
    if innerD isa LogNormalSGS
        # Keep one vertical model for all vertically resolved LogNormal schemes:
        # convert reconstructed q-face values to ln(q), then reproject as slopes.
        ε = ϵ_numerics(FT)
        q_face_dn = μ_q_p - dq_dz_dn * (H / FT(2))
        q_face_up = μ_q_p + dq_dz_up * (H / FT(2))
        ln_q_mean = log(max(μ_q_p, ε))
        ln_q_face_dn = log(max(q_face_dn, ε))
        ln_q_face_up = log(max(q_face_up, ε))
        dq_dz_dn_eff = (ln_q_mean - ln_q_face_dn) / (H / FT(2))
        dq_dz_up_eff = (ln_q_face_up - ln_q_mean) / (H / FT(2))
    end
    # Reject any non-finite half-cell means / slopes / F2 terms before the various
    # quadrature paths (prevents `NaN` from bypassing the `α_sq <= ε` early exit in
    # `_two_slope_rosenblatt_params` and then poisoning the inner `u`/`v` map).
    if !(
        isfinite(μ_T_p) & isfinite(μ_q_p) &
        isfinite(T′T′) & isfinite(q′q′) & isfinite(ρ_Tq) &
        isfinite(H) & isfinite(∂T∂θ_li) &
        isfinite(dq_dz_dn) & isfinite(dq_dz_up) &
        isfinite(dθ_dz_dn) & isfinite(dθ_dz_up) &
        isfinite(dT_dz_dn) & isfinite(dT_dz_up) &
        isfinite(dqq_dz_dn) & isfinite(dqq_dz_up) &
        isfinite(dTT_dz_dn) & isfinite(dTT_dz_up)
    )
        error(
            "integrate_over_sgs (layer-profile): non-finite SGS layer-mean inputs. " *
            "μ_T=$(μ_T_p), μ_q=$(μ_q_p), T′T′=$(T′T′), q′q′=$(q′q′), ρ_Tq=$(ρ_Tq), " *
            "H=$(H), ∂T∂θ=$(∂T∂θ_li), slopes: " *
            "dq=($(dq_dz_dn),$(dq_dz_up)), dθ=($(dθ_dz_dn),$(dθ_dz_up)), " *
            "dT=($(dT_dz_dn),$(dT_dz_up)), dqq=($(dqq_dz_dn),$(dqq_dz_up)), " *
            "dTT=($(dTT_dz_dn),$(dTT_dz_up)).",
        )
    end
    if dist isa VerticallyResolvedSGS{<:SubgridProfileRosenblatt}
        seed = f(μ_T_p, μ_q_p)
        return _integrate_over_sgs_profile_rosenblatt(
            f, quad, dist, innerD, μ_q_p, μ_T_p, q′q′, T′T′, ρ_Tq, H,
            dq_dz_dn_eff, dq_dz_up_eff, dT_dz_dn, dT_dz_up,
            dqq_dz_dn, dqq_dz_up, dTT_dz_dn, dTT_dz_up, seed,
        )::typeof(seed)
    end
    # Center stddevs for inner Hermite (clamped / Cauchy–Schwarz enforced).
    σ_q_c, σ_T_c, ρ_c = sgs_stddevs_and_correlation(q′q′, T′T′, ρ_Tq)

    # ------------------------------------------------------------------
    # SubgridColumnTensor (outer GL × inner mapped GH on `(T,q)` at each depth).
    # ------------------------------------------------------------------
    if dist isa VerticallyResolvedSGS{SubgridColumnTensor}
        quad.z_t === nothing && error(
            "SGSQuadrature requires z-axis nodes for SubgridColumnTensor.",
        )
        Nz = length(quad.z_t)
        acc = rzero(f(μ_T_p, μ_q_p))
        @inbounds for k in 1:Nz
            ζ = (H / FT(2)) * quad.z_t[k]
            wk = quad.z_w[k] / FT(2)
            sμ_T = _pick_half_slope(ζ, dT_dz_dn, dT_dz_up)
            sσT = _pick_half_slope(ζ, dTT_dz_dn, dTT_dz_up)
            sσq = _pick_half_slope(ζ, dqq_dz_dn, dqq_dz_up)
            μ_Tk = μ_T_p + ζ * sμ_T
            μ_qk = _vertical_layer_mean_q(
                innerD, μ_q_p, ζ, dq_dz_dn_eff, dq_dz_up_eff, FT,
            )
            σT²k = max(T′T′ + ζ * sσT, zero(FT))
            σq²k = max(q′q′ + ζ * sσq, zero(FT))
            σ_Tk = sqrt(σT²k)
            σ_qk = sqrt(σq²k)
            transform = PhysicalPointTransform(
                innerD,
                μ_Tk, μ_qk,
                oftype(μ_T_p, σ_Tk), oftype(μ_T_p, σ_qk),
                oftype(μ_T_p, ρ_c),
                oftype(μ_T_p, quad.T_min), oftype(μ_T_p, quad.q_max),
            )
            inner_quad = SGSQuadrature(
                FT;
                quadrature_order = quadrature_order(quad),
                distribution = innerD,
                T_min = quad.T_min,
                q_max = quad.q_max,
            )
            acc = acc ⊞ sum_over_quadrature_points(f, transform, inner_quad) ⊠ wk
        end
        return acc
    end

    # ------------------------------------------------------------------
    # LHS-style pairing: N² (i,j) tensor with z level k = 1+mod(i+j-2, N)
    # (cyclic Latin-style stratification; see `SubgridLatinHypercubeZ`).
    # ------------------------------------------------------------------
    if dist isa VerticallyResolvedSGS{SubgridLatinHypercubeZ}
        quad.z_t === nothing &&
            error("SGSQuadrature requires z-axis nodes for SubgridLatinHypercubeZ.")
        N = length(quad.z_t)
        χ, wgh = quad.a, quad.w
        inv_sqrt_pi = one(FT) / sqrt(FT(π))
        acc = rzero(f(μ_T_p, μ_q_p))
        wsum = zero(FT)
        @inbounds for i in 1:N
            @inbounds for j in 1:N
                k = mod(i + j - 2, N) + 1
                ζ = (H / FT(2)) * quad.z_t[k]
                w_z = quad.z_w[k] / FT(2)
                wi, wj = wgh[i] * inv_sqrt_pi, wgh[j] * inv_sqrt_pi
                wp = w_z * wi * wj
                sμ_T = _pick_half_slope(ζ, dT_dz_dn, dT_dz_up)
                sσT = _pick_half_slope(ζ, dTT_dz_dn, dTT_dz_up)
                sσq = _pick_half_slope(ζ, dqq_dz_dn, dqq_dz_up)
                μ_Tk = μ_T_p + ζ * sμ_T
                μ_qk = _vertical_layer_mean_q(
                    innerD, μ_q_p, ζ, dq_dz_dn_eff, dq_dz_up_eff, FT,
                )
                σT²k = max(T′T′ + ζ * sσT, zero(FT))
                σq²k = max(q′q′ + ζ * sσq, zero(FT))
                σ_Tk = sqrt(σT²k)
                σ_qk = sqrt(σq²k)
                transform = PhysicalPointTransform(
                    innerD, μ_Tk, μ_qk,
                    oftype(μ_T_p, σ_Tk), oftype(μ_T_p, σ_qk),
                    oftype(μ_T_p, ρ_c), oftype(μ_T_p, quad.T_min),
                    oftype(μ_T_p, quad.q_max),
                )
                T_hat, q_hat = transform(χ[i], χ[j])
                acc = acc ⊞ f(T_hat, q_hat) ⊠ wp
                wsum += wp
            end
        end
        return wsum > zero(FT) ? acc ⊠ (one(FT) / wsum) : acc
    end

    # ------------------------------------------------------------------
    # Principal axis: N depth nodes × 1D Hermite on dominant covariance axis (`SubgridPrincipalAxisLayer`).
    # ------------------------------------------------------------------
    if dist isa VerticallyResolvedSGS{SubgridPrincipalAxisLayer}
        quad.z_t === nothing &&
            error("SGSQuadrature requires z-axis nodes for SubgridPrincipalAxisLayer.")
        N = length(quad.z_t)
        χ, wgh = quad.a, quad.w
        inv_sqrt_pi = one(FT) / sqrt(FT(π))
        acc = rzero(f(μ_T_p, μ_q_p))
        sqrt2 = sqrt(FT(2))
        @inbounds for kz in 1:N
            ζ = (H / FT(2)) * quad.z_t[kz]
            wk = quad.z_w[kz] / FT(2)
            sμ_T = _pick_half_slope(ζ, dT_dz_dn, dT_dz_up)
            sσT = _pick_half_slope(ζ, dTT_dz_dn, dTT_dz_up)
            sσq = _pick_half_slope(ζ, dqq_dz_dn, dqq_dz_up)
            μ_Tk = μ_T_p + ζ * sμ_T
            μ_qk = _vertical_layer_mean_q(
                innerD, μ_q_p, ζ, dq_dz_dn_eff, dq_dz_up_eff, FT,
            )
            σT²k = max(T′T′ + ζ * sσT, zero(FT))
            σq²k = max(q′q′ + ζ * sσq, zero(FT))
            σ_Tk = sqrt(σT²k)
            σ_qk = sqrt(σq²k)
            ρ_σT_σq = ρ_c * σ_Tk * σ_qk
            λ1, u1, u2 = _largest_eigval_unit_evec_Tq(σT²k, ρ_σT_σq, σq²k)
            @inbounds for m in 1:N
                # `λ1` = variance along principal axis; match GH scaling to column tensor.
                sm = sqrt2 * sqrt(λ1) * χ[m]
                δT = sm * u1
                δq = sm * u2
                T_hat, q_hat = _physical_Tq_from_fluctuations(
                    innerD, μ_qk, μ_Tk, δq, δT, σ_qk, σ_Tk,
                    oftype(μ_T_p, ρ_c), quad.T_min, quad.q_max,
                )
                wi = wgh[m] * inv_sqrt_pi
                acc = acc ⊞ f(T_hat, q_hat) ⊠ wk ⊠ wi
            end
        end
        return acc
    end

    # ------------------------------------------------------------------
    # Voronoi-style: weighted clustering of full N³ pool into N² representatives.
    # ------------------------------------------------------------------
    if dist isa VerticallyResolvedSGS{SubgridVoronoiRepresentatives}
        quad.z_t === nothing &&
            error("SGSQuadrature requires z-axis nodes for SubgridVoronoiRepresentatives.")
        N = length(quad.z_t)
        χ, wgh = quad.a, quad.w
        inv_sqrt_pi = one(FT) / sqrt(FT(π))
        n3 = N^3
        n2 = N^2
        cand_T = Vector{FT}(undef, n3)
        cand_q = Vector{FT}(undef, n3)
        cand_w = Vector{FT}(undef, n3)
        cand_x = Vector{SA.SVector{3, FT}}(undef, n3)
        idx = 0
        @inbounds for k in 1:N
            ζ = (H / FT(2)) * quad.z_t[k]
            sμ_T = _pick_half_slope(ζ, dT_dz_dn, dT_dz_up)
            sσT = _pick_half_slope(ζ, dTT_dz_dn, dTT_dz_up)
            sσq = _pick_half_slope(ζ, dqq_dz_dn, dqq_dz_up)
            μ_Tk = μ_T_p + ζ * sμ_T
            μ_qk = _vertical_layer_mean_q(
                innerD, μ_q_p, ζ, dq_dz_dn_eff, dq_dz_up_eff, FT,
            )
            σT²k = max(T′T′ + ζ * sσT, zero(FT))
            σq²k = max(q′q′ + ζ * sσq, zero(FT))
            σ_Tk = sqrt(σT²k)
            σ_qk = sqrt(σq²k)
            transform = PhysicalPointTransform(
                innerD, μ_Tk, μ_qk, oftype(μ_T_p, σ_Tk), oftype(μ_T_p, σ_qk),
                oftype(μ_T_p, ρ_c), oftype(μ_T_p, quad.T_min), oftype(μ_T_p, quad.q_max),
            )
            wz = quad.z_w[k] / FT(2)
            @inbounds for i in 1:N, j in 1:N
                idx += 1
                T_hat, q_hat = transform(χ[i], χ[j])
                wi = wgh[i] * inv_sqrt_pi
                wj = wgh[j] * inv_sqrt_pi
                cand_T[idx] = T_hat
                cand_q[idx] = q_hat
                cand_w[idx] = wz * wi * wj
                cand_x[idx] = _normalized_kij_coord(k, i, j, FT, N)
            end
        end
        # Deterministic seed indices: evenly spread across lexicographic pool.
        seed_idx = Vector{Int}(undef, n2)
        if n2 == 1
            seed_idx[1] = 1
        else
            @inbounds for s in 1:n2
                seed_idx[s] = round(Int, 1 + (s - 1) * (n3 - 1) / (n2 - 1))
            end
        end
        seed_x = [cand_x[seed_idx[s]] for s in 1:n2]
        assign = Vector{Int}(undef, n3)
        # Weighted Lloyd updates in normalized index space.
        for _ in 1:3
            @inbounds for p in 1:n3
                best_s = 1
                best_d = typemax(FT)
                xp = cand_x[p]
                for s in 1:n2
                    ds = sum(abs2, xp - seed_x[s])
                    if ds < best_d
                        best_d = ds
                        best_s = s
                    end
                end
                assign[p] = best_s
            end
            mass = zeros(FT, n2)
            xsum = fill(SA.SVector{3, FT}(zero(FT), zero(FT), zero(FT)), n2)
            @inbounds for p in 1:n3
                s = assign[p]
                wp = cand_w[p]
                mass[s] += wp
                xsum[s] = xsum[s] + wp * cand_x[p]
            end
            @inbounds for s in 1:n2
                if mass[s] > zero(FT)
                    seed_x[s] = xsum[s] / mass[s]
                end
            end
        end
        # Representative: nearest pool point to final Voronoi centroid.
        rep = copy(seed_idx)
        msum = zeros(FT, n2)
        @inbounds for s in 1:n2
            best_p = rep[s]
            best_d = typemax(FT)
            for p in 1:n3
                if assign[p] == s
                    ds = sum(abs2, cand_x[p] - seed_x[s])
                    if ds < best_d
                        best_d = ds
                        best_p = p
                    end
                    msum[s] += cand_w[p]
                end
            end
            rep[s] = best_p
        end
        acc = rzero(f(μ_T_p, μ_q_p))
        @inbounds for s in 1:n2
            if msum[s] > zero(FT)
                acc = acc ⊞ f(cand_T[rep[s]], cand_q[rep[s]]) ⊠ msum[s]
            end
        end
        return acc
    end

    # ------------------------------------------------------------------
    # Barycentric-style seeds: stream full N³ pool into N² nearest seed centroids.
    # ------------------------------------------------------------------
    if dist isa VerticallyResolvedSGS{SubgridBarycentricSeeds}
        quad.z_t === nothing &&
            error("SGSQuadrature requires z-axis nodes for SubgridBarycentricSeeds.")
        N = length(quad.z_t)
        χ, wgh = quad.a, quad.w
        inv_sqrt_pi = one(FT) / sqrt(FT(π))
        n2 = N^2
        seed_x = Vector{SA.SVector{3, FT}}(undef, n2)
        seed_m = zeros(FT, n2)
        seed_T = zeros(FT, n2)
        seed_q = zeros(FT, n2)
        center_j = (N + 1) ÷ 2
        @inbounds for m in 1:n2
            i0 = m - 1
            k = i0 ÷ N + 1
            i = i0 - (k - 1) * N + 1
            seed_x[m] = _normalized_kij_coord(k, i, center_j, FT, N)
        end
        @inbounds for k in 1:N
            ζ = (H / FT(2)) * quad.z_t[k]
            sμ_T = _pick_half_slope(ζ, dT_dz_dn, dT_dz_up)
            sσT = _pick_half_slope(ζ, dTT_dz_dn, dTT_dz_up)
            sσq = _pick_half_slope(ζ, dqq_dz_dn, dqq_dz_up)
            μ_Tk = μ_T_p + ζ * sμ_T
            μ_qk = _vertical_layer_mean_q(
                innerD, μ_q_p, ζ, dq_dz_dn_eff, dq_dz_up_eff, FT,
            )
            σT²k = max(T′T′ + ζ * sσT, zero(FT))
            σq²k = max(q′q′ + ζ * sσq, zero(FT))
            σ_Tk = sqrt(σT²k)
            σ_qk = sqrt(σq²k)
            transform = PhysicalPointTransform(
                innerD, μ_Tk, μ_qk, oftype(μ_T_p, σ_Tk), oftype(μ_T_p, σ_qk),
                oftype(μ_T_p, ρ_c), oftype(μ_T_p, quad.T_min), oftype(μ_T_p, quad.q_max),
            )
            wz = quad.z_w[k] / FT(2)
            for i in 1:N, j in 1:N
                wi = wgh[i] * inv_sqrt_pi
                wj = wgh[j] * inv_sqrt_pi
                wp = wz * wi * wj
                T_hat, q_hat = transform(χ[i], χ[j])
                xp = _normalized_kij_coord(k, i, j, FT, N)
                best_s = 1
                best_d = typemax(FT)
                for s in 1:n2
                    ds = sum(abs2, xp - seed_x[s])
                    if ds < best_d
                        best_d = ds
                        best_s = s
                    end
                end
                m_old = seed_m[best_s]
                m_new = m_old + wp
                if m_new > zero(FT)
                    seed_x[best_s] = (m_old * seed_x[best_s] + wp * xp) / m_new
                    seed_T[best_s] = (m_old * seed_T[best_s] + wp * T_hat) / m_new
                    seed_q[best_s] = (m_old * seed_q[best_s] + wp * q_hat) / m_new
                    seed_m[best_s] = m_new
                end
            end
        end
        acc = rzero(f(μ_T_p, μ_q_p))
        @inbounds for s in 1:n2
            if seed_m[s] > zero(FT)
                acc = acc ⊞ f(seed_T[s], seed_q[s]) ⊠ seed_m[s]
            end
        end
        return acc
    end

    error("integrate_over_sgs (layer-profile): unsupported distribution $(typeof(dist)).")
end
