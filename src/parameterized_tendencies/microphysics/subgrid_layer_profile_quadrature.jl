# Layer-mean profile quadrature for gridscale-corrected SGS distributions
# (column-tensor and profile–Rosenblatt schemes). Two-slope face-anchored
# reconstruction per
# `calibration/experiments/variance_adjustments/OverleafPaper/sections/_method_notes_1.tex`.
# No persistent scratch fields; all half-slopes are consumed as scalar inputs
# at the call site (see microphysics_cache.jl broadcasts).
#
# Scheme III (`SubgridProfileRosenblatt`): outer Hermite in `v` × inner
# Gauss–Legendre in `p`, inverting the CDF of the **two-component**
# uniform–Gaussian mixture along the inner `u` axis. Each half contributes
# one convolution with **face-anchored** conditional std `σ_{u|v}` (constant
# on that half’s segment in `u`). There is **no** implemented closed form for a
# linear-in-`z` conditional variance inside the half’s inner marginal; variance
# slopes from the cache still enter the **face** reconstruction that sets
# `s_u_cond_dn` / `s_u_cond_up`. The `uniform_gaussian_convolution_*` primitives
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

This is **not** the two-component layer mixture; for that use
[`mixture_convolution_quantile_brent`](@ref) / [`mixture_convolution_quantile_halley`](@ref).
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
    coeffs = chebyshev_convolution_coeffs(FT, N_gl, i_node)
    return chebyshev_evaluate(coeffs, τ) * Lp
end

# ----------------------------------------------------------------------------
# Two-component uniform–Gaussian convolution mixture (Scheme III inner `u`).
#
# The u-marginal along the mean-gradient direction is a length-weighted
# (½, ½) mixture of two uniform–Gaussian convolutions, one per half-cell:
#     lower half-cell component: uniform[-L_-, 0] ⊛ N(0, s_-²)   (mean -L_-/2)
#     upper half-cell component: uniform[ 0, L_+] ⊛ N(0, s_+²)   (mean +L_+/2)
# Each half-cell σ is a single scalar (face-anchored evaluation of the
# conditional std), so σ² is held constant within each half-cell for the inner
# `u` marginal. Deeper `z`–`(T,q)` coupling is handled by `SubgridColumnTensor`
# and the alternate §3 layouts (LHS / Voronoi / barycentric / principal axis),
# not by a second analytic branch here.
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

"""
    mixture_uniform_gaussian_convolution_pdf_prime(x, L_dn, s_dn, L_up, s_up)

Derivative of the two-component uniform–Gaussian mixture PDF. Used by the
Halley quantile step.
"""
@inline function mixture_uniform_gaussian_convolution_pdf_prime(
    x::FT, L_dn::FT, s_dn::FT, L_up::FT, s_up::FT,
) where {FT}
    fp_dn = _uniform_gaussian_convolution_pdf_prime_shifted(
        x, -L_dn, zero(FT), s_dn,
    )
    fp_up = _uniform_gaussian_convolution_pdf_prime_shifted(
        x, zero(FT), L_up, s_up,
    )
    return (fp_dn + fp_up) / FT(2)
end

"""
    mixture_uniform_gaussian_convolution_mean_var(L_dn, s_dn, L_up, s_up)

Closed-form mean and variance of the two-component uniform–Gaussian mixture.
Variance decomposition:
`var = (5L_+² + 6 L_− L_+ + 5L_−²)/48 + (s_−² + s_+²)/2`
(the first term is the length-weighted mixture of the two shifted uniforms,
the second is the mean of the two local Gaussian variances). Derivation in
`OverleafPaper/sections/_method_notes_1.tex`.
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

# ----------------------------------------------------------------------------
# Quantile inversion on the two-component uniform–Gaussian mixture CDF.
# Production default is Halley (one step, ≈ 4 erf/exp per call).
# Brent is a robust fallback / validation reference.
# Chebyshev surrogate is a dispatch placeholder in the mixture setting
# (the one-parameter `η = s/L` tabulation for the single-component
# uniform–Gaussian convolution does not generalize to the 4-parameter mixture;
# dedicated re-tabulation is future work).
# ----------------------------------------------------------------------------

"""
    mixture_convolution_quantile_brent(p, L_dn, s_dn, L_up, s_up)

Bracketed Brent root of `F_mix(u) - p = 0`, where `F_mix` is the CDF of the
two-component uniform–Gaussian mixture
([`mixture_uniform_gaussian_convolution_cdf`](@ref)). Initial bracket uses
the union support `[-L_−, L_+]` padded by `6·max(s_−, s_+)` for tail
coverage. Robust but costs `O(10–20)` CDF evaluations per call; prefer
`mixture_convolution_quantile_halley` in production.
"""
function mixture_convolution_quantile_brent(
    p::FT, L_dn::FT, s_dn::FT, L_up::FT, s_up::FT,
) where {FT}
    ε = ϵ_numerics(FT)
    s_max = max(s_dn, s_up, ε)
    lo = -L_dn - FT(6) * s_max
    hi = L_up + FT(6) * s_max
    f =
        x ->
            mixture_uniform_gaussian_convolution_cdf(
                x, L_dn, s_dn, L_up, s_up,
            ) - p
    sol = RS.find_zero(
        f,
        RS.BrentsMethod{FT}(lo, hi),
        RS.CompactSolution(),
    )
    return sol.root
end

"""
    mixture_convolution_quantile_halley(p, L_dn, s_dn, L_up, s_up)

**Production default**. One Halley correction to `F_mix(u) = p` from a
closed-form Cornish–Fisher initial guess on the moment-matched Gaussian of
the two-component uniform–Gaussian mixture. Total cost ≈ 4 `erf` / `exp`
calls per quantile, no rootfinding. Mixture moments from
[`mixture_uniform_gaussian_convolution_mean_var`](@ref).
"""
function mixture_convolution_quantile_halley(
    p::FT, L_dn::FT, s_dn::FT, L_up::FT, s_up::FT,
) where {FT}
    ε = ϵ_numerics(FT)
    t = FT(2) * p - one(FT)
    t = clamp(t, -one(FT) + FT(100) * ε, one(FT) - FT(100) * ε)
    z = sqrt(FT(2)) * SF.erfinv(t)
    μ_mix, var_mix =
        mixture_uniform_gaussian_convolution_mean_var(L_dn, s_dn, L_up, s_up)
    σ_mix = sqrt(max(var_mix, ε^2))
    u = μ_mix + σ_mix * z
    Fv = mixture_uniform_gaussian_convolution_cdf(u, L_dn, s_dn, L_up, s_up)
    g = Fv - p
    fv = mixture_uniform_gaussian_convolution_pdf(u, L_dn, s_dn, L_up, s_up)
    abs(fv) < ε && return u
    fpv = mixture_uniform_gaussian_convolution_pdf_prime(
        u, L_dn, s_dn, L_up, s_up,
    )
    denom = FT(2) * fv^2 - g * fpv
    # Stabilize division: |denom| floored at 1e-15, sign preserved.
    denom = copysign(
        max(abs(denom), FT(1.0e-15)),
        denom == zero(FT) ? one(FT) : denom,
    )
    return u - FT(2) * fv * g / denom
end

@inline function _mixture_quantile_u(
    ::ConvolutionQuantilesHalley,
    p::FT, L_dn::FT, s_dn::FT, L_up::FT, s_up::FT,
) where {FT}
    return mixture_convolution_quantile_halley(p, L_dn, s_dn, L_up, s_up)
end

@inline function _mixture_quantile_u(
    ::ConvolutionQuantilesBracketed,
    p::FT, L_dn::FT, s_dn::FT, L_up::FT, s_up::FT,
) where {FT}
    return mixture_convolution_quantile_brent(p, L_dn, s_dn, L_up, s_up)
end

function _mixture_quantile_u(
    ::ConvolutionQuantilesChebyshevLogEta,
    p::FT, L_dn::FT, s_dn::FT, L_up::FT, s_up::FT,
) where {FT}
    error(
        "ConvolutionQuantilesChebyshevLogEta: the single-component " *
        "uniform–Gaussian convolution `η = s/L` Chebyshev tabulation does " *
        "not carry over to the two-slope two-component uniform–Gaussian " *
        "mixture (4-parameter family (L_-, L_+, s_-, s_+)). The dispatch " *
        "type is kept as a placeholder for a future mixture Chebyshev " *
        "re-tabulation. Use `ConvolutionQuantilesHalley` (production " *
        "default) or `ConvolutionQuantilesBracketed` (Brent safety net).",
    )
end

# ----------------------------------------------------------------------------
# Inner-distribution mapping and physical-point recovery.
# ----------------------------------------------------------------------------

@inline _inner_dist_linear_profile(::GaussianGridscaleCorrectedSGS) = GaussianSGS()
@inline _inner_dist_linear_profile(::LogNormalGridscaleCorrectedSGS) = LogNormalSGS()

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

# ----------------------------------------------------------------------------
# Two-slope helpers (piecewise-linear reconstruction; no persistent fields).
# ----------------------------------------------------------------------------

@inline function _pick_half_slope(ζ::FT, s_dn::FT, s_up::FT) where {FT}
    return ζ >= zero(FT) ? s_up : s_dn
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
    _two_slope_rosenblatt_params(...)

Assemble Scheme III **inner marginal** parameters: half-widths `L_±`,
conditional stds at **cell center and each face** (from piecewise-linear
turbulent covariance in `(T,q)` then rotation to `(u,v)`), and the ratios
`r = s12/s2²` at center and faces so that for fixed outer `v`,
`μ_{u|v}(z) ≈ μ_0 + μ'_z z` with `μ_0 = r_c v` and linear slopes in each half.
Conditional variance `σ²_{u|v}(z)` is taken **linear from center to face** on
each half (the segment law in Overleaf `sections/_method_notes_1.tex`,
subsection on ``Ψ_ν`` / half-cell segments; each Ψ_ν uses `SpecialFunctions`
(`gamma`, `erfc`, `erfcx`, …). `ξ_01`, `w_01` from `quadrature_order(quad)` feed
the outer `p`-rule (unused inside the segment primitives).

Returns `(; M_inv, s_v, L_dn, L_up, s_u_cond_c, s_u_cond_dn, s_u_cond_up, r_c, r_fdn, r_fup)` or
`nothing` when the **chosen** transport gradient is degenerate (`‖d‖² ≤ ε²`).

`transport`:
  - `:avg` — legacy centered mean gradient (average of DN/UP slopes).
  - `:dn` / `:up` — DN-only or UP-only mean-gradient vector for Rosenblatt axes (two-half cell;
    profile–Rosen cubature averages the `:dn` and `:up` accumulations).
"""
function _two_slope_rosenblatt_params(
    μ_T::FT, μ_q::FT,
    σ_T²_c::FT, σ_q²_c::FT, ρ_Tq::FT,
    dT_dz_dn::FT, dT_dz_up::FT, dq_dz_dn::FT, dq_dz_up::FT,
    sTT_dn::FT, sTT_up::FT, sqq_dn::FT, sqq_up::FT,
    H::FT;
    transport::Symbol = :avg,
) where {FT}
    ε = ϵ_numerics(FT)
    # Mean-gradient direction for Rosenblatt / M_inv (must match Python `mean_gradient`).
    if transport == :avg
        d_T = (dT_dz_dn + dT_dz_up) / FT(2)
        d_q = (dq_dz_dn + dq_dz_up) / FT(2)
    elseif transport == :dn
        d_T = dT_dz_dn
        d_q = dq_dz_dn
    elseif transport == :up
        d_T = dT_dz_up
        d_q = dq_dz_up
    else
        error("transport must be :avg, :dn, or :up, got $(repr(transport))")
    end
    α_sq = d_T^2 + d_q^2
    if α_sq <= ε^2
        return nothing
    end
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
    # Face-level Σ_turb from piecewise-linear reconstruction of σ²
    σT²_fdn = max(σ_T²_c - (H / FT(2)) * sTT_dn, zero(FT))
    σT²_fup = max(σ_T²_c + (H / FT(2)) * sTT_up, zero(FT))
    σq²_fdn = max(σ_q²_c - (H / FT(2)) * sqq_dn, zero(FT))
    σq²_fup = max(σ_q²_c + (H / FT(2)) * sqq_up, zero(FT))
    # ρ_Tq is a ClimaParams constant ⇒ face s_Tq = ρ √(σ_T² · σ_q²)
    ρc = clamp(ρ_Tq, -one(FT), one(FT))
    sTq_c = ρc * sqrt(max(σ_T²_c, zero(FT)) * max(σ_q²_c, zero(FT)))
    sTq_fdn = ρc * sqrt(σT²_fdn * σq²_fdn)
    sTq_fup = ρc * sqrt(σT²_fup * σq²_fup)
    # Rotations (center → axes; faces → conditional std)
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
    # Outer v axis uses centered Σ_turb (consistent with outer Gaussian-v quadrature).
    s_v = sqrt(max(s2sq_c, zero(FT)))
    s2_c_eff = max(s2sq_c, ε)
    # Conditional σ_{u|v} at center and each face (piecewise-linear Σ_turb(z)).
    s_u_cond_c = sqrt(max(s1sq_c - s12_c^2 / s2_c_eff, zero(FT)))
    s2_fdn_eff = max(s2sq_fdn, ε)
    s2_fup_eff = max(s2sq_fup, ε)
    s_u_cond_dn = sqrt(max(s1sq_fdn - s12_fdn^2 / s2_fdn_eff, zero(FT)))
    s_u_cond_up = sqrt(max(s1sq_fup - s12_fup^2 / s2_fup_eff, zero(FT)))
    # Linear-in-z conditional mean μ_{u|v}(z) ≈ (s12/s2²)(z) · v  → slopes from center to face.
    r_c = s12_c / s2_c_eff
    r_fdn = s12_fdn / s2_fdn_eff
    r_fup = s12_fup / s2_fup_eff
    # M_inv maps (u, v) → (δT, δq). Columns: [d; d*inv_α ⊥]
    M_inv = SA.@SMatrix [
        d_T  -d_q*inv_α
        d_q   d_T*inv_α
    ]
    return (;
        M_inv, s_v,
        L_dn, L_up,
        s_u_cond_c, s_u_cond_dn, s_u_cond_up,
        r_c, r_fdn, r_fup,
    )
end

# ----------------------------------------------------------------------------
# Main kernel
# ----------------------------------------------------------------------------

"""
    integrate_over_sgs_linear_profile(
        f, quad, μ_q, μ_T, q′q′, T′T′, ρ_Tq,
        H, local_geometry,
        grad_q_dn, grad_q_up, grad_θ_dn, grad_θ_up, ∂T∂θ_li,
        grad_qq_dn, grad_qq_up, grad_TT_dn, grad_TT_up,
    )

Two-slope face-anchored layer-mean quadrature for
`GaussianGridscaleCorrectedSGS{S}` / `LogNormalGridscaleCorrectedSGS{S}`
with `S <: AbstractSubgridLayerProfileQuadrature`.

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

Scheme dispatch:
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
  - `SubgridProfileRosenblatt{M}`: Gauss–Hermite outer `v` × Gauss–Legendre inner `p`,
    inverting the **two-component uniform–Gaussian mixture** CDF via
    [`_mixture_quantile_u`](@ref) per `M`.
"""
function integrate_over_sgs_linear_profile(
    f, quad::SGSQuadrature,
    μ_q, μ_T, q′q′, T′T′, ρ_Tq,
    H, local_geometry,
    grad_q_dn, grad_q_up,
    grad_θ_dn, grad_θ_up,
    ∂T∂θ_li,
    grad_qq_dn, grad_qq_up,
    grad_TT_dn, grad_TT_up,
)
    dist = quad.dist
    μ_T_p, μ_q_p = promote(μ_T, μ_q)
    FT = typeof(μ_T_p)
    innerD = _inner_dist_linear_profile(dist)
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
    # Center stddevs for inner Hermite (clamped / Cauchy–Schwarz enforced).
    σ_q_c, σ_T_c, ρ_c = sgs_stddevs_and_correlation(q′q′, T′T′, ρ_Tq)

    # ------------------------------------------------------------------
    # Scheme II: Column-Tensor (outer GL × inner mapped GH).
    # ------------------------------------------------------------------
    if dist isa GaussianGridscaleCorrectedSGS{SubgridColumnTensor} ||
       dist isa LogNormalGridscaleCorrectedSGS{SubgridColumnTensor}
        quad.z_t === nothing && error(
            "SGSQuadrature requires z-axis nodes for SubgridColumnTensor.",
        )
        Nz = length(quad.z_t)
        acc = rzero(f(μ_T_p, μ_q_p))
        @inbounds for k in 1:Nz
            ζ = (H / FT(2)) * quad.z_t[k]
            wk = quad.z_w[k] / FT(2)
            sμ_T = _pick_half_slope(ζ, dT_dz_dn, dT_dz_up)
            sμ_q = _pick_half_slope(ζ, dq_dz_dn, dq_dz_up)
            sσT = _pick_half_slope(ζ, dTT_dz_dn, dTT_dz_up)
            sσq = _pick_half_slope(ζ, dqq_dz_dn, dqq_dz_up)
            μ_Tk = μ_T_p + ζ * sμ_T
            μ_qk = μ_q_p + ζ * sμ_q
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
    # (cyclic Latin-style stratification; §3.2 of variance_adjustments README).
    # ------------------------------------------------------------------
    if dist isa GaussianGridscaleCorrectedSGS{SubgridLatinHypercubeZ} ||
       dist isa LogNormalGridscaleCorrectedSGS{SubgridLatinHypercubeZ}
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
                sμ_q = _pick_half_slope(ζ, dq_dz_dn, dq_dz_up)
                sσT = _pick_half_slope(ζ, dTT_dz_dn, dTT_dz_up)
                sσq = _pick_half_slope(ζ, dqq_dz_dn, dqq_dz_up)
                μ_Tk = μ_T_p + ζ * sμ_T
                μ_qk = μ_q_p + ζ * sμ_q
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
    # Principal axis: N depth nodes × 1D Hermite on dominant covariance axis (§3.5).
    # ------------------------------------------------------------------
    if dist isa GaussianGridscaleCorrectedSGS{SubgridPrincipalAxisLayer} ||
       dist isa LogNormalGridscaleCorrectedSGS{SubgridPrincipalAxisLayer}
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
            sμ_q = _pick_half_slope(ζ, dq_dz_dn, dq_dz_up)
            sσT = _pick_half_slope(ζ, dTT_dz_dn, dTT_dz_up)
            sσq = _pick_half_slope(ζ, dqq_dz_dn, dqq_dz_up)
            μ_Tk = μ_T_p + ζ * sμ_T
            μ_qk = μ_q_p + ζ * sμ_q
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
                    innerD, μ_q_p, μ_T_p, δq, δT, σ_qk, σ_Tk,
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
    if dist isa GaussianGridscaleCorrectedSGS{SubgridVoronoiRepresentatives} ||
       dist isa LogNormalGridscaleCorrectedSGS{SubgridVoronoiRepresentatives}
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
            sμ_q = _pick_half_slope(ζ, dq_dz_dn, dq_dz_up)
            sσT = _pick_half_slope(ζ, dTT_dz_dn, dTT_dz_up)
            sσq = _pick_half_slope(ζ, dqq_dz_dn, dqq_dz_up)
            μ_Tk = μ_T_p + ζ * sμ_T
            μ_qk = μ_q_p + ζ * sμ_q
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
    if dist isa GaussianGridscaleCorrectedSGS{SubgridBarycentricSeeds} ||
       dist isa LogNormalGridscaleCorrectedSGS{SubgridBarycentricSeeds}
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
            sμ_q = _pick_half_slope(ζ, dq_dz_dn, dq_dz_up)
            sσT = _pick_half_slope(ζ, dTT_dz_dn, dTT_dz_up)
            sσq = _pick_half_slope(ζ, dqq_dz_dn, dqq_dz_up)
            μ_Tk = μ_T_p + ζ * sμ_T
            μ_qk = μ_q_p + ζ * sμ_q
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

    # ------------------------------------------------------------------
    # Scheme III: Profile–Rosenblatt (linear μ_{u|v}(z) along the half;
    # face-anchored σ_{u|v} per half in the inner mixture).
    # ------------------------------------------------------------------
    if !(dist isa GaussianGridscaleCorrectedSGS{<:SubgridProfileRosenblatt} ||
         dist isa LogNormalGridscaleCorrectedSGS{<:SubgridProfileRosenblatt})
        error("integrate_over_sgs_linear_profile: unsupported distribution $(typeof(dist)).")
    end
    Sparam = typeof(dist).parameters[1]
    Btype = Sparam.parameters[1]
    method = Btype()
    function _profile_rosenblatt_accumulate(params)
        params === nothing && return nothing
        (;
            M_inv, s_v,
            L_dn, L_up,
            s_u_cond_c, s_u_cond_dn, s_u_cond_up,
            r_c,
        ) = params
        ε = ϵ_numerics(FT)
        N = quadrature_order(quad)
        p_nodes, p_w = gauss_legendre_01(FT, N)
        χ = quad.a
        wgh = quad.w
        inv_sqrt_pi = one(FT) / sqrt(FT(π))
        acc = rzero(f(μ_T_p, μ_q_p))
        sqrt2 = sqrt(FT(2))
        @inbounds for j in 1:N
            vj = sqrt2 * s_v * χ[j]
            wvj = wgh[j] * inv_sqrt_pi
            μ_0 = r_c * vj
            @inbounds for i in 1:N
                pi = p_nodes[i]
                wi = p_w[i]
                if (L_dn + L_up) <= ε &&
                   max(s_u_cond_c, s_u_cond_dn, s_u_cond_up) <= ε
                    ui = μ_0
                else
                    # Face-anchored conditional std per half (constant σ on each half in `u`).
                    ui =
                        _mixture_quantile_u(
                            method, pi, L_dn, s_u_cond_dn, L_up, s_u_cond_up,
                        ) + μ_0
                end
                δvec = M_inv * SA.SVector(ui, vj)
                δT, δq = δvec[1], δvec[2]
                T_hat, q_hat = _physical_Tq_from_fluctuations(
                    innerD, μ_q_p, μ_T_p, δq, δT,
                    σ_q_c, σ_T_c, oftype(μ_T_p, ρ_c),
                    quad.T_min, quad.q_max,
                )
                acc = acc ⊞ f(T_hat, q_hat) ⊠ wvj ⊠ wi
            end
        end
        return acc
    end
    p_dn = _two_slope_rosenblatt_params(
        μ_T_p, μ_q_p,
        oftype(μ_T_p, T′T′), oftype(μ_T_p, q′q′), oftype(μ_T_p, ρ_c),
        dT_dz_dn, dT_dz_up, dq_dz_dn, dq_dz_up,
        dTT_dz_dn, dTT_dz_up, dqq_dz_dn, dqq_dz_up,
        oftype(μ_T_p, H);
        transport = :dn,
    )
    p_up = _two_slope_rosenblatt_params(
        μ_T_p, μ_q_p,
        oftype(μ_T_p, T′T′), oftype(μ_T_p, q′q′), oftype(μ_T_p, ρ_c),
        dT_dz_dn, dT_dz_up, dq_dz_dn, dq_dz_up,
        dTT_dz_dn, dTT_dz_up, dqq_dz_dn, dqq_dz_up,
        oftype(μ_T_p, H);
        transport = :up,
    )
    a_dn = _profile_rosenblatt_accumulate(p_dn)
    a_up = _profile_rosenblatt_accumulate(p_up)
    if a_dn === nothing && a_up === nothing
        return f(μ_T_p, μ_q_p)
    elseif a_dn === nothing
        return a_up
    elseif a_up === nothing
        return a_dn
    else
        return a_dn ⊠ FT(0.5) ⊞ a_up ⊠ FT(0.5)
    end
end
