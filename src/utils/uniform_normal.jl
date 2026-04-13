# =============================================================================
# Optional Distributions.jl-style definitions (not loaded by default from any module).
#
#   using Distributions, SpecialFunctions, Random, LinearAlgebra
#   include("uniform_normal.jl")
#
# Contents:
#   • `UniformNormalConv` — additive convolution `Uniform(a,b) ⊗ Normal(0,σ)`.
#   • `BivariateUniformNormalSegment` — **X** = **a** + U(**b**−**a**) + **Z** with
#     U ~ Uniform(0,1), **Z** ~ MvNormal(**0**,**Σ**), 2×2 **Σ** ≻ 0; PDF via Cholesky
#     whitening of **Σ** (parallel uniform–normal × perpendicular Gaussian in warped space).
#   • `BivariateUniformNormalIsotropic` — independent `UniformNormalConv` on two coordinates,
#     common `σ` (turbulent covariance **Σ** = σ² **I**; axis-aligned structural box).
#
# API: [multivariate](https://juliastats.org/Distributions.jl/stable/multivariate/),
# [extends](https://juliastats.org/Distributions.jl/stable/extends/).
# =============================================================================

using Distributions
using LinearAlgebra: Cholesky, Diagonal, Symmetric, cholesky, det, dot, norm
using Random: Random, AbstractRNG, rand, randn
using SpecialFunctions: erf

# --- internal kernels ---

@inline function _ϵ_variance(FT)
    # floor for σ in denominators (same scale as ~ cbrt(floatmin))
    cbrt(floatmin(FT))
end

"""PDF at `x` of `Uniform(a,b) ⊗ Normal(0,σ)` (univariate convolution)."""
function uniform_normal_convolution_pdf(x::Real, a::Real, b::Real, σ::Real)
    FT = promote_type(typeof(x), typeof(a), typeof(b), typeof(σ), Float64)
    x = FT(x)
    a = FT(a)
    b = FT(b)
    σ = FT(σ)
    πf = FT(π)
    σp = max(σ, _ϵ_variance(FT))
    s2 = σp * sqrt(FT(2))
    if a == b
        return (one(FT) / (σp * sqrt(FT(2) * πf))) * exp(-((x - a) / σp)^2 / 2)
    end
    return (one(FT) / (2 * (b - a))) * (erf((x - a) / s2) - erf((x - b) / s2))
end

"""CDF of `Uniform(a,b) ⊗ Normal(0,σ)` via ∫ Φ((x−y)/σ) dy / (b−a)."""
function uniform_normal_convolution_cdf(x::Real, a::Real, b::Real, σ::Real)
    FT = promote_type(typeof(x), typeof(a), typeof(b), typeof(σ), Float64)
    x = FT(x)
    a = FT(a)
    b = FT(b)
    σ = FT(σ)
    if a == b
        return cdf(Normal(a, σ), x)
    end
    σp = max(σ, _ϵ_variance(FT))
    α = (x - b) / σp
    β = (x - a) / σp
    Φ(u) = cdf(Normal(), u)
    φ(u) = pdf(Normal(), u)
    F(u) = u * Φ(u) + φ(u)
    return (σp / (b - a)) * (F(β) - F(α))
end

function _quantile_bisection(d, p::Real; maxiter = 200, tol = 1e-12)
    !(0 ≤ p ≤ 1) && throw(ArgumentError("p must be in [0,1], got $p"))
    p == 0 && return -Inf
    p == 1 && return Inf
    μ = (d.a + d.b) / 2
    span = max(d.σ, abs(d.b - d.a)) * 12 + 6 * d.σ
    lo, hi = μ - span, μ + span
    # expand until CDF brackets p
    for _ in 1:80
        cdf(d, lo) ≤ p && cdf(d, hi) ≥ p && break
        span *= 2
        lo, hi = μ - span, μ + span
    end
    flo, fhi = cdf(d, lo), cdf(d, hi)
    flo > p && error("quantile: could not bracket lower tail (try wider parameters)")
    fhi < p && error("quantile: could not bracket upper tail (try wider parameters)")
    for _ in 1:maxiter
        mid = (lo + hi) / 2
        cm = cdf(d, mid)
        if cm < p
            lo = mid
        else
            hi = mid
        end
        hi - lo < tol * (1 + abs(mid)) && return mid
    end
    return (lo + hi) / 2
end

# --- univariate distribution ---

"""
    UniformNormalConv{T}

Convolution of `Uniform(a,b)` with `Normal(0,σ)` (additive).

Implements the [univariate checklist](https://juliastats.org/Distributions.jl/stable/extends/)
(`rand`, `sampler`, `logpdf`, `cdf`, `quantile`, `minimum`, `maximum`, `insupport`)
plus `mean`, `var`, `std`.
"""
struct UniformNormalConv{T <: Real} <: ContinuousUnivariateDistribution
    a::T
    b::T
    σ::T
end

function UniformNormalConv(a::Real, b::Real, σ::Real)
    T = promote_type(typeof(a), typeof(b), typeof(σ), Float64)
    UniformNormalConv{T}(T(a), T(b), T(σ))
end

Base.eltype(::Type{<:UniformNormalConv{T}}) where {T} = T

Distributions.sampler(d::UniformNormalConv) = d

function Random.rand(rng::AbstractRNG, d::UniformNormalConv)
    T = eltype(d)
    u = d.a + (d.b - d.a) * rand(rng, T)
    return u + d.σ * randn(rng, T)
end

Distributions.pdf(d::UniformNormalConv, x::Real) =
    uniform_normal_convolution_pdf(x, d.a, d.b, d.σ)

function Distributions.logpdf(d::UniformNormalConv, x::Real)
    p = pdf(d, x)
    return p > zero(p) ? log(p) : oftype(p, -Inf)
end

Distributions.cdf(d::UniformNormalConv, x::Real) =
    uniform_normal_convolution_cdf(x, d.a, d.b, d.σ)

Distributions.quantile(d::UniformNormalConv, p::Real) = _quantile_bisection(d, p)

Distributions.minimum(::UniformNormalConv) = -Inf
Distributions.maximum(::UniformNormalConv) = Inf
Distributions.insupport(d::UniformNormalConv, x::Real) = !isnan(x) && isfinite(x)

function Distributions.mean(d::UniformNormalConv)
    d.a == d.b && return d.a
    (d.a + d.b) / 2
end

function Distributions.var(d::UniformNormalConv)
    d.a == d.b && return d.σ^2
    (d.b - d.a)^2 / 12 + d.σ^2
end

Distributions.std(d::UniformNormalConv) = sqrt(var(d))

# Unimodal symmetric-ish case: use mean as a cheap `mode` surrogate (see doc note above).
Distributions.mode(d::UniformNormalConv) = mean(d)
Distributions.modes(d::UniformNormalConv) = [mode(d)]

# --- bivariate: segment + MvNormal noise (full 2×2 Σ) ---

@inline function _segment_degenerate_tol(a::AbstractVector, b::AbstractVector, ::Type{T}) where {T <: Real}
    scale = one(T) + norm(a) + norm(b) + norm(b .- a)
    return max(sqrt(eps(T)), T(1e-14)) * scale
end

function _bivariate_uniform_normal_segment_pdf_impl(
    x::AbstractVector{T},
    a::AbstractVector{T},
    b::AbstractVector{T},
    Σm::AbstractMatrix{T},
    F::Cholesky{T, Matrix{T}},
) where {T <: Real}
    dphys = b .- a
    if norm(dphys) < _segment_degenerate_tol(a, b, T)
        return pdf(MvNormal(Vector(a), Σm), Vector(x))
    end
    # x̃ = L⁻¹ x with Σ = L Lᵀ (lower triangular L)
    x̃ = F.L \ x
    ã = F.L \ a
    b̃ = F.L \ b
    ṽ = x̃ .- ã
    d̃ = b̃ .- ã
    L̃ = norm(d̃)
    if L̃ < _segment_degenerate_tol(ã, b̃, T)
        return pdf(MvNormal(Vector(a), Σm), Vector(x))
    end
    p = dot(ṽ, d̃) / L̃
    d_perp_sq = max(zero(T), dot(ṽ, ṽ) - p^2)
    f_perp = (one(T) / sqrt(T(2) * T(π))) * exp(-d_perp_sq / 2)
    f_parallel = uniform_normal_convolution_pdf(p, zero(T), L̃, one(T))
    f_tilde = f_perp * f_parallel
    return f_tilde / sqrt(det(F))
end

"""
    bivariate_uniform_normal_segment_pdf(x, a, b, Σ) -> Real

PDF at `x ∈ ℝ²` for **X** = **a** + U(**b**−**a**) + **Z** with U ~ Uniform(0,1),
**Z** ~ N(0, **Σ**), U ⟂ **Z**, and **Σ** positive definite (2×2). Evaluated by Cholesky
whitening: in warped space the noise is standard isotropic; density is a product of a
univariate uniform–normal convolution along the segment and a Gaussian perpendicular to it,
then scaled by `1 / sqrt(det(Σ))`. Collapses to `pdf(MvNormal(a,Σ), x)` when **a** ≈ **b**.
"""
function bivariate_uniform_normal_segment_pdf(
    x::AbstractVector{<:Real},
    a::AbstractVector{<:Real},
    b::AbstractVector{<:Real},
    Σ::AbstractMatrix{<:Real},
)
    length(x) == 2 && length(a) == 2 && length(b) == 2 ||
        throw(DimensionMismatch("x, a, b must have length 2"))
    size(Σ) == (2, 2) || throw(DimensionMismatch("Σ must be 2×2"))
    T = promote_type(eltype(x), eltype(a), eltype(b), eltype(Σ), Float64)
    x = convert.(T, x)
    a = convert.(T, a)
    b = convert.(T, b)
    Σm = convert(Matrix{T}, Σ)
    Σs = Symmetric(Σm)
    isposdef(Σs) || throw(ArgumentError("Σ must be positive definite"))
    F = cholesky(Σs)
    return _bivariate_uniform_normal_segment_pdf_impl(x, a, b, Σm, F)
end

"""
    BivariateUniformNormalSegment{T}

Bivariate distribution **X** = **a** + U(**b**−**a**) + **Z** with U ~ Uniform(0,1),
**Z** ~ N(0, **Σ**), independent; **Σ** is 2×2 positive definite. Implements
[`ContinuousMultivariateDistribution`](https://juliastats.org/Distributions.jl/stable/multivariate/)
(`length`, `_rand!`, `_logpdf`, `mean`, `var`, `cov`, `std`).
"""
struct BivariateUniformNormalSegment{T <: Real} <: ContinuousMultivariateDistribution
    a::Vector{T}
    b::Vector{T}
    Σ::Matrix{T}
    chol::Cholesky{T, Matrix{T}}
end

function BivariateUniformNormalSegment(
    a::AbstractVector{<:Real},
    b::AbstractVector{<:Real},
    Σ::AbstractMatrix{<:Real},
)
    length(a) == 2 && length(b) == 2 || throw(DimensionMismatch("endpoints a, b must have length 2"))
    size(Σ) == (2, 2) || throw(DimensionMismatch("Σ must be 2×2"))
    T = promote_type(eltype(a), eltype(b), eltype(Σ), Float64)
    a2 = convert.(T, a)
    b2 = convert.(T, b)
    Σm = convert(Matrix{T}, Σ)
    Σstored = copy(Σm)
    Σs = Symmetric(Σstored)
    isposdef(Σs) || throw(ArgumentError("Σ must be positive definite"))
    F = cholesky(Σs)
    BivariateUniformNormalSegment{T}(a2, b2, Σstored, F)
end

Base.length(::BivariateUniformNormalSegment) = 2
Base.size(d::BivariateUniformNormalSegment) = (length(d),)

Base.eltype(::Type{<:BivariateUniformNormalSegment{T}}) where {T} = T

Distributions.sampler(d::BivariateUniformNormalSegment) = d

function Distributions._rand!(
    rng::AbstractRNG,
    d::BivariateUniformNormalSegment,
    x::AbstractVector{<:Real},
)
    length(x) == 2 || throw(DimensionMismatch("need length-2 vector"))
    T = eltype(d)
    u = rand(rng, T)
    z = rand(rng, MvNormal(zeros(T, 2), d.Σ))
    x[1] = d.a[1] + u * (d.b[1] - d.a[1]) + z[1]
    x[2] = d.a[2] + u * (d.b[2] - d.a[2]) + z[2]
    return x
end

function Distributions._logpdf(d::BivariateUniformNormalSegment, x::AbstractVector{<:Real})
    length(x) == 2 || throw(DimensionMismatch("need length-2 vector"))
    T = eltype(d)
    xv = convert.(T, x)
    p = _bivariate_uniform_normal_segment_pdf_impl(xv, d.a, d.b, d.Σ, d.chol)
    return p > zero(p) ? log(p) : oftype(p, -Inf)
end

Distributions.mean(d::BivariateUniformNormalSegment) = (d.a .+ d.b) ./ 2

function Distributions.cov(d::BivariateUniformNormalSegment)
    T = eltype(d)
    δ = d.b .- d.a
    outer = δ * δ'
    return T(1 / 12) .* outer .+ d.Σ
end

function Distributions.var(d::BivariateUniformNormalSegment)
    c = cov(d)
    [c[1, 1], c[2, 2]]
end

Distributions.std(d::BivariateUniformNormalSegment) = sqrt.(var(d))

Distributions.insupport(::BivariateUniformNormalSegment, x::AbstractVector{<:Real}) =
    length(x) == 2 && all(isfinite, x)

# --- bivariate: independent marginals, σ² I ---

"""
    BivariateUniformNormalIsotropic{T}

Product of two independent `UniformNormalConv` distributions with the same `σ`: an
**axis-aligned** structural rectangle in ℝ² and bivariate turbulent covariance
**Σ** = σ² **I** (diagonal). This is **not** the same random object as
`BivariateUniformNormalSegment`, which concentrates structural mass on the line segment
from **a** to **b** and allows a general 2×2 **Σ**.

Subtype of `ContinuousMultivariateDistribution`; implements the
[multivariate checklist](https://juliastats.org/Distributions.jl/stable/extends/)
(`length`, `sampler`, `eltype`, `_rand!`, `_logpdf`) plus `mean`, `var`, `cov`, `std`.
"""
struct BivariateUniformNormalIsotropic{T <: Real} <: ContinuousMultivariateDistribution
    μ_q::T
    μ_T::T
    δq_half::T
    δT_half::T
    σ::T
    mq::UniformNormalConv{T}
    mT::UniformNormalConv{T}
end

function BivariateUniformNormalIsotropic(μ_q::Real, μ_T::Real, δq_half::Real, δT_half::Real, σ::Real)
    T = promote_type(typeof(μ_q), typeof(μ_T), typeof(δq_half), typeof(δT_half), typeof(σ), Float64)
    μ_q, μ_T, δq_half, δT_half, σ = T(μ_q), T(μ_T), T(δq_half), T(δT_half), T(σ)
    mq = UniformNormalConv(μ_q - δq_half, μ_q + δq_half, σ)
    mT = UniformNormalConv(μ_T - δT_half, μ_T + δT_half, σ)
    BivariateUniformNormalIsotropic{T}(μ_q, μ_T, δq_half, δT_half, σ, mq, mT)
end

Base.length(::BivariateUniformNormalIsotropic) = 2
Base.size(d::BivariateUniformNormalIsotropic) = (length(d),)

Base.eltype(::Type{<:BivariateUniformNormalIsotropic{T}}) where {T} = T

Distributions.sampler(d::BivariateUniformNormalIsotropic) = d

function Distributions._rand!(
    rng::AbstractRNG,
    d::BivariateUniformNormalIsotropic,
    x::AbstractVector{<:Real},
)
    length(x) == 2 || throw(DimensionMismatch("need length-2 vector"))
    x[1] = rand(rng, d.mq)
    x[2] = rand(rng, d.mT)
    return x
end

function Distributions._logpdf(d::BivariateUniformNormalIsotropic, x::AbstractVector{<:Real})
    length(x) == 2 || throw(DimensionMismatch("need length-2 vector"))
    return logpdf(d.mq, x[1]) + logpdf(d.mT, x[2])
end

Distributions.mean(d::BivariateUniformNormalIsotropic) = [mean(d.mq), mean(d.mT)]

function Distributions.var(d::BivariateUniformNormalIsotropic)
    v1 = var(d.mq)
    v2 = var(d.mT)
    [v1, v2]
end

function Distributions.cov(d::BivariateUniformNormalIsotropic)
    Diagonal(var(d))
end

Distributions.std(d::BivariateUniformNormalIsotropic) = sqrt.(var(d))

Distributions.insupport(::BivariateUniformNormalIsotropic, x::AbstractVector{<:Real}) =
    length(x) == 2 && all(isfinite, x)

"""Joint PDF at `(q, T)` as `pdf(d.mq, q) * pdf(d.mT, T)` (product of marginals)."""
function bivariate_uniform_normal_isotropic_pdf(
    q::Real,
    T_::Real,
    d::BivariateUniformNormalIsotropic,
)
    return pdf(d.mq, q) * pdf(d.mT, T_)
end
