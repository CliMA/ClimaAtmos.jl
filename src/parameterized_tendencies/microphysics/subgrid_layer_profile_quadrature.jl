# Linear layer-mean profile quadrature for gridscale-corrected SGS (column-tensor and profile–Rosenblatt).

import ClimaCore.Geometry as Geometry
import RootSolvers as RS
import SpecialFunctions as SF
import StaticArrays as SA

"""
    layer_profile_dq_dT_dz(grad_q, grad_θ, ∂T∂θ_li, local_geometry)

Physical vertical derivatives `∂q/∂z` and `∂T/∂z ≈ (∂T/∂θ_li)(∂θ_li/∂z)` from precomputed
`ᶜgradᵥ` fields. Uses `WVector(·, local_geometry)[1]` — **not** raw `components(grad)[end]`,
which is a covariant component and can differ from ∂/∂z in SI by metric factors (O(10³) in tests).
"""
@inline function layer_profile_dq_dT_dz(grad_q, grad_θ, ∂T∂θ_li, local_geometry)
    dq_dz = Geometry.WVector(grad_q, local_geometry)[1]
    dθ_dz = Geometry.WVector(grad_θ, local_geometry)[1]
    dT_dz = ∂T∂θ_li * dθ_dz
    return dq_dz, dT_dz
end

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

"""CDF of uniform–Gaussian convolution on `ℝ` with Gaussian s.d. `s` and uniform support `[-L/2,L/2]`."""
@inline function uniform_gaussian_convolution_cdf(x::FT, L::FT, s::FT) where {FT}
    ε = ϵ_numerics(FT)
    sp = max(s, ε)
    u1 = (x + L / FT(2)) / sp
    u2 = (x - L / FT(2)) / sp
    return (sp / L) * (_normal_mills_integral(u1) - _normal_mills_integral(u2))
end

@inline function uniform_gaussian_convolution_pdf(x::FT, L::FT, s::FT) where {FT}
    ε = ϵ_numerics(FT)
    sp = max(s, ε)
    s2 = sp * sqrt(FT(2))
    u1 = (x + L / FT(2)) / s2
    u2 = (x - L / FT(2)) / s2
    return (one(FT) / (FT(2) * L)) * (SF.erf(u1) - SF.erf(u2))
end

@inline function uniform_gaussian_convolution_pdf_prime(x::FT, L::FT, s::FT) where {FT}
    ε = ϵ_numerics(FT)
    sp = max(s, ε)
    u1 = (x + L / FT(2)) / sp
    u2 = (x - L / FT(2)) / sp
    c = one(FT) / (L * sp * sqrt(FT(2) * FT(π)))
    return c * (exp(-u1^2 / FT(2)) - exp(-u2^2 / FT(2)))
end

function _convolution_quantile_bracketed(p::FT, L::FT, s::FT) where {FT}
    ε = ϵ_numerics(FT)
    sp = max(s, ε)
    lo = -(L / FT(2) + FT(6) * sp)
    hi = L / FT(2) + FT(6) * sp
    f = x -> uniform_gaussian_convolution_cdf(x, L, s) - p
    sol = RS.find_zero(
        f,
        RS.BrentsMethod{FT}(lo, hi),
        RS.CompactSolution(),
    )
    return sol.root
end

"""
    _convolution_quantile_halley(p, L, s)

**One** Halley step for `F(u) = p` on the uniform–Gaussian convolution CDF. Initial `u` is a Cornish–Fisher / uniform
blend in Gauss–Legendre probability `p` (with `L < 1e-12` → Gaussian quantile `s * √(2) * erfinv(2p-1)`). **Not** used
by the Chebyshev quantile path.
"""
function _convolution_quantile_halley(p::FT, L::FT, s::FT) where {FT}
    ε = ϵ_numerics(FT)
    t = FT(2) * p - one(FT)
    t = clamp(t, -one(FT) + FT(100) * ε, one(FT) - FT(100) * ε)
    z = sqrt(FT(2)) * SF.erfinv(t)
    Lthin = FT(1.0e-12)
    u = if L < Lthin
        s * z
    else
        η = s / max(L, ε)
        x_uni = t * (L / FT(2))
        var_u = s^2 + L^2 / FT(12)
        exkur = (-L^4 / FT(120)) / var_u^2
        x_cf = sqrt(var_u) * (z + (exkur / FT(24)) * (z^3 - FT(3) * z))
        W = η^2 / (η^2 + FT(0.002))
        (one(FT) - W) * x_uni + W * x_cf
    end
    Fv = uniform_gaussian_convolution_cdf(u, L, s)
    g = Fv - p
    fv = uniform_gaussian_convolution_pdf(u, L, s)
    abs(fv) < ε && return u
    fpv = uniform_gaussian_convolution_pdf_prime(u, L, s)
    denom = FT(2) * fv^2 - g * fpv
    # Stabilize division: |denom| floored at 1e-15, sign preserved.
    denom = copysign(max(abs(denom), FT(1.0e-15)), denom == zero(FT) ? one(FT) : denom)
    return u - FT(2) * fv * g / denom
end

"""Chebyshev surrogate in mapped `log10(η)`; errors if η or `(N_GL, i_node)` are out of range (no iterative fallback)."""
function _convolution_quantile_chebyshev_logeta(
    p::FT,
    L::FT,
    s::FT,
    N_GL::Int,
    i_node::Int,
) where {FT}
    ε = ϵ_numerics(FT)
    η = s / max(L, ε)
    if η <= ε
        error(
            "Chebyshev convolution quantile: η = s/L = $η is not above ϵ_numerics (~$ε), so log10(η) is undefined. " *
            "Use `ConvolutionQuantilesBracketed` (Brent) or `ConvolutionQuantilesHalley` (one Halley step).",
        )
    end
    c = chebyshev_convolution_coeffs(FT, N_GL, i_node)
    ℓ0 = log10(FT(1.0e-4))
    ℓ1 = log10(FT(1.0e2))
    ℓ = log10(η)
    τ = (FT(2) * (ℓ - ℓ0)) / (ℓ1 - ℓ0) - one(FT)
    τ = clamp(τ, -one(FT), one(FT))
    # Pure surrogate: coeffs from `convolution_quantile_chebyshev_tables.jl` (regenerate via
    # `calibration/experiments/variance_adjustments/scripts/gen_convolution_chebyshev_tables.jl`); no root iteration.
    u_over_L = chebyshev_evaluate(c, τ)
    return L * u_over_L
end

function _quantile_u(
    ::ConvolutionQuantilesBracketed,
    p::FT,
    L::FT,
    s::FT,
    N_GL::Int,
    i::Int,
) where {FT}
    return _convolution_quantile_bracketed(p, L, s)
end

function _quantile_u(
    ::ConvolutionQuantilesHalley,
    p::FT,
    L::FT,
    s::FT,
    N_GL::Int,
    i::Int,
) where {FT}
    return _convolution_quantile_halley(p, L, s)
end

function _quantile_u(
    ::ConvolutionQuantilesChebyshevLogEta,
    p::FT,
    L::FT,
    s::FT,
    N_GL::Int,
    i::Int,
) where {FT}
    return _convolution_quantile_chebyshev_logeta(p, L, s, N_GL, i)
end

"""Build `Σ_uv = M Σ M'`; state order for `Σ` and `M^{-1}` columns is `[δT, δq]`."""
function profile_rosenblatt_covariance(
    σ_q::FT,
    σ_T::FT,
    ρ::FT,
    dT_dz::FT,
    dq_dz::FT,
) where {FT}
    ε = ϵ_numerics(FT)
    Σ = SA.@SMatrix [
        max(σ_T^2, zero(FT)) ρ*σ_T*σ_q
        ρ*σ_T*σ_q max(σ_q^2, zero(FT))
    ]
    α = dT_dz^2 + dq_dz^2
    if α <= ε^2
        return nothing
    end
    invα = one(FT) / α
    M = SA.@SMatrix [
        dT_dz*invα dq_dz*invα
        -dq_dz dT_dz
    ]
    Σuv = M * Σ * M'
    s1sq = Σuv[1, 1]
    s12 = Σuv[1, 2]
    s2sq = max(Σuv[2, 2], zero(FT))
    s_v = sqrt(s2sq)
    s2eff = max(s2sq, ε)
    s_u_cond = sqrt(max(s1sq - s12^2 / s2eff, zero(FT)))
    μ_u_slope = s12 / s2eff
    M_inv = SA.@SMatrix [
        dT_dz -dq_dz*invα
        dq_dz dT_dz*invα
    ]
    return (; M_inv, s_v, s_u_cond, μ_u_slope)
end

@inline function _inner_dist_linear_profile(::GaussianGridscaleCorrectedSGS)
    return GaussianSGS()
end
@inline function _inner_dist_linear_profile(::LogNormalGridscaleCorrectedSGS)
    return LogNormalSGS()
end

"""Recover Hermite nodes `(χ1, χ2)` that yield Gaussian fluctuations `(δT, δq)` with correlation `ρ`."""
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
    innerD::GaussianSGS,
    μ_q,
    μ_T,
    δq,
    δT,
    σ_q,
    σ_T,
    ρ,
    T_min,
    q_max,
)
    χ1, χ2 = _hermite_from_gaussian_fluctuations(δq, δT, σ_q, σ_T, ρ)
    return get_physical_point(
        GaussianSGS(),
        χ1,
        χ2,
        μ_q,
        μ_T,
        σ_q,
        σ_T,
        ρ,
        T_min,
        q_max,
    )
end

@inline function _physical_Tq_from_fluctuations(
    innerD::LogNormalSGS,
    μ_q,
    μ_T,
    δq,
    δT,
    σ_q,
    σ_T,
    ρ,
    T_min,
    q_max,
)
    χ1, χ2 = _hermite_from_gaussian_fluctuations(δq, δT, σ_q, σ_T, ρ)
    return get_physical_point(
        LogNormalSGS(),
        χ1,
        χ2,
        μ_q,
        μ_T,
        σ_q,
        σ_T,
        ρ,
        T_min,
        q_max,
    )
end

"""
    integrate_over_sgs_linear_profile(
        f, quad, μ_q, μ_T, q′q′, T′T′, corr_Tq,
        H, local_geometry, grad_q, grad_θ, ∂T∂θ_li,
    )

Layer-mean quadrature for `GaussianGridscaleCorrectedSGS{S}` / `LogNormalGridscaleCorrectedSGS{S}`
with `S` in (`SubgridColumnTensor`, `SubgridProfileRosenblatt`).

`f` is called as `f(T_hat, q_hat)` (same as [`integrate_over_sgs`](@ref)).
"""
function integrate_over_sgs_linear_profile(
    f,
    quad::SGSQuadrature,
    μ_q,
    μ_T,
    q′q′,
    T′T′,
    corr_Tq,
    H,
    local_geometry,
    grad_q,
    grad_θ,
    ∂T∂θ_li,
)
    dist = quad.dist
    σ_q, σ_T, ρ = sgs_stddevs_and_correlation(q′q′, T′T′, corr_Tq)
    dq_dz, dT_dz = layer_profile_dq_dT_dz(grad_q, grad_θ, ∂T∂θ_li, local_geometry)
    μ_T_p, μ_q_p = promote(μ_T, μ_q)
    FT = typeof(μ_T_p)
    innerD = _inner_dist_linear_profile(dist)

    if dist isa GaussianGridscaleCorrectedSGS{SubgridColumnTensor} ||
       dist isa LogNormalGridscaleCorrectedSGS{SubgridColumnTensor}
        quad.z_t === nothing && error("SGSQuadrature requires z-axis nodes for SubgridColumnTensor.")
        Nz = length(quad.z_t)
        acc = rzero(f(μ_T_p, μ_q_p))
        @inbounds for k in 1:Nz
            zk = (H / FT(2)) * quad.z_t[k]
            wk = quad.z_w[k] / FT(2)
            μ_Tk = μ_T_p + zk * dT_dz
            μ_qk = μ_q_p + zk * dq_dz
            transform = PhysicalPointTransform(
                innerD,
                μ_Tk,
                μ_qk,
                oftype(μ_T_p, σ_T),
                oftype(μ_T_p, σ_q),
                oftype(μ_T_p, ρ),
                oftype(μ_T_p, quad.T_min),
                oftype(μ_T_p, quad.q_max),
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

    if !(dist isa GaussianGridscaleCorrectedSGS{<:SubgridProfileRosenblatt} ||
         dist isa LogNormalGridscaleCorrectedSGS{<:SubgridProfileRosenblatt})
        error("integrate_over_sgs_linear_profile: unsupported distribution $(typeof(dist)).")
    end
    Sparam = typeof(dist).parameters[1]
    Btype = Sparam.parameters[1]
    method = Btype()

    covp = profile_rosenblatt_covariance(σ_q, σ_T, ρ, dT_dz, dq_dz)
    if covp === nothing
        return f(μ_T_p, μ_q_p)
    end
    (; M_inv, s_v, s_u_cond, μ_u_slope) = covp
    L = H
    ε = ϵ_numerics(FT)
    sp = s_u_cond

    N = quadrature_order(quad)
    χ = quad.a
    wgh = quad.w
    inv_sqrt_pi = one(FT) / sqrt(FT(π))
    p_nodes, p_w = gauss_legendre_01(FT, N)

    acc = rzero(f(μ_T_p, μ_q_p))
    sqrt2 = sqrt(FT(2))
    @inbounds for j in 1:N
        vj = sqrt2 * s_v * χ[j]
        wvj = wgh[j] * inv_sqrt_pi
        μ_u = μ_u_slope * vj
        @inbounds for i in 1:N
            pi = p_nodes[i]
            wi = p_w[i]
            if sp <= ε
                ui = L * (pi - FT(0.5))
            else
                ui = _quantile_u(method, pi, L, sp, N, i)
            end
            ui = ui + μ_u
            δvec = M_inv * SA.SVector(ui, vj)
            δT, δq = δvec[1], δvec[2]
            T_hat, q_hat = _physical_Tq_from_fluctuations(
                innerD,
                μ_q_p,
                μ_T_p,
                δq,
                δT,
                σ_q,
                σ_T,
                ρ,
                quad.T_min,
                quad.q_max,
            )
            acc = acc ⊞ f(T_hat, q_hat) ⊠ wvj ⊠ wi
        end
    end
    return acc
end
