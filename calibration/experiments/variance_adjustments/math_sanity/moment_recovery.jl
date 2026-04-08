using LinearAlgebra
using Statistics

include(joinpath(@__DIR__, "gauss_hermite_tables.jl"))
include(joinpath(@__DIR__, "transforms.jl"))

function mathsanity_weighted_mean_cov(xs::AbstractVector{FT}, ys::AbstractVector{FT}, ω::AbstractVector{FT}) where {FT}
    s = sum(ω)
    tol = max(sqrt(eps(FT)), FT(1e-12))
    @assert s ≈ 1 rtol = tol atol = tol
    mx = sum(ω .* xs)
    my = sum(ω .* ys)
    cxx = sum(ω .* (xs .- mx) .^ 2)
    cyy = sum(ω .* (ys .- my) .^ 2)
    cxy = sum(ω .* (xs .- mx) .* (ys .- my))
    Σ = [cxx cxy; cxy cyy]
    return (mx, my), Σ
end

"""Product Gauss–Hermite expectation for (Z₁,Z₂) iid N(0,1) mapped to correlated (q,T)."""
function mathsanity_quadrature_moments_bivariate(μ_q::FT, μ_T::FT, σ_q::FT, σ_T::FT, ρ::FT, N::Int; clamped::Bool = false) where {FT}
    χ1, ω1 = mathsanity_std_normal_expectation_weights(FT, N)
    χ2, ω2 = mathsanity_std_normal_expectation_weights(FT, N)
    T_min = FT(0.0)
    q_max = FT(1.0)
    qs = FT[]
    Ts = FT[]
    ωs = FT[]
    for (i, c1) in enumerate(χ1), (j, c2) in enumerate(χ2)
        ωij = ω1[i] * ω2[j]
        q, T = if clamped
            mathsanity_gaussian_sgs_clamped(c1, c2, μ_q, μ_T, σ_q, σ_T, ρ, T_min, q_max)
        else
            mathsanity_gaussian_sgs_unclamped(c1, c2, μ_q, μ_T, σ_q, σ_T, ρ)
        end
        push!(qs, q)
        push!(Ts, T)
        push!(ωs, ωij)
    end
    return qs, Ts, ωs
end

"""Univariate X = μ + σ Z, Z ~ N(0,1); quadrature should recover μ and σ² exactly (polynomial moments up to quadrature degree)."""
function mathsanity_univariate_mean_var(μ::FT, σ::FT, N::Int) where {FT}
    χ, ω = mathsanity_std_normal_expectation_weights(FT, N)
    z = mathsanity_z_from_χ.(χ)
    x = μ .+ σ .* z
    m1 = sum(ω .* x)
    v = sum(ω .* (x .- m1) .^ 2)
    return m1, v
end

function mathsanity_analytic_mean_cov(μ_q::FT, μ_T::FT, σ_q::FT, σ_T::FT, ρ::FT) where {FT} 
    μ = [μ_q, μ_T]
    Σ = [σ_q^2 ρ*σ_q*σ_T; ρ*σ_q*σ_T σ_T^2]
    return μ, Σ
end

"""Finite-difference check: d(σ²)/dσ = 2σ using the same GH weights as ClimaAtmos."""
function mathsanity_d_variance_d_sigma(μ::FT, σ0::FT, N::Int; h = 1e-7) where {FT}
    χ, ω = mathsanity_std_normal_expectation_weights(FT, N)
    function quad_var(σ::FT) where {FT}
        z = mathsanity_z_from_χ.(χ)
        x = μ .+ σ .* z
        m = sum(ω .* x)
        return sum(ω .* (x .- m) .^ 2)
    end
    d_fd = (quad_var(σ0 + h) - quad_var(σ0 - h)) / (2h)
    return d_fd, 2σ0
end

"""Scan σ used in quadrature vs σ_true (same analytic target); shows moment error from wrong spread."""
function mathsanity_sigma_ratio_scan(μ_q::FT, μ_T::FT, σ_q_true::FT, σ_T_true::FT, ρ::FT, N::Int, ratios::AbstractVector{FT}) where {FT}
    μ, Σ_true = mathsanity_analytic_mean_cov(μ_q, μ_T, σ_q_true, σ_T_true, ρ)
    rows = Tuple{FT,FT,FT,FT,FT}[]
    for r in ratios
        σq = r * σ_q_true
        σT = r * σ_T_true
        qs, Ts, ωs = mathsanity_quadrature_moments_bivariate(μ_q, μ_T, σq, σT, ρ, N; clamped = false)
        (mq, mT), Σq = mathsanity_weighted_mean_cov(qs, Ts, ωs)
        tol = max(sqrt(eps(FT)), FT(1e-12))
        ferr = norm([mq, mT] - μ) / (norm(μ) + tol)
        serr = opnorm(Σq - Σ_true) / (opnorm(Σ_true) + tol)
        push!(rows, (r, mq, mT, ferr, serr))
    end
    return rows
end
