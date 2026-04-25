using RootSolvers
using StaticArrays
import SpecialFunctions as SF

ϵ_numerics(::Type{FT}) where {FT} = eps(FT)

@inline function _std_normal_cdf(y::FT) where {FT}
    return FT(0.5) * (one(FT) + SF.erf(y / sqrt(FT(2))))
end

@inline function _std_normal_pdf(y::FT) where {FT}
    return exp(-y^2 / FT(2)) / sqrt(FT(2) * FT(π))
end

@inline function _normal_mills_integral(y::FT) where {FT}
    return y * _std_normal_cdf(y) + _std_normal_pdf(y)
end

@inline function uniform_gaussian_convolution_cdf(x::FT, L::FT, s::FT) where {FT}
    ε = ϵ_numerics(FT)
    sp = max(s, ε)
    Lp = max(L, ε)
    u1 = (x + Lp / FT(2)) / sp
    u2 = (x - Lp / FT(2)) / sp
    return (sp / Lp) * (_normal_mills_integral(u1) - _normal_mills_integral(u2))
end

function centered_uniform_gaussian_convolution_quantile_brent(p::FT, L::FT, s::FT) where {FT}
    ε = ϵ_numerics(FT)
    Lp = max(L, ε)
    sp = max(s, ε)
    smax = sp
    lo = -Lp / FT(2) - FT(6) * smax
    hi = Lp / FT(2) + FT(6) * smax
    f = x -> uniform_gaussian_convolution_cdf(x, Lp, sp) - p
    sol = RootSolvers.find_zero(f, RootSolvers.BrentsMethod{FT}(lo, hi), RootSolvers.CompactSolution())
    return sol.root
end

function test_scenario()
    FT = Float64
    μ_q = 1.0e-3
    σ_q = 1.0e-6
    L = 1.0e-3
    p = 0.8873
    
    u = centered_uniform_gaussian_convolution_quantile_brent(p, L, σ_q)
    println("u = ", u)
    
    χ1 = u / (sqrt(2.0) * σ_q)
    println("χ1 = ", χ1)
    
    ratio = σ_q / μ_q
    σ_ln = sqrt(log(1.0 + ratio^2))
    μ_ln = log(μ_q) - σ_ln^2 / 2
    println("σ_ln = ", σ_ln)
    
    q_lognormal = exp(μ_ln + sqrt(2.0) * σ_ln * χ1)
    println("q_lognormal = ", q_lognormal)
    
    println("μ_q + u = ", μ_q + u)
end

test_scenario()
