using RootSolvers
import SpecialFunctions as SF

ϵ_numerics(::Type{FT}) where {FT} = eps(FT)

function _normal_mills_integral(y::FT) where {FT}
    return y * 0.5 * (1.0 + SF.erf(y / sqrt(2.0))) + exp(-y^2 / 2.0) / sqrt(2.0 * π)
end

function uniform_gaussian_convolution_cdf(x::FT, L::FT, s::FT) where {FT}
    ε = ϵ_numerics(FT)
    sp = max(s, ε)
    Lp = max(L, ε)
    u1 = (x + Lp / 2.0) / sp
    u2 = (x - Lp / 2.0) / sp
    return (sp / Lp) * (_normal_mills_integral(u1) - _normal_mills_integral(u2))
end

function centered_uniform_gaussian_convolution_quantile_brent(p::FT, L::FT, s::FT) where {FT}
    ε = ϵ_numerics(FT)
    Lp = max(L, ε)
    sp = max(s, ε)
    smax = sp
    lo = -Lp / 2.0 - 6.0 * smax
    hi = Lp / 2.0 + 6.0 * smax
    f = x -> uniform_gaussian_convolution_cdf(x, Lp, sp) - p
    sol = RootSolvers.find_zero(f, RootSolvers.BrentsMethod{FT}(lo, hi), RootSolvers.CompactSolution())
    return sol.root
end

function get_q_hat(μ_q, σ_q, δq)
    # The normal ClimaAtmos reverse-engineering step
    χ1 = δq / (sqrt(2.0) * σ_q)
    
    # UNCLAMPED LOGNORMAL
    ratio = σ_q / μ_q
    σ_ln = sqrt(log(1.0 + ratio^2))
    μ_ln = log(μ_q) - σ_ln^2 / 2.0
    q_unclamped = exp(μ_ln + sqrt(2.0) * σ_ln * χ1)
    
    # CLAMPED LOGNORMAL (z_q clamped to [-5, 5])
    z_q = clamp(χ1, -5.0, 5.0)
    q_clamped = exp(μ_ln + sqrt(2.0) * σ_ln * z_q)
    
    # GAUSSIAN (for reference)
    q_gaussian = μ_q + sqrt(2.0) * σ_q * χ1
    
    return q_unclamped, q_clamped, q_gaussian
end

function test_scenario()
    FT = Float64
    μ_q = 1.0e-4
    σ_q = 1.0e-6
    L = 1.0e-3
    p = 0.8873
    
    u = centered_uniform_gaussian_convolution_quantile_brent(p, L, σ_q)
    δq = u
    
    q_unclamped, q_clamped, q_gaussian = get_q_hat(μ_q, σ_q, δq)
    
    println("μ_q = ", μ_q)
    println("L (cell thickness) = ", L)
    println("q_gaussian (linear mapped) = ", q_gaussian)
    println("q_unclamped (current code) = ", q_unclamped)
    println("q_clamped (with z_q in [-5, 5]) = ", q_clamped)
end

test_scenario()
