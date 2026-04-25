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

function demo()
    FT = Float64
    ε = ϵ_numerics(FT)
    μ_q = 1.0e-4
    σ_q = 1.0e-6
    L_physical = 1.0e-3
    p = 0.8873
    
    # physical limits
    q_bottom = max(ε, μ_q - L_physical / 2.0)
    q_top = μ_q + L_physical / 2.0
    
    println("q_bottom: ", q_bottom)
    println("q_top: ", q_top)
    
    # define log space gradient so ends are constrained
    L_ln = log(q_top) - log(q_bottom)
    
    # log space turbulence
    ratio = σ_q / μ_q
    σ_ln = sqrt(log(1.0 + ratio^2))
    
    # The log-mean is roughly log(μ_q)
    # But wait, if the profile is log(q_bottom) to log(q_top), the center is:
    μ_ln_center = (log(q_top) + log(q_bottom)) / 2.0
    
    println("L_ln (constrained): ", L_ln)
    println("σ_ln: ", σ_ln)
    println("μ_ln_center: ", μ_ln_center)
    
    # Run Rosenblatt in log space with constrained gradient!
    δq_ln = centered_uniform_gaussian_convolution_quantile_brent(p, L_ln, σ_ln)
    
    # Evaluate physical q
    q_hat = exp(μ_ln_center + δq_ln)
    
    println("q_hat (constrained log gradient): ", q_hat)
    
    # Compare to local mean at quantile p
    p_uniform = (p - 0.5) # approximate relative position
    q_target = μ_q + 2.0 * p_uniform * L_physical
    println("Target physical q at p: ", q_target)
end

demo()
