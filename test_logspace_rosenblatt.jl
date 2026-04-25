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
    p = 0.8873  # Top edge quantile
    
    # --- Physical Space (Current Buggy Julia Implementation) ---
    δq_physical = centered_uniform_gaussian_convolution_quantile_brent(p, L_physical, σ_q)
    χ1_physical = δq_physical / (sqrt(2.0) * σ_q)
    
    ratio = σ_q / μ_q
    σ_ln = sqrt(log(1.0 + ratio^2))
    μ_ln = log(μ_q) - σ_ln^2 / 2.0
    
    q_hat_buggy = exp(μ_ln + sqrt(2.0) * σ_ln * χ1_physical)
    
    # --- Log Space (Proposed Fix) ---
    # Transform inputs to log space
    L_ln = L_physical / μ_q
    σ_q_ln = σ_q / μ_q
    
    # Run Rosenblatt in log space
    δq_ln = centered_uniform_gaussian_convolution_quantile_brent(p, L_ln, σ_q_ln)
    
    # Compute the "fake" physical fluctuation that will trick get_physical_point
    # into using the correct χ1
    δq_fake = δq_ln * (σ_q / max(σ_ln, ε))
    
    # _physical_Tq_from_fluctuations divides by σ_q
    χ1_fixed = δq_fake / (sqrt(2.0) * σ_q)
    
    # get_physical_point evaluates the lognormal
    q_hat_fixed = exp(μ_ln + sqrt(2.0) * σ_ln * χ1_fixed)
    
    println("Cell Mean μ_q = ", μ_q)
    println("Cell Physical Gradient L = ", L_physical)
    println("\n[Current Buggy Implementation]")
    println("q_hat = ", q_hat_buggy)
    println("\n[Proposed Log-Space Fix]")
    println("q_hat = ", q_hat_fixed)
    
    # Compare with a pure LogNormal distribution evaluated at the local mean
    local_mean = μ_q + (L_physical/2.0) * (2.0 * p - 1.0)
    println("\nTrue local mean at quantile p: ", local_mean)
end

demo()
