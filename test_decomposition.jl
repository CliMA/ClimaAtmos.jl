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
    
    # 1. Brent Quantile
    u_i = centered_uniform_gaussian_convolution_quantile_brent(p, L_physical, σ_q)
    
    # 2. Decompose
    z_i = L_physical * (p - 0.5)
    u_turb = u_i - z_i
    
    # standard normal deviate for the turbulence
    χ_turb = u_turb / σ_q
    
    println("p = ", p)
    println("u_i (total fluc) = ", u_i)
    println("z_i (macroscopic) = ", z_i)
    println("u_turb (microscopic) = ", u_turb)
    println("χ_turb (standard dev) = ", χ_turb)
    
    # 3. Apply LogNormal ONLY to turbulence
    q_macro = max(ε, μ_q + z_i)
    
    ratio = σ_q / max(ε, q_macro)
    σ_ln_local = sqrt(log(1.0 + ratio^2))
    μ_ln_local = log(q_macro) - σ_ln_local^2 / 2.0
    
    q_hat = exp(μ_ln_local + sqrt(2.0) * σ_ln_local * χ_turb)
    
    println("\nq_macro = ", q_macro)
    println("q_hat (Decomposed LogNormal) = ", q_hat)
    
    # Compare with pure affine
    q_affine = max(ε, μ_q + u_i)
    println("q_affine = ", q_affine)
end

demo()
