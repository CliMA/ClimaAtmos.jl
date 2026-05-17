"""Fixed path — always the same env on this machine (avoids hash(~/repo) ≠ hash(/net/.../repo))."""

"""`Pkg.activate` the plot env; first time only, `develop` ClimaAtmos + `add` CairoMakie."""
function wf_quad_activate_env!()
    env = joinpath(tempdir(), "climaatmos_wf_new_solver_2")
    if !ispath(joinpath(env, "Project.toml"))
        mkpath(env)
        Pkg.activate(env)
        Pkg.add([])
        println("WF new solver 2: initialized ", env)
    else
        Pkg.activate(env)
    end
    return env
end



wf_quad_activate_env!()
using Pkg; Pkg
Pkg.instantiate()

using Test

# ==========================================
# 1. THE SOLVER MODULE
# ==========================================
module FastWaterFilling

@inline function _Φ_fast(x::FT) where {FT}
    a = FT(0.147)
    s = ifelse(x >= zero(FT), FT(1), FT(-1))
    xx = abs(x) / sqrt(FT(2))
    t = exp(-xx * xx * (FT(4) / FT(π) + a * xx * xx) / (FT(1) + a * xx * xx))
    return FT(0.5) * (FT(1) + s * sqrt(FT(1) - t))
end

@inline function _φ_fast(x::FT) where {FT}
    return exp(-FT(0.5) * x * x) / sqrt(FT(2π))
end

@inline function _qc_tail_Q(a::FT) where {FT}
    Q_std = _φ_fast(a) - a * (FT(1) - _Φ_fast(a))
    a_safe = max(a, floatmin(FT))
    inv_a2 = FT(1) / (a_safe * a_safe)
    
    # CORRECTED: 2*log(a) and -3*inv_a2
    log_Q = -FT(0.5) * a_safe * a_safe - FT(2) * log(a_safe) - FT(0.5) * log(FT(2π)) + log1p(-FT(3) * inv_a2 + FT(15) * inv_a2 * inv_a2)
    Q_asymp = ifelse(log_Q < log(floatmin(FT)), zero(FT), exp(log_Q))
    
    return ifelse(Q_std > zero(FT), Q_std, Q_asymp)
end


@inline function qc_from_lambda_1d(λ::FT, A::FT) where {FT}
    A_safe = max(A, floatmin(FT))
    x = λ / A_safe
    qc_pos = λ * _Φ_fast(x) + A * _φ_fast(x)
    qc_neg = A * _qc_tail_Q(-x)
    qc = ifelse(λ >= zero(FT), qc_pos, qc_neg)
    return ifelse(A > zero(FT), qc, max(zero(FT), λ))
end

# ---------------------------------------------------------
# THE 0-ITERATION POLYNOMIAL SOLVER (Maximum SYPD)
# ---------------------------------------------------------
@inline function lambda_shift_fast_sub1percent(μ::FT, A::FT, q_target::FT) where {FT}
    Q = q_target / max(A, floatmin(FT))
    
    # Moist Regime FMA Chain (Fits x directly to Q)
    x_moist = FT(-0.8407889410145899) + Q*(FT(1.4883445831206121) + Q*(FT(-0.1983059089903901) + Q*(FT(0.04416198642777178) + Q*(FT(-0.005527878347738202) + Q*(FT(0.0003923315668516086) + Q*(FT(-1.4925769747517651e-5) + Q*FT(2.3686867690623296e-7)))))))
    
    # Dry Tail FMA Chain (Fits x to L-space transformed Q)
    Q_safe = max(Q, floatmin(FT))
    L = sqrt(max(zero(FT), -FT(2) * log(Q_safe * FT(2.5066282746310002))))
    x_dry = FT(0.24545934526363673) + L*(FT(-1.442036070624898) + L*(FT(0.3986926315835698) + L*(FT(-0.25881432420718536) + L*(FT(0.1264858022722108) + L*(FT(-0.03842188444458872) + L*(FT(0.006509413243292415) + L*FT(-0.0004724734185265492)))))))
    
    # Branchless Multiplex
    is_moist = Q > FT(0.39894228)
    x = ifelse(is_moist, x_moist, x_dry)
    
    return max(A * x, μ)
end

end # module

# ==========================================
# 2. VERIFICATION TEST (< 1% Error)
# ==========================================
@testset "0-Iteration Sub-1% Verification" begin
    FT = Float64
    μ_test = FT(-3e-3)
    A_test = FT(2.923707064472669e-4)
    q_target_test = FT(1e-6)
    
    λ_poly = FastWaterFilling.lambda_shift_fast_sub1percent(μ_test, A_test, q_target_test)
    q_rec_poly = FastWaterFilling.qc_from_lambda_1d(λ_poly, A_test)
    rel_err = abs(q_rec_poly - q_target_test) / q_target_test

    println("\n--- VERIFICATION: PHYSICAL COUNTEREXAMPLE ---")
    println("Target Q: ", q_target_test)
    println("Reconstructed Q: ", q_rec_poly)
    println("Relative Error: ", rel_err, " (", round(rel_err * 100, digits=3), "%)")

    # Strictly testing for < 1% error
    @test rel_err <= FT(0.01)
end