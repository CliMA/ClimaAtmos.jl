using Test
using BenchmarkTools

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
    
    # CORRECTED MATH: 2*log(a) and -3*inv_a2 reflect the true asymptotic expansion
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
# THE SUB-1% GPU SOLVER (Rational + 1-Step Polish)
# ---------------------------------------------------------
@inline function lambda_shift_fast_sub1percent(μ::FT, A::FT, q_target::FT) where {FT}
    Q = q_target / max(A, floatmin(FT))
    Q_safe = max(Q, floatmin(FT))
    
    # 1. BRANCHLESS RATIONAL GUESSES (Pole-Free, 256-bit safe)
    num_m = FT(-1.00880080266539) + Q * (FT(1.8494190718785757) + Q * (FT(1.1531336813462716) + Q * FT(1.4447670100538137)))
    den_m = FT(1.0) + Q * (FT(1.4038548603431869) + Q * (FT(1.4140460979469127) + Q * FT(0.001325844935640976)))
    x_moist = num_m / den_m
    
    num_d = FT(-5.373961465182952) + Q * (FT(-2.871325565000998e6) + Q * (FT(-1.6788061346206386e9) + Q * FT(3.5108679530464826e9)))
    den_d = FT(1.0) + Q * (FT(805356.6460356136) + Q * (FT(8.560681885478382e8) + Q * FT(6.64925331426817e9)))
    x_dry = num_d / den_d
    
    is_moist = Q > FT(0.39894228)
    x = ifelse(is_moist, x_moist, x_dry)
    
    # 2. EXACTLY ONE LOG-NEWTON POLISH STEP (Crushes error to ~0.3%)
    is_pos = x >= zero(FT)
    Q_val = ifelse(is_pos, x * _Φ_fast(x) + _φ_fast(x), _qc_tail_Q(-x))
    dq = ifelse(is_pos, _Φ_fast(x), FT(1) - _Φ_fast(-x))
    
    dq_safe = max(dq, FT(1e-15))
    Q_val_safe = max(Q_val, floatmin(FT))
    
    step_std = (Q_val - Q) / dq_safe
    step_log = (Q_val_safe / dq_safe) * log(Q_val_safe / Q_safe)
    
    x = x - ifelse(is_pos, step_std, step_log)
    
    return max(A * x, μ)
end

end # module

# ==========================================
# 2. REFERENCE BISECTION (For testing comparison)
# ==========================================
function _lambda_bisect_ref(μ, A, q_target, λ_lo, λ_hi; maxiter = 80)
    FT = typeof(μ)
    for _ in 1:maxiter
        λ_mid = FT(0.5) * (λ_lo + λ_hi)
        q_mid = FastWaterFilling.qc_from_lambda_1d(λ_mid, A)
        if q_mid < q_target
            λ_lo = max(λ_mid, μ)
        else
            λ_hi = λ_mid
        end
        if abs(λ_hi - λ_lo) ≤ max(abs(λ_hi), FT(1e-30)) * FT(1e-12)
            break
        end
    end
    return max(FT(0.5) * (λ_lo + λ_hi), μ)
end

# ==========================================
# 3. THE VERIFICATION TESTS
# ==========================================
@testset "Sub-1% GPU Solver Verification" begin
    for FT in (Float32, Float64)
        @testset "Precision Test: FT = $FT" begin
            A = FT(1e-5) 
            
            # Moist Regime
            μ_moist = FT(1e-4)
            q_star = FastWaterFilling.qc_from_lambda_1d(μ_moist, A)
            q_tgt_moist = q_star + FT(0.05) * A
            
            λ_moist = FastWaterFilling.lambda_shift_fast_sub1percent(μ_moist, A, q_tgt_moist)
            qrec_moist = FastWaterFilling.qc_from_lambda_1d(λ_moist, A)
            rel_err_moist = abs(qrec_moist - q_tgt_moist) / max(q_tgt_moist, FT(1e-20))
            @test rel_err_moist ≤ FT(0.01) # Sub 1%

            # Deep Dry Tail (Toy A)
            μ_dry = FT(-3e-3)
            q_tgt_dry = FT(1e-6)
            λ_dry = FastWaterFilling.lambda_shift_fast_sub1percent(μ_dry, A, q_tgt_dry)
            qrec_dry = FastWaterFilling.qc_from_lambda_1d(λ_dry, A)
            rel_err_dry = abs(qrec_dry - q_tgt_dry) / max(q_tgt_dry, FT(1e-20))
            @test rel_err_dry ≤ FT(0.01) # Sub 1%
        end
    end

    @testset "Counterexample: mu<0 trace at physical A" begin
        using ClimaAtmos
        import ClimaParams as CP
        import Thermodynamics as TD

        FT = Float64
        thp = TD.Parameters.ThermodynamicsParameters(CP.create_toml_dict(FT))
        ρ = FT(1)
        T_mean = FT(280)
        μ = FT(-3e-3)
        q_target = FT(1e-6)
        q_tot = TD.q_vap_saturation(thp, T_mean, ρ) + μ
        q′q′ = FT(1e-10)
        T′T′ = FT(0.25)
        corr = zero(FT)
        lf = FT(0.5)
        
        A = ClimaAtmos.compute_sigma_S(thp, ρ, q_tot, T_mean, q′q′, T′T′, corr; lf=lf)

        λ_pol = FastWaterFilling.lambda_shift_fast_sub1percent(μ, A, q_target)
        q_rec_pol = FastWaterFilling.qc_from_lambda_1d(λ_pol, A)
        rel_err_pol = abs(q_rec_pol - q_target) / q_target

        λ_ref = _lambda_bisect_ref(μ, A, q_target, μ, zero(FT))
        q_ref = FastWaterFilling.qc_from_lambda_1d(λ_ref, A)

        println("\n--- Counterexample (physical A) ---")
        println("A = ", A, "  Q = q_target/A = ", q_target/A)
        println("\n[POLISHED RATIONAL]")
        println("lambda = ", λ_pol, "  q_rec = ", q_rec_pol, "  rel_err = ", rel_err_pol)
        println("\n[REFERENCE BISECTION]")
        println("lambda = ", λ_ref, "  q_ref = ", q_ref)
        
        # Strictly testing for < 1% error
        @test rel_err_pol ≤ FT(0.01)
    end
end

# ==========================================
# 4. BENCHMARKING SUITE
# ==========================================
println("\n=== BENCHMARKING (Float64) ===")
let 
    FT = Float64
    μ = FT(-3e-3)
    A_dry = FT(2.923707064472669e-4)
    q_tgt_dry = FT(1e-6) 
    
    println("\nPolished Rational (1 iteration):")
    @btime FastWaterFilling.lambda_shift_fast_sub1percent($μ, $A_dry, $q_tgt_dry)
end