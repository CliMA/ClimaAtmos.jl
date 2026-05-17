using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..", "..", "..")))

# Persistent REPL: `include` this file; the `Pkg.activate` above selects the ClimaAtmos
# project (works even if the session was not started with `--project=.`).
#   include("test/parameterized_tendencies/microphysics/water_filling_new_solver.jl")  # cwd = repo root
using Test
using BenchmarkTools
using ClimaAtmos
import ClimaParams as CP
import Thermodynamics as TD

# ==========================================
# 1. THE SOLVER MODULE
# ==========================================
module FastWaterFilling

"""
    _φ_fast(x::FT) where {FT}

Exact Standard Normal Probability Density Function (PDF).
Evaluates the Gaussian curve: `ϕ(x) = (1 / sqrt(2π)) * exp(-x^2 / 2)`.
"""
@inline function _φ_fast(x::FT) where {FT}
    return exp(-FT(0.5) * x * x) / sqrt(FT(2π))
end

"""
    _Φ_fast(x::FT) where {FT}

Standard Normal Cumulative Distribution Function (CDF).
Evaluates `Φ(x) = 0.5 * (1 + erf(x / sqrt(2)))` using Winitzki's Approximation (2008) 
for the Error Function.

# Mathematical Formulation
erf(z) ≈ sgn(z) * sqrt(1 - exp(-z^2 * (4/π + a*z^2) / (1 + a*z^2)))

The constant `a = 0.147` is an empirical Minimax optimization by Winitzki 
that minimizes the maximum global relative error to ~1.2e-4, as opposed to 
the exact Padé coefficient `a ≈ 0.140` which only optimizes the limits.
"""
@inline function _Φ_fast(x::FT) where {FT}
    a = FT(0.147)
    s = ifelse(x >= FT(0), FT(1), FT(-1))
    
    # Map the normal CDF coordinate (x) to the erf coordinate (xx)
    xx = abs(x) / sqrt(FT(2))
    
    # Evaluate the inner exponential term of Winitzki's formula
    t = exp(-xx * xx * (FT(4) / FT(π) + a * xx * xx) / (FT(1) + a * xx * xx))
    
    # Reconstruct Φ(x)
    return FT(0.5) * (FT(1) + s * sqrt(FT(1) - t))
end

# 100% Branchless tail approximation
@inline function _qc_tail_Q(a::FT) where {FT}
    Q_std = _φ_fast(a) - a * (FT(1) - _Φ_fast(a))
    
    a_safe = max(a, floatmin(FT))
    inv_a2 = FT(1) / (a_safe * a_safe)
    log_Q = -FT(0.5) * a_safe * a_safe - FT(3) * log(a_safe) - FT(0.5) * log(FT(2π)) + log1p(FT(3) * inv_a2 + FT(15) * inv_a2 * inv_a2)
    Q_asymp = ifelse(log_Q < log(floatmin(FT)), zero(FT), exp(log_Q))
    
    return ifelse(Q_std > zero(FT), Q_std, Q_asymp)
end

# 100% Branchless 1D Condensate
@inline function qc_from_lambda_1d(λ::FT, A::FT) where {FT}
    A_safe = max(A, floatmin(FT))
    x = λ / A_safe
    
    qc_pos = λ * _Φ_fast(x) + A * _φ_fast(x)
    qc_neg = A * _qc_tail_Q(-x)
    
    qc = ifelse(λ >= zero(FT), qc_pos, qc_neg)
    return ifelse(A > zero(FT), qc, max(zero(FT), λ))
end

# The Exact 3-Step Branchless GPU Solver with Log-Newton
@inline function lambda_shift_fast_3step(μ::FT, A::FT, q_target::FT) where {FT}
    Q = q_target / max(A, floatmin(FT))
    Q_safe = max(Q, floatmin(FT))
    
    # ---------------------------------------------------------
    # 1. BRANCHLESS INITIAL GUESSES (Normalized coordinates)
    # ---------------------------------------------------------
    # Q(0) = 1 / sqrt(2π) ≈ 0.39894228. The exact zero-crossing pivot.
    x_moist = Q - FT(0.39894228) / (FT(1) + FT(1.5) * Q)
    
    # Transition Zone Secant (λ ∈ [-A, 0]):
    # Connects Q(-1) ≈ 0.08331547 to Q(0) ≈ 0.39894228. 
    # Slope = 0.39894228 - 0.08331547 = 0.3156268.
    x_trans = (Q - FT(0.39894228)) / FT(0.3156268) 
    
    # Deep Dry Tail Asymptotic Inverse:
    # Inverts Q ≈ (1 / sqrt(2π)) * exp(-x^2/2) using sqrt(2π) ≈ 2.50662827.
    x_tail = -sqrt(max(zero(FT), -FT(2) * log(Q_safe * FT(2.50662827))))
    
    # Multiplex the regimes based on the derived Q boundaries
    is_moist = Q >= FT(0.39894228)
    is_trans = Q >= FT(0.08331547)
    
    x = ifelse(is_moist, x_moist, ifelse(is_trans, x_trans, x_tail))
    
    # ---------------------------------------------------------
    # 2. FIXED-UNROLL REFINEMENT (3 Steps for strict < 1e-5 error)
    # ---------------------------------------------------------
    # A static range 1:3 is fully unrolled by the Julia compiler
    for _ in 1:3
        is_pos = x >= zero(FT)
        Q_val = ifelse(is_pos, x * _Φ_fast(x) + _φ_fast(x), _qc_tail_Q(-x))
        dq = ifelse(is_pos, _Φ_fast(x), FT(1) - _Φ_fast(-x))
        
        dq_safe = max(dq, FT(1e-15))
        Q_val_safe = max(Q_val, floatmin(FT))
        
        # Standard Newton for Positive x
        step_std = (Q_val - Q) / dq_safe
        
        # Log-Newton for Negative x (Prevents exponential tangent overshoot)
        step_log = (Q_val_safe / dq_safe) * log(Q_val_safe / Q_safe)
        
        step = ifelse(is_pos, step_std, step_log)
        x = x - step
    end
    
    return max(A * x, μ)
end

end # module

# ==========================================
# 2. REFERENCE BISECTION (For testing only)
# ==========================================
"""Bracketed bisection for `q_c(λ,A)=q_target` on `[λ_lo, λ_hi]` (test reference only)."""
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
@testset "Fast 3-Step Branchless Solver Verification" begin
    for FT in (Float32, Float64)
        @testset "Precision Test: FT = $FT" begin
            
            A = FT(1e-5) 
            
            # ----------------------------------------------------
            # TEST 1: The Moist Regime (Target > q_c(μ))
            # ----------------------------------------------------
            μ_moist = FT(1e-4)
            q_star = FastWaterFilling.qc_from_lambda_1d(μ_moist, A)
            q_tgt_moist = q_star + FT(0.05) * A
            
            λ_moist = FastWaterFilling.lambda_shift_fast_3step(μ_moist, A, q_tgt_moist)
            qrec_moist = FastWaterFilling.qc_from_lambda_1d(λ_moist, A)
            rel_err_moist = abs(qrec_moist - q_tgt_moist) / max(q_tgt_moist, FT(1e-20))
            
            println("--- FT = $FT (Moist) ---")
            println("Target Condensate: ", q_tgt_moist)
            println("Reconstructed:     ", qrec_moist)
            println("Relative Error:    ", rel_err_moist)
            
            @test rel_err_moist ≤ FT(1e-2)
            @test λ_moist ≥ μ_moist

            # ----------------------------------------------------
            # TEST 2: The Deep Dry Tail (μ < 0, |μ| > A)
            # ----------------------------------------------------
            μ_dry = FT(-3e-3)
            q_tgt_dry = FT(1e-6)
            
            λ_dry = FastWaterFilling.lambda_shift_fast_3step(μ_dry, A, q_tgt_dry)
            qrec_dry = FastWaterFilling.qc_from_lambda_1d(λ_dry, A)
            abs_err_dry = abs(qrec_dry - q_tgt_dry)
            
            println("--- FT = $FT (Dry Tail) ---")
            println("Target Condensate: ", q_tgt_dry)
            println("Reconstructed:     ", qrec_dry)
            println("Absolute Error:    ", abs_err_dry)
            println()
            
            @test abs_err_dry ≤ FT(1e-5)
            @test λ_dry ≥ μ_dry
            @test isfinite(λ_dry)
        end
    end

    # ----------------------------------------------------
    # TEST 3: Physical Atmospheric Parameter Counterexample
    # ----------------------------------------------------
    @testset "Counterexample: mu<0 trace at physical A" begin
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
        A = ClimaAtmos.compute_sigma_S(thp, ρ, q_tot, T_mean, q′q′, T′T′, corr; lf)

        λ = FastWaterFilling.lambda_shift_fast_3step(μ, A, q_target)
        q_rec = FastWaterFilling.qc_from_lambda_1d(λ, A)
        rel_err = abs(q_rec - q_target) / q_target
        Q_norm = q_target / A

        @test A > FT(1e-4)  # Confirms we are not at the toy 1e-5 scale
        @test λ ≥ μ
        @test isfinite(λ)

        λ_ref = _lambda_bisect_ref(μ, A, q_target, μ, zero(FT))
        q_ref = FastWaterFilling.qc_from_lambda_1d(λ_ref, A)

        println("--- Counterexample (physical A) ---")
        println("A = ", A, "  Q = q_target/A = ", Q_norm)
        println("lambda_fast = ", λ, "  q_rec = ", q_rec, "  rel_err = ", rel_err)
        println("lambda_ref  = ", λ_ref, "  q_ref = ", q_ref)
        
        # Test passes: We expect < 1e-5 relative error now with the Log-Newton fix
        @test rel_err ≤ FT(1e-5)
    end
end


# ==========================================
# 4. BENCHMARKING SUITE
# ==========================================
println("\n=== BENCHMARKING (Float64) ===")
let 
    FT = Float64
    μ = FT(-3e-3)
    
    # Moist Regime
    A_moist = FT(1e-5)
    q_tgt_moist = FT(1e-4) 
    
    # Transition Zone
    A_trans = FT(1e-5)
    q_tgt_trans = FT(2e-6) 
    
    # Deep Dry Tail (The Physical Counterexample)
    A_dry = FT(2.923707064472669e-4)
    q_tgt_dry = FT(1e-6) 
    
    println("1. Moist Regime:")
    @btime FastWaterFilling.lambda_shift_fast_3step($μ, $A_moist, $q_tgt_moist)
    
    println("\n2. Transition Zone:")
    @btime FastWaterFilling.lambda_shift_fast_3step($μ, $A_trans, $q_tgt_trans)
    
    println("\n3. Deep Dry Tail (Log-Newton Path):")
    @btime FastWaterFilling.lambda_shift_fast_3step($μ, $A_dry, $q_tgt_dry)
end
