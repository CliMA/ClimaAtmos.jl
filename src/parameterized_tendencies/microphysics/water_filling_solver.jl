import Thermodynamics as TD


# =============================
# Fast 1D water-filling helpers
# =============================

@inline function _Φ_fast(x::FT) where {FT}
    a = FT(0.147)
    s = ifelse(x >= FT(0), FT(1), FT(-1))
    xx = abs(x) / sqrt(FT(2))
    t = exp(-xx * xx * (FT(4) / FT(π) + a * xx * xx) / (FT(1) + a * xx * xx))
    erf_approx = s * sqrt(FT(1) - t)
    return FT(0.5) * (FT(1) + erf_approx)
end

@inline function _φ_fast(x::FT) where {FT}
    return exp(-FT(0.5) * x * x) / sqrt(FT(2π))
end


"""
    _qc_tail_Q(a)

`∫_a^∞ (t - a) φ(t) dt = φ(a) - a (1 - Φ(a))` for `a > 0`.

Uses the direct formula for small `a`. For `a ≥`3` or when the direct
value is non-positive, uses a large-`a` log-asymptotic (accurate when `φ(a)` underflows):

`Q(a) ~ φ(a)/a² · (1 - 3/a² + 15/a⁴ - …)` ⇒
`log Q ≈ -a²/2 - (1/2)log(2π) - 2log(a) + log1p(-3/a² + 15/a⁴ + …)`.
"""
@inline function _qc_tail_Q(a::FT) where {FT}
    φa = _φ_fast(a)
    Φa = _Φ_fast(a)
    Q = φa - a * (FT(1) - Φa)
    if (Q > zero(FT)) && (a < FT(3)) # Winitzki `Φ` is used for speed; for `a ≳ 3` the direct tail integral can be positive but inaccurate, so switch to the log-asymptotic below (verified vs high-precision quadrature).
        return Q
    end
    a_safe = max(a, floatmin(FT))
    inv_a2 = FT(1) / (a_safe * a_safe)
    log_Q =
        -FT(0.5) * a_safe * a_safe - FT(2) * log(a_safe) - FT(0.5) * log(FT(2π)) +
        log1p(-FT(3) * inv_a2 + FT(15) * inv_a2 * inv_a2)
    log_min = log(floatmin(FT))
    return log_Q < log_min ? zero(FT) : exp(log_Q)
end

"""
    qc_from_lambda_1d(λ, A)

Grid-mean condensate from the 1D water-filling closure
`q_c = ∫ max(0, λ + A x) φ(x) dx` with `A = α σ_S`.

For `λ ≥ 0` uses `q_c = λ Φ(λ/A) + A φ(λ/A)`. For `λ < 0` uses the equivalent tail form
`q_c = A [φ(a) - a (1 - Φ(a))]` with `a = -λ/A > 0`, which avoids evaluating `Φ` and `φ`
at large negative arguments (where they underflow to zero).
"""
@inline function qc_from_lambda_1d(λ::FT, A::FT) where {FT}
    if !(A > FT(0))
        return max(FT(0), λ)
    end
    if λ >= zero(λ)
        x = λ / A
        return λ * _Φ_fast(x) + A * _φ_fast(x)
    end
    return A * _qc_tail_Q(-λ / A)
end

"""
    water_filling_subcapacity_scale_pos(q_cond_mean, μ, A)

Uniform scale on the positive part at the sub-capacity branch (`λ = μ`):
`scale = q_cond_mean / q_c(μ, A)` with `q_c` from the 1D Gaussian closure
[`water_filling_bulk_condensate_at_mu`](@ref) (continuous limit; Gauss–Hermite
quadrature approximates the same integral when applied to tendencies).
"""
@inline function water_filling_bulk_condensate_at_mu(μ::FT, A::FT) where {FT}
    if A > FT(0)
        return qc_from_lambda_1d(μ, A)
    else
        return max(FT(0), μ)
    end
end

"""
    water_filling_on_shift_branch(q_cond_mean, μ, A)

`true` when the shift branch applies (`q_cond_mean > q_c(μ, A)`). Caller handles `A == 0`.
"""
@inline function water_filling_on_shift_branch(q_cond_mean::FT, μ::FT, A::FT) where {FT}
    return q_cond_mean > qc_from_lambda_1d(μ, A)
end

@inline function water_filling_subcapacity_scale_pos(q_cond_mean::FT, μ::FT, A::FT) where {FT}
    if !(q_cond_mean > zero(FT))
        return zero(FT)
    end
    q_c_mu = water_filling_bulk_condensate_at_mu(μ, A)
    if !(q_c_mu > zero(FT))
        error(
            "sub-capacity scale requires q_c(μ, A) > 0 (got $q_c_mu); " *
            "use shift branch when q_cond_mean > q_c(μ, A)",
        )
    end
    return q_cond_mean / q_c_mu
end

"""
    water_filling_λ_scale_pos(q_cond_mean, μ, A, α, thermo_params, ρ, q_tot_mean, T_mean, q′q′, T′T′, corr_Tq; iters, lf)

Return `(λ, scale_pos)` for [`qcond_hat_water_filling`](@ref) / [`WaterFillingSGSEvaluator`](@ref).

When ``A = α σ_S = 0`` and ``q_{cond,mean} > 0``, use a **uniform** partition with
``\\hat{q}_c = q_{cond,mean}`` on every node (``λ = q_{cond,mean}``, ``scale_{pos} = 1``).
If ``q_{cond,mean} = 0``, fall back to ``λ = μ`` (zero condensate).
"""
@inline function water_filling_λ_scale_pos(
    q_cond_mean, μ, A, α,
    thermo_params, ρ, q_tot_mean, T_mean, q′q′, T′T′, corr_Tq;
    iters::Int = 1,
    lf = nothing,
)
    FT = typeof(μ)
    if lf === nothing
        lf = FT(0.5)
    end
    if !(A > FT(0))
        if q_cond_mean > FT(0)
            return q_cond_mean, one(FT)
        else
            return μ, one(FT)
        end
    elseif !water_filling_on_shift_branch(q_cond_mean, μ, A)
        return μ, water_filling_subcapacity_scale_pos(q_cond_mean, μ, A)
    else
        λ = lambda_from_qc_1d(
            thermo_params, ρ, q_tot_mean, T_mean, q′q′, T′T′, corr_Tq, μ, q_cond_mean;
            α = α, iters = iters, lf = lf,
        )
        return λ, one(FT)
    end
end

@inline function dqc_dlambda_1d(λ::FT, A::FT) where {FT}
    x = λ / max(A, FT(1e-30))
    if λ < zero(λ)
        # λ = -A a, q_c = A Q(a) ⇒ dq_c/dλ = 1 - Φ(a) with a = -λ/A.
        return FT(1) - _Φ_fast(-x)
    end
    return _Φ_fast(x)
end

"""
    _secant_chord(x_lo, y_lo, x_hi, y_hi, y_target)

Chord step toward `y_target` between `(x_lo, y_lo)` and `(x_hi, y_hi)`.
"""
@inline function _secant_chord(
    x_lo::FT, y_lo::FT, x_hi::FT, y_hi::FT, y_target::FT,
) where {FT}
    Δy = y_hi - y_lo
    tol = max(abs(y_hi), abs(y_target), abs(y_lo), FT(1e-30)) * FT(1e-12)
    if Δy > tol
        return x_lo + (y_target - y_lo) * (x_hi - x_lo) / Δy
    end
    return FT(0.5) * (x_lo + x_hi)
end

"""
    _lambda_shift_bracket_hi(μ, A, q_target)

Smallest convenient `λ_hi ≥ max(μ, 0)` with `q_c(λ_hi, A) > q_target` (monotone inverse).
"""
@inline function _lambda_shift_bracket_hi(μ::FT, A::FT, q_target::FT) where {FT}
    λ_hi = max(zero(FT), μ)
    if qc_from_lambda_1d(λ_hi, A) > q_target
        return λ_hi
    end
    return max(q_target + A, μ + A)
end

"""
    _newton_step_tail_a(a, A, q_target)

One Newton correction on `A Q(a) = q_target` with `Q(a) = φ(a) - a(1 - Φ(a))` and `dQ/da = Φ(a) - 1`.
"""
@inline function _newton_step_tail_a(a::FT, A::FT, q_target::FT) where {FT}
    Qa = _qc_tail_Q(a)
    f = A * Qa - q_target
    dQda = _Φ_fast(a) - FT(1)
    return max(a - f / (A * dQda), zero(FT))
end

"""
    _lambda_shift_secant(μ, A, q_target)

Solve `q_c(λ, A) = q_target` with `λ ≥ μ` (`q_c` strictly increasing on the shift branch).

Single recipe (no sign of `μ`, no `|μ|/A` threshold):

1. Evaluate `q_c` at four ascending candidates  
   `μ`, `max(-A, μ)`, `max(0, μ)`, and `_lambda_shift_bracket_hi`.
2. Pick the consecutive pair that brackets `q_target` in `q` (prefer the interior
   `max(-A, μ) … max(0, μ)` chord when it brackets — avoids a bad `μ → 0` chord when the
   mean is far below saturation).
3. One secant step in `λ`; if `λ < 0`, one Newton step in `a = -λ/A` (curved tail `Q(a)`).
"""
@inline function _lambda_shift_secant(μ::FT, A::FT, q_target::FT) where {FT}
    q_mu = qc_from_lambda_1d(μ, A)
    λ_in = max(-A, μ)
    q_in = qc_from_lambda_1d(λ_in, A)
    λ_sat = max(zero(FT), μ)
    q_sat = qc_from_lambda_1d(λ_sat, A)
    λ_hi = _lambda_shift_bracket_hi(μ, A, q_target)
    q_hi = qc_from_lambda_1d(λ_hi, A)

    λ_lo = μ
    q_lo = q_mu
    λ_hi_use = λ_hi
    q_hi_use = q_hi

    if (q_mu < q_target) && (q_in > q_target) && (λ_in > μ)
        λ_lo, q_lo, λ_hi_use, q_hi_use = μ, q_mu, λ_in, q_in
    elseif (q_in < q_target) && (q_sat > q_target) && (λ_in < λ_sat)
        λ_lo, q_lo, λ_hi_use, q_hi_use = λ_in, q_in, λ_sat, q_sat
    elseif (q_sat < q_target) && (q_hi > q_target) && (λ_sat < λ_hi)
        λ_lo, q_lo, λ_hi_use, q_hi_use = λ_sat, q_sat, λ_hi, q_hi
    elseif (q_mu < q_target) && (q_hi > q_target)
        λ_lo, q_lo = μ, q_mu
    end

    λ = max(_secant_chord(λ_lo, q_lo, λ_hi_use, q_hi_use, q_target), μ)
    if λ < zero(FT)
        a = _newton_step_tail_a(-λ / A, A, q_target)
        λ = max(-A * a, μ)
    end
    return λ
end

"""One guarded Newton step on `q_c(λ) = q_target` in `λ` (skip when `dq_c/dλ` is negligible)."""
@inline function _lambda_newton_step(λ::FT, μ::FT, A::FT, q_target::FT) where {FT}
    f = qc_from_lambda_1d(λ, A) - q_target
    dq = dqc_dlambda_1d(λ, A)
    if dq > FT(1e-6)
        return max(λ - f / dq, μ)
    end
    return λ
end

@inline function compute_sigma_S(thermo_params, ρ, q_tot_mean, T_mean, q′q′, T′T′, corr_Tq; lf=nothing)
    FT = typeof(T_mean)
    qsat_l = TD.q_vap_saturation(thermo_params, T_mean, ρ, TD.Liquid())
    qsat_i = TD.q_vap_saturation(thermo_params, T_mean, ρ, TD.Ice())
    R_v = TD.Parameters.R_v(thermo_params)
    L_v = TD.latent_heat_vapor(thermo_params, T_mean)
    L_s = TD.latent_heat_sublim(thermo_params, T_mean)
    b_l = L_v * qsat_l / (R_v * T_mean * T_mean)
    b_i = L_s * qsat_i / (R_v * T_mean * T_mean)
    w = lf === nothing ? TD.liquid_fraction_ramp(thermo_params, T_mean) : FT(lf)
    b_eff = w * b_l + (FT(1) - w) * b_i
    σ_q = sqrt(max(q′q′, FT(0)))
    σ_T = sqrt(max(T′T′, FT(0)))
    sig2 = q′q′ + b_eff * b_eff * T′T′ - FT(2) * b_eff * corr_Tq * σ_T * σ_q
    return sqrt(max(sig2, FT(0)))
end

"""
    lambda_from_qc_1d(...)

Shift-branch inverse: find `λ` with `q_c(λ, α σ_S) = q_cond_mean` and `λ ≥ μ`.
[`_lambda_shift_secant`](@ref); optional `iters` extra Newton steps in `λ`.

When ``A = α σ_S = 0``, delegates to [`water_filling_λ_scale_pos`](@ref) (uniform ``λ = q_{cond,mean}`` when
``q_{cond,mean} > 0``).
"""
@inline function lambda_from_qc_1d(
    thermo_params, ρ, q_tot_mean, T_mean, q′q′, T′T′, corr_Tq, μ, q_cond_mean;
    α=one(typeof(T_mean)), iters::Int=0, lf=nothing,
)
    FT = typeof(T_mean)
    σ_S = compute_sigma_S(thermo_params, ρ, q_tot_mean, T_mean, q′q′, T′T′, corr_Tq; lf=lf)
    A = α * σ_S

    if !(A > FT(0))
        λ, = water_filling_λ_scale_pos(
            q_cond_mean, μ, A, α, thermo_params, ρ, q_tot_mean, T_mean, q′q′, T′T′, corr_Tq;
            iters, lf,
        )
        return λ
    end
    if !water_filling_on_shift_branch(q_cond_mean, μ, A)
        return μ
    end
    λ = _lambda_shift_secant(μ, A, q_cond_mean)
    for _ in 1:iters
        λ = _lambda_newton_step(λ, μ, A, q_cond_mean)
    end
    return λ
end

@inline function qcond_hat_water_filling(
    thermo_params, ρ, q_tot_hat, T_hat, S_mean, λ, α; scale_pos=one(typeof(T_hat)),
)
    FT = typeof(T_hat)
    q_sat_hat = TD.q_vap_saturation(thermo_params, T_hat, ρ)
    ΔS_hat = (q_tot_hat - q_sat_hat) - S_mean
    return max(FT(0), λ + α * ΔS_hat) * scale_pos
end

struct WaterFillingSGSEvaluator{TPS, FT}
    tps::TPS
    ρ::FT
    S_mean::FT
    λ::FT
    α::FT
    lf::FT
    scale_pos::FT
end

@inline function (eval::WaterFillingSGSEvaluator)(T_hat, q_tot_hat)
    FT = typeof(eval.ρ)
    q_cond = qcond_hat_water_filling(
        eval.tps, eval.ρ, q_tot_hat, T_hat, eval.S_mean, eval.λ, eval.α;
        scale_pos=eval.scale_pos,
    )
    q_liq = eval.lf * q_cond
    q_ice = (FT(1) - eval.lf) * q_cond
    return (; q_liq, q_ice)
end

@inline function compute_sgs_condensate_water_filling(
    thermo_params, SG_quad, ρ, q_tot_mean, T_mean, q′q′, T′T′, corr_Tq,
    q_liq_mean, q_ice_mean; α=one(typeof(T_mean)), iters::Int=1,
)
    FT = typeof(T_mean)
    q_cond_mean = q_liq_mean + q_ice_mean
    μ = q_tot_mean - TD.q_vap_saturation(thermo_params, T_mean, ρ)
    if !(q_cond_mean > FT(0))
        return (; q_liq = FT(0), q_ice = FT(0), λ = μ, A = FT(0))
    end
    lf = q_liq_mean / max(q_cond_mean, FT(1e-30))
    σ_S = compute_sigma_S(thermo_params, ρ, q_tot_mean, T_mean, q′q′, T′T′, corr_Tq; lf = lf)
    A = α * σ_S
    λ, scale_pos = water_filling_λ_scale_pos(
        q_cond_mean, μ, A, α, thermo_params, ρ, q_tot_mean, T_mean, q′q′, T′T′, corr_Tq;
        iters, lf,
    )
    eval = WaterFillingSGSEvaluator(thermo_params, ρ, μ, λ, α, lf, scale_pos)
    res = integrate_over_sgs(eval, SG_quad, q_tot_mean, T_mean, q′q′, T′T′, corr_Tq)
    return (; q_liq = res.q_liq, q_ice = res.q_ice, λ, A)
end
