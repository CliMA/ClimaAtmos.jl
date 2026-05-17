using BenchmarkTools
using Statistics
using Printf

# ============================================================
# FAST erf APPROXIMATION (no dependencies)
# ============================================================

@inline function erf_fast(x::T) where {T<:AbstractFloat}
    a = T(0.147)

    s = sign(x)
    xx = abs(x)

    t = one(T) + a * xx * xx

    y = one(T) - exp(-(xx * xx) *
        ((T(4) / T(pi)) + a * xx * xx) / t)

    return s * sqrt(y)
end

# ============================================================
# Φ and φ
# ============================================================

@inline Φ_fast(x::T) where {T<:AbstractFloat} =
    T(0.5) * (one(T) + erf_fast(x / sqrt(T(2))))

@inline φ_fast(x::T) where {T<:AbstractFloat} =
    exp(-T(0.5) * x * x) / sqrt(T(2π))

# ============================================================
# FAST erfinv APPROXIMATION
# ============================================================

@inline function erfinv_fast(x::T) where {T<:AbstractFloat}
    a = T(0.147)

    xx = clamp(abs(x), T(1e-12), T(0.999999999))

    ln_term = log(one(T) - xx * xx)

    t1 = T(2)/(T(pi)*a) + ln_term/T(2)

    inside = t1*t1 - ln_term/a

    return sign(x) * sqrt(sqrt(inside) - t1)
end

# ============================================================
# FORWARD MODEL
# ============================================================

@inline function qc_fast(λ::T, A::T) where {T<:AbstractFloat}
    x = λ / A
    Φ = Φ_fast(x)
    φ = φ_fast(x)
    return λ * Φ + A * φ
end

@inline dqc_fast(λ::T, A::T) where {T<:AbstractFloat} =
    Φ_fast(λ / A)

# ============================================================
# INITIAL GUESS (RESTORED — NOT SIMPLIFIED)
# ============================================================

@inline function λ_initial_guess_older(q::T, A::T) where {T<:AbstractFloat}
    p = q / (q + A)

    p = clamp(p, T(1e-12), one(T) - T(1e-12))

    x0 = sqrt(T(2)) * erfinv_fast(T(2)*p - one(T))

    return max(A * x0, zero(T))
end

@inline function λ_initial_guess_old(q::T, A::T) where {T}

    # normalize
    qn = q / A

    # asymptotic linear branch (dominant for moderate/large λ)
    x_lin = qn

    # small-x correction anchor (Gaussian core)
    φ0 = inv(sqrt(T(2π)))

    x_small = qn / φ0

    # blending weight (smooth transition)
    w = qn / (qn + T(1))

    x0 = (one(T) - w) * x_small + w * x_lin

    return max(A * x0, zero(T))
end

@inline function λ_initial_guess(q::T, A::T) where {T<:AbstractFloat}

    # small-λ asymptote
    q0 = A / sqrt(T(2π))
    λ_small = T(2) * (q - q0)

    # large-λ asymptote
    λ_large = q

    # smooth blending variable (based on regime strength)
    r = q / (q + A)

    w = clamp(r, T(0), T(1))

    λ0 = w * λ_large + (one(T) - w) * max(λ_small, zero(T))

    return max(λ0, zero(T))
end

# ============================================================
# FAST INVERSE (1 Newton step default)
# ============================================================

function λ_from_q_fast(q::T, A::T; iters::Int=1) where {T}
    λ = λ_initial_guess(q, A)

    for _ in 1:iters
        f = qc_fast(λ, A) - q
        df = max(dqc_fast(λ, A), T(1e-12))
        λ -= f / df
        λ = max(λ, zero(T))
    end

    return λ
end

# ============================================================
# REFERENCE SOLVER
# ============================================================

function λ_from_q_reference(q::T, A::T) where {T}
    λ = max(q, zero(T))

    for _ in 1:25
        f = qc_fast(λ, A) - q
        df = max(dqc_fast(λ, A), T(1e-12))
        λ -= f / df
        λ = max(λ, zero(T))
    end

    return λ
end

# ============================================================
# VALIDATION
# ============================================================

function run_validation(::Type{T}) where {T}

    println("\n" * "="^80)
    println("VALIDATION: $T")
    println("="^80)

    A = T(1)

    λ_vals = collect(range(T(0), T(8), length=20))

    λ_errs = T[]
    q_errs = T[]

    println("\nSAMPLE POINTS (λ true | q true | λ est | q recon | q rel err)")
    println("-"^80)

    for λ_true in λ_vals

        q_true = qc_fast(λ_true, A)
        λ_est  = λ_from_q_fast(q_true, A; iters=1)
        q_recon = qc_fast(λ_est, A)

        push!(λ_errs, abs(λ_est - λ_true) / max(abs(λ_true), T(1e-12)))
        push!(q_errs, abs(q_recon - q_true) / max(abs(q_true), T(1e-12)))

        @printf("λ true=%8.4f | q true=%10.4e | λ est=%8.4f | q recon=%10.4e | q rel=%.3e\n",
            λ_true, q_true, λ_est, q_recon,
            abs(q_recon - q_true) / max(abs(q_true), T(1e-12)))
    end

    println("\nERROR STATS")
    println("-"^80)

    @printf("Mean λ error : %.3e\n", mean(λ_errs))
    @printf("Max  λ error : %.3e\n", maximum(λ_errs))
    @printf("Mean q error : %.3e\n", mean(q_errs))
    @printf("Max  q error : %.3e\n", maximum(q_errs))
end

# ============================================================
# BENCHMARKS (FIXED INTERPOLATION BUG)
# ============================================================

function run_benchmarks(::Type{T}) where {T}

    println("\n" * "="^80)
    println("BENCHMARKS: $T")
    println("="^80)

    A = T(1)
    q_test = qc_fast(T(3), A)

    println("\nFast inverse:")
    display(@benchmark λ_from_q_fast($q_test, $A; iters=1))

    println("\nReference inverse:")
    display(@benchmark λ_from_q_reference($q_test, $A))

    println("\nForward model:")
    display(@benchmark qc_fast($(T(3)), $A))
end

# ============================================================
# MAIN
# ============================================================

function main()
    println("\n" * "="^80)
    println("FAST SGS ROOT SOLVER (STABLE VERSION)")
    println("="^80)

    run_validation(Float32)
    run_benchmarks(Float32)

    run_validation(Float64)
    run_benchmarks(Float64)

    println("\nDONE")
end

main()