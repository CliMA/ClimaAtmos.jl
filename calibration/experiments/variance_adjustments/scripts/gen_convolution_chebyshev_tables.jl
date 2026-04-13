#!/usr/bin/env julia
# Offline fit: Chebyshev series in τ for u/L (uniform–Gaussian convolution quantile), per (N_GL, i_node).
#
#   julia --project=/path/to/calibration/experiments/variance_adjustments \
#       /path/to/gen_convolution_chebyshev_tables.jl
#
# Tune DEG (must match `NCOEFF_CONV_CHEB = DEG+1` in `convolution_quantile_chebyshev_tables.jl`).

import Pkg
const ROOT = joinpath(@__DIR__, "..")
Pkg.activate(ROOT)
using ClimaAtmos

const ℓ0 = log10(1.0e-4)
const ℓ1 = log10(1.0e2)
const DEG = 12  # polynomial degree; NCOEFF = DEG + 1 (must match `NCOEFF_CONV_CHEB` in tables)
const NCOEFF = DEG + 1

function τ_from_η(η::Float64)
    ℓ = log10(η)
    τ = 2 * (ℓ - ℓ0) / (ℓ1 - ℓ0) - 1
    return clamp(τ, -1.0, 1.0)
end

function cheb_T_matrix(τv::Vector{Float64}, ncoef::Int = NCOEFF)
    n = length(τv)
    A = zeros(n, ncoef)
    for (m, τ) in enumerate(τv)
        T_prev2 = 1.0
        T_prev1 = τ
        A[m, 1] = T_prev2
        ncoef >= 2 && (A[m, 2] = T_prev1)
        for j in 3:ncoef
            Tj = 2τ * T_prev1 - T_prev2
            A[m, j] = Tj
            T_prev2, T_prev1 = T_prev1, Tj
        end
    end
    return A
end

function eval_cheb(c::Vector{Float64}, τ::Float64)
    ncoef = length(c)
    T_prev2 = 1.0
    T_prev1 = τ
    acc = c[1] * T_prev2
    ncoef >= 2 && (acc += c[2] * T_prev1)
    for j in 3:ncoef
        Tj = 2τ * T_prev1 - T_prev2
        acc += c[j] * Tj
        T_prev2, T_prev1 = T_prev1, Tj
    end
    return acc
end

function fit_coeffs(p::Float64; nη::Int = 1200)
    ηs = 10 .^ range(ℓ0, ℓ1; length = nη)
    τv = [τ_from_η(η) for η in ηs]
    y = zeros(nη)
    for (k, η) in enumerate(ηs)
        L = 1.0
        s = η
        u = ClimaAtmos._convolution_quantile_bracketed(p, L, s)
        y[k] = u / L
    end
    A = cheb_T_matrix(τv)
    c = A \ y
    max_err = maximum(abs(eval_cheb(c, τv[k]) - y[k]) for k in 1:nη)
    return c, max_err
end

function main()
    println(stderr, "# Fitting Chebyshev degree-$DEG ($NCOEFF coeffs, Float64) …")
    worst = 0.0
    worst_key = (0, 0)
    allc = Dict{Tuple{Int,Int},Vector{Float64}}()
    for N in 1:5
        p_nodes, _ = ClimaAtmos.gauss_legendre_01(Float64, N)
        for i in 1:N
            c, err = fit_coeffs(Float64(p_nodes[i]))
            allc[(N, i)] = c
            if err > worst
                worst = err
                worst_key = (N, i)
            end
            println(stderr, "  N=$N i=$i max|fit-true| = $err")
        end
    end
    println(stderr, "# worst case: ", worst_key, " max_err = ", worst)
    println()
    println("# ---- paste into chebyshev_convolution_coeffs (replace body) ----")
    for N in 1:5
        for i in 1:N
            c = allc[(N, i)]
            nums = join([repr(x) for x in c], ", ")
            println("    if N_GL == $(N) && i_node == $(i)")
            println("        return SA.SVector{$(NCOEFF),FT}($(nums))")
            println("    end")
        end
    end
end

main()
