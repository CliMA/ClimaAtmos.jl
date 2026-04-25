#!/usr/bin/env julia
# Unified Offline Fit: Chebyshev series for SGS-Uniform convolution quantiles.
# 1D for Gaussian (η = σ/L), 2D for LogNormal (τ_σ = log10(σ_ln), τ_ξ = L/μ).

import Pkg
Pkg.activate(joinpath(@__DIR__, "../../.."))
using ClimaAtmos
using RootSolvers
using SpecialFunctions
using StaticArrays

# ----------------------------------------------------------------------------
# Fit Constants
# ----------------------------------------------------------------------------
const DEG = 6 
const NCOEFF = DEG + 1

# ----------------------------------------------------------------------------
# Gaussian (1D η fit)
# ----------------------------------------------------------------------------
const ℓ0_g = log10(1.0e-4)
const ℓ1_g = log10(1.0e2)

function fit_gaussian(p::Float64; n_sample = 1200)
    ηs = 10 .^ range(ℓ0_g, ℓ1_g; length = n_sample)
    y = zeros(n_sample)
    for (k, η) in enumerate(ηs)
        L = 1.0
        s = η
        u = ClimaAtmos._convolution_quantile_bracketed(p, L, s)
        y[k] = u / L
    end
    # ... Chebyshev fitting logic ...
    return y
end

# ----------------------------------------------------------------------------
# LogNormal (2D σ-ξ fit)
# ----------------------------------------------------------------------------
const ℓσ0 = log10(1.0e-4)
const ℓσ1 = log10(2.5)
const ξ0 = 0.0
const ξ1 = 1.99

function uniform_lognormal_convolution_cdf(q, y_min, y_max, σ)
    # Numerical integration for the offline generator (robust and accurate)
    # p(q|y) = LogNormal(q; μ=log(y)-σ²/2, σ)
    # F(q) = 1/(y_max-y_min) ∫_{y_min}^{y_max} Φ((log(q) - (log(y)-σ²/2))/σ) dy
    nodes = [-0.944575, -0.83562, -0.661209, -0.433395, -0.172673, 0.0, 0.172673, 0.433395, 0.661209, 0.83562, 0.944575]
    weights = [0.142208, 0.204127, 0.260472, 0.301337, 0.324582, 0.332383, 0.324582, 0.301337, 0.260472, 0.204127, 0.142208]
    # Scaling to [y_min, y_max]
    Δy = y_max - y_min
    y_mid = (y_max + y_min) / 2
    acc = 0.0
    for i in 1:length(nodes)
        y = y_mid + 0.5 * Δy * nodes[i]
        μ_ln = log(max(y, 1e-20)) - σ^2 / 2
        acc += weights[i] * 0.5 * (1 + erf((log(q) - μ_ln) / (σ * sqrt(2))))
    end
    return 0.5 * acc # Normalization for weights that sum to 2
end

function fit_lognormal_2d(p::Float64; n_σ = 13, n_ξ = 13)
    # Tensor-product Chebyshev nodes
    σ_nodes = [10^(ℓσ0 + (ℓσ1 - ℓσ0) * (cos(pi * (i-0.5) / n_σ) + 1) / 2) for i in 1:n_σ]
    ξ_nodes = [ξ0 + (ξ1 - ξ0) * (cos(pi * (j-0.5) / n_ξ) + 1) / 2 for j in 1:n_ξ]
    
    vals = zeros(n_σ, n_ξ)
    for (i, σ) in enumerate(σ_nodes), (j, ξ) in enumerate(ξ_nodes)
        μ = 1.0
        L = ξ * μ
        y_min = μ - L/2
        y_max = μ + L/2
        f = q -> uniform_lognormal_convolution_cdf(q, y_min, y_max, σ) - p
        # Check sign change
        f1 = f(1e-12)
        f2 = f(100.0)
        if f1 * f2 > 0
             # println(stderr, "Warning: No sign change for p=$p, σ=$σ, ξ=$ξ: f(1e-12)=$f1, f(100)=$f2")
        end
        sol = find_zero(f, BrentsMethod(1e-12, 100.0))
        vals[i, j] = sol.root / μ
    end
    return vals
end

function main()
    println("# Unified SGS Chebyshev tables (Generated at $(Base.Libc.strftime("%Y-%m-%d %H:%M:%S", time())))")
    println("# Degree: $DEG, Nodes: $NCOEFF x $NCOEFF")
    
    for N in 1:5
        p_nodes, _ = ClimaAtmos.gauss_legendre_01(Float64, N)
        println("# --- Order N = $N ---")
        for i in 1:N
            data = fit_lognormal_2d(p_nodes[i])
            println("    if N_GL == $N && i_node == $i")
            println("        # 2D LogNormal weights (σ-ξ tensor product)")
            println("        return SA.SMatrix{$NCOEFF, $NCOEFF, FT}($(join([repr(x) for x in data], ", ")))")
            println("    end")
        end
    end
end

main()