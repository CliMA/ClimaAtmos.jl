#!/usr/bin/env julia
# Offline fit utility for convolution-quantile surrogates.
# Gaussian uses 1D η = s/L.
# LogNormal uses a 2-parameter surface in (σ_ln, ξ=L/μ) for q/μ.
# IMPORTANT: the LogNormal runtime PDF/CDF are analytic; the 2D structure here is
# parameter-space tabulation for fast quantile inversion, not 2D physical quadrature.

import Pkg
Pkg.activate(joinpath(@__DIR__, "../../.."))
using ClimaAtmos
using RootSolvers
using SpecialFunctions
using StaticArrays

# ----------------------------------------------------------------------------
# Fit Constants
# ----------------------------------------------------------------------------
# Gaussian table degree (1D coefficients)
const DEG_GAUSS = 12
const NCOEFF_GAUSS = DEG_GAUSS + 1
# LogNormal table resolution (2D Chebyshev-root nodal grid)
const DEG_LN = 6
const NCOEFF_LN = DEG_LN + 1

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
# LogNormal (2D parameter-space fit in σ-ξ)
# ----------------------------------------------------------------------------
const ℓσ0 = log10(1.0e-4)
const ℓσ1 = log10(2.5)
const ξ0 = 0.0
const ξ1 = 1.99

function fit_lognormal_2d(p::Float64; n_σ = NCOEFF_LN, n_ξ = NCOEFF_LN)
    # Tensor-product Chebyshev nodes
    σ_nodes = [10^(ℓσ0 + (ℓσ1 - ℓσ0) * (cos(pi * (i-0.5) / n_σ) + 1) / 2) for i in 1:n_σ]
    ξ_nodes = [ξ0 + (ξ1 - ξ0) * (cos(pi * (j-0.5) / n_ξ) + 1) / 2 for j in 1:n_ξ]
    
    vals = zeros(n_σ, n_ξ)
    for (i, σ) in enumerate(σ_nodes), (j, ξ) in enumerate(ξ_nodes)
        μ = 1.0
        L = ξ * μ
        y_min = μ - L/2
        y_max = μ + L/2
        # Use the same analytic runtime primitive as the model code.
        f = q -> ClimaAtmos.uniform_lognormal_convolution_cdf(q, y_min, y_max, σ) - p
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
    println("# LogNormal table degree: $DEG_LN, Nodes: $NCOEFF_LN x $NCOEFF_LN")
    
    for N in 1:5
        p_nodes, _ = ClimaAtmos.gauss_legendre_01(Float64, N)
        println("# --- Order N = $N ---")
        for i in 1:N
            data = fit_lognormal_2d(p_nodes[i])
            println("    if N_GL == $N && i_node == $i")
            println("        # 2D LogNormal weights (σ-ξ tensor product)")
            println("        return SA.SMatrix{$NCOEFF_LN, $NCOEFF_LN, FT}($(join([repr(x) for x in data], ", ")))")
            println("    end")
        end
    end
end

main()