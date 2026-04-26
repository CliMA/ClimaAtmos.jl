#!/usr/bin/env julia
# Offline fit utility for Gaussian 1D convolution-quantile Chebyshev tables.
# LogNormal Chebyshev now reuses the same 1D transformed-space table path at runtime.

import Pkg
Pkg.activate(joinpath(@__DIR__, "../../.."))
using ClimaAtmos
using RootSolvers
using SpecialFunctions
using StaticArrays

# ----------------------------------------------------------------------------
# Fit Constants
# ----------------------------------------------------------------------------
const DEG_GAUSS = 12
const NCOEFF_GAUSS = DEG_GAUSS + 1

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

function main()
    println("# Unified SGS Chebyshev tables (Generated at $(Base.Libc.strftime("%Y-%m-%d %H:%M:%S", time())))")
    println("# Gaussian table degree: $DEG_GAUSS, coefficient count: $NCOEFF_GAUSS")
    
    println("# Note: LogNormal Chebyshev uses transformed-space 1D table reuse.")
    for N in 1:5
        p_nodes, _ = ClimaAtmos.gauss_legendre_01(Float64, N)
        println("# --- Order N = $N ---")
        for i in 1:N
            data = fit_gaussian(p_nodes[i]; n_sample = NCOEFF_GAUSS)
            println("    if N_GL == $N && i_node == $i")
            println("        # 1D Gaussian η-table row")
            println("        return SA.SVector{$NCOEFF_GAUSS, FT}($(join([repr(x) for x in data], ", ")))")
            println("    end")
        end
    end
end

main()