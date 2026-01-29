"""
    SGS Quadrature Utilities

Subgrid-scale (SGS) quadrature infrastructure for integrating point-wise functions
over thermodynamic fluctuations. Supports multiple distribution types and provides
reusable utilities for cloud fraction, microphysics tendencies, and other SGS diagnostics.
"""

import StaticArrays as SA

#####
##### Recursive Operations (local implementations to avoid RecursiveApply dependency; could 
##### be replaced with ClimaCore.RecursiveApply if needed)
#####

"""
    recursive_zero(x)

Return a zero of the same type/structure as `x`.
Works recursively on `NamedTuple`s.
"""
@inline recursive_zero(x::Number) = zero(x)
@inline recursive_zero(x::NamedTuple{K}) where {K} = NamedTuple{K}(map(recursive_zero, values(x)))

"""
    recursive_add(a, b)

Element-wise addition of `a` and `b`.
Works recursively on `NamedTuple`s.
"""
@inline recursive_add(a::Number, b::Number) = a + b
@inline recursive_add(a::NamedTuple{K}, b::NamedTuple{K}) where {K} = 
    NamedTuple{K}(map(recursive_add, values(a), values(b)))

"""
    recursive_mul(a, s)

Multiply `a` by scalar `s`.
Works recursively on `NamedTuple`s.
"""
@inline recursive_mul(a::Number, s::Number) = a * s
@inline recursive_mul(a::NamedTuple{K}, s::Number) where {K} = 
    NamedTuple{K}(map(x -> recursive_mul(x, s), values(a)))

#####
##### Gauss-Hermite Quadrature 
#####

"""
    gauss_hermite(FT, N)

Gauss-Hermite quadrature nodes and weights for order `N`.

Nodes are roots of the physicists' Hermite polynomial ``H_N(x)``.
Weights are standard Gauss-Hermite weights for integration against ``e^{-x^2}``.

# Arguments
- `FT`: Floating-point type
- `N::Int`: Quadrature order (1-5 supported)

# Returns
Tuple `(nodes, weights)` as `Vector{FT}`.
"""
function gauss_hermite(::Type{FT}, N::Int) where {FT}
    # Precomputed values for common orders (physicists' Hermite polynomials)
    if N == 1
        return (FT[0], FT[sqrt(π)])
    elseif N == 2
        a = sqrt(FT(0.5))
        return (FT[-a, a], FT[sqrt(π)/2, sqrt(π)/2])
    elseif N == 3
        a = sqrt(FT(1.5))
        w0 = FT(2) * sqrt(FT(π)) / 3
        w1 = sqrt(FT(π)) / 6
        return (FT[-a, 0, a], FT[w1, w0, w1])
    elseif N == 4
        a1 = sqrt(FT(3) - sqrt(FT(6)))  / sqrt(FT(2))
        a2 = sqrt(FT(3) + sqrt(FT(6)))  / sqrt(FT(2))
        w1 = sqrt(FT(π)) / (4 * (FT(3) - sqrt(FT(6))))
        w2 = sqrt(FT(π)) / (4 * (FT(3) + sqrt(FT(6))))
        return (FT[-a2, -a1, a1, a2], FT[w2, w1, w1, w2])
    elseif N == 5
        a1 = sqrt(FT(5) - sqrt(FT(10))) / sqrt(FT(2))
        a2 = sqrt(FT(5) + sqrt(FT(10))) / sqrt(FT(2))
        w0 = FT(8) * sqrt(FT(π)) / 15
        w1 = sqrt(FT(π)) * (FT(7) + FT(2) * sqrt(FT(10))) / 60
        w2 = sqrt(FT(π)) * (FT(7) - FT(2) * sqrt(FT(10))) / 60
        return (FT[-a2, -a1, 0, a1, a2], FT[w2, w1, w0, w1, w2])
    else
        error("Gauss-Hermite quadrature order $N not implemented. Use N ∈ {1,2,3,4,5}.")
    end
end

"""
    gauss_legendre(FT, N)

Gauss-Legendre quadrature nodes and weights for order `N` on ``[-1, 1]``.

# Arguments
- `FT`: Floating-point type
- `N::Int`: Quadrature order (1-5 supported)

# Returns
Tuple `(nodes, weights)` as `Vector{FT}`.
"""
function gauss_legendre(::Type{FT}, N::Int) where {FT}
    # Precomputed values for common orders
    if N == 1
        return (FT[0], FT[2])
    elseif N == 2
        a = one(FT) / sqrt(FT(3))
        return (FT[-a, a], FT[1, 1])
    elseif N == 3
        a = sqrt(FT(3) / FT(5))
        return (FT[-a, 0, a], FT[FT(5)/9, FT(8)/9, FT(5)/9])
    elseif N == 4
        a1 = sqrt(FT(3)/FT(7) - FT(2)/FT(7) * sqrt(FT(6)/FT(5)))
        a2 = sqrt(FT(3)/FT(7) + FT(2)/FT(7) * sqrt(FT(6)/FT(5)))
        w1 = (FT(18) + sqrt(FT(30))) / FT(36)
        w2 = (FT(18) - sqrt(FT(30))) / FT(36)
        return (FT[-a2, -a1, a1, a2], FT[w2, w1, w1, w2])
    elseif N == 5
        a1 = FT(1)/FT(3) * sqrt(FT(5) - FT(2)*sqrt(FT(10)/FT(7)))
        a2 = FT(1)/FT(3) * sqrt(FT(5) + FT(2)*sqrt(FT(10)/FT(7)))
        w0 = FT(128) / FT(225)
        w1 = (FT(322) + FT(13)*sqrt(FT(70))) / FT(900)
        w2 = (FT(322) - FT(13)*sqrt(FT(70))) / FT(900)
        return (FT[-a2, -a1, 0, a1, a2], FT[w2, w1, w0, w1, w2])
    else
        error("Gauss-Legendre quadrature order $N not implemented. Use N ∈ {1,2,3,4,5}.")
    end
end

"""
    gauss_legendre_01(FT, N)

Gauss-Legendre quadrature nodes and weights for order `N` on ``[0, 1]``.

Transformed from ``[-1,1]`` via ``x = (t+1)/2``.

# Arguments
- `FT`: Floating-point type
- `N::Int`: Quadrature order (1-5 supported)

# Returns
Tuple `(nodes, weights)` as `Vector{FT}`.
"""
function gauss_legendre_01(::Type{FT}, N::Int) where {FT}
    nodes, weights = gauss_legendre(FT, N)
    # Transform [-1,1] -> [0,1]: x = (t+1)/2, dx = dt/2
    nodes_01 = [(t + one(FT)) / FT(2) for t in nodes]
    weights_01 = [w / FT(2) for w in weights]
    return (nodes_01, weights_01)
end

#####
##### Distribution Types
#####

"""
    AbstractSGSDistribution

Abstract supertype for subgrid-scale probability distributions.

Subtypes determine the quadrature method and physical point transformation.
"""
abstract type AbstractSGSDistribution end

"""
    GaussianSGS <: AbstractSGSDistribution

Gaussian (normal) distribution for SGS fluctuations.

Uses Gauss-Hermite quadrature. Appropriate for unbounded variables.
"""
struct GaussianSGS <: AbstractSGSDistribution end

"""
    LogNormalSGS <: AbstractSGSDistribution

Log-normal distribution for positive-definite quantities.

Uses Gauss-Hermite quadrature in log-space. Ensures ``q > 0``.
"""
struct LogNormalSGS <: AbstractSGSDistribution end

"""
    BetaSGS <: AbstractSGSDistribution

Beta distribution for bounded quantities on ``[0, 1]``.

Uses Gauss-Legendre quadrature on ``[0, 1]``.
"""
struct BetaSGS <: AbstractSGSDistribution end

#####
##### Quadrature Struct
#####

"""
    SGSQuadrature{N, A, W, D} <: AbstractSGSamplingType

Subgrid-scale quadrature configuration for integrating over thermodynamic fluctuations.

# Type Parameters
- `N`: Quadrature order
- `A`: Type of quadrature nodes (`SVector`)
- `W`: Type of quadrature weights (`SVector`)
- `D`: Distribution type (`<: AbstractSGSDistribution`)

# Fields
- `a::A`: Quadrature nodes
- `w::W`: Quadrature weights
- `dist::D`: Distribution type

# Constructors

    SGSQuadrature(FT; quadrature_order=3, distribution=GaussianSGS())

Create an `SGSQuadrature` with the specified floating-point type `FT`,
quadrature order, and distribution type.
"""
struct SGSQuadrature{N, A, W, D <: AbstractSGSDistribution} <: AbstractSGSamplingType
    a::A    # quadrature points
    w::W    # quadrature weights
    dist::D # distribution type
    function SGSQuadrature(
        ::Type{FT};
        quadrature_order = 3,
        distribution::D = GaussianSGS(),
    ) where {FT, D <: AbstractSGSDistribution}
        N = quadrature_order
        a, w = get_quadrature_nodes_weights(distribution, FT, N)
        a, w = SA.SVector{N, FT}(a), SA.SVector{N, FT}(w)
        return new{N, typeof(a), typeof(w), D}(a, w, distribution)
    end
end

"""
    quadrature_order(quad::SGSQuadrature)

Return the quadrature order `N`.
"""
@inline quadrature_order(::SGSQuadrature{N}) where {N} = N

#####
##### Quadrature Nodes and Weights
#####

"""
    get_quadrature_nodes_weights(dist, FT, N)

Return quadrature nodes and weights for distribution type `dist`.

Dispatches to appropriate quadrature method based on distribution:
- `GaussianSGS`: Gauss-Hermite
- `LogNormalSGS`: Gauss-Hermite (log-space transform in `get_physical_point`)
- `BetaSGS`: Gauss-Legendre on ``[0, 1]``
"""
@inline get_quadrature_nodes_weights(::GaussianSGS, FT, N) = gauss_hermite(FT, N)

@inline function get_quadrature_nodes_weights(::LogNormalSGS, FT, N)
    # Log-normal uses Gauss-Hermite in log-space
    # Transformation is applied in get_physical_point
    gauss_hermite(FT, N)
end

@inline function get_quadrature_nodes_weights(::BetaSGS, FT, N)
    # Beta distribution on [0,1]: use Gauss-Legendre quadrature
    gauss_legendre_01(FT, N)
end

#####
##### Covariance Limiting
#####

"""
    limit_covariances(q′q′, T′T′, T′q′, q_mean, quad)

Limit covariances to ensure physical validity.

Applies two constraints:
1. ``\\sigma_q`` is bounded to prevent negative ``q_{tot}``:
   ``\\sigma_q \\leq -q_{mean} / (\\sqrt{2} \\chi_1)``
2. Cauchy-Schwarz inequality: ``|\\rho| \\leq 1``

# Arguments
- `q′q′`: Variance of total water ``\\langle q'^2 \\rangle``
- `T′T′`: Variance of temperature ``\\langle T'^2 \\rangle``
- `T′q′`: Covariance ``\\langle T' q' \\rangle``
- `q_mean`: Mean total water
- `quad`: `SGSQuadrature` struct

# Returns
Tuple `(σ_q, σ_T, corr)` of limited standard deviations and correlation.
"""
@inline function limit_covariances(q′q′, T′T′, T′q′, q_mean, quad)
    FT = typeof(q_mean)
    ε = eps(FT)
    eps_q = ε * max(ε, q_mean)
    eps_T = ε

    # Limit σ_q to prevent negative q_tot_hat
    # The most negative quadrature point is quad.a[1] (for Gauss-Hermite)
    sqrt2 = sqrt(FT(2))
    σ_q_lim = -q_mean / (sqrt2 * quad.a[1])
    σ_q = min(sqrt(q′q′), σ_q_lim)
    σ_T = sqrt(T′T′)

    # Enforce Cauchy-Schwarz inequality: |corr| ≤ 1
    _corr = T′q′ / max(σ_q, eps_q)
    corr = clamp(_corr / max(σ_T, eps_T), -one(FT), one(FT))

    return (σ_q, σ_T, corr)
end

#####
##### Physical Point Computation
#####

"""
    get_physical_point(dist, χ1, χ2, μ_q, μ_T, σ_q, σ_T, corr)

Transform quadrature points ``(\\chi_1, \\chi_2)`` to physical space ``(T, q)``.

For correlated bivariate Gaussian (`GaussianSGS`):
```math
q = \\mu_q + \\sqrt{2} \\sigma_q \\chi_1
```
```math
T = \\mu_T + \\sqrt{2} \\sigma_T (\\rho \\chi_1 + \\sqrt{1-\\rho^2} \\chi_2)
```

# Arguments
- `dist`: Distribution type (`GaussianSGS`, `LogNormalSGS`, or `BetaSGS`)
- `χ1`, `χ2`: Quadrature points
- `μ_q`, `μ_T`: Mean values
- `σ_q`, `σ_T`: Standard deviations
- `corr`: Correlation coefficient

# Returns
Tuple `(T_hat, q_hat)` of physical values.
"""
@inline function get_physical_point(::GaussianSGS, χ1, χ2, μ_q, μ_T, σ_q, σ_T, corr)
    FT = typeof(μ_q)
    sqrt2 = sqrt(FT(2))
    
    # Conditional mean and std for T given q
    σ_c = sqrt(max(one(FT) - corr^2, zero(FT))) * σ_T
    μ_c = μ_T + sqrt2 * corr * σ_T * χ1
    
    T_hat = μ_c + sqrt2 * σ_c * χ2
    q_hat = max(zero(FT), μ_q + sqrt2 * σ_q * χ1)
    
    return (T_hat, q_hat)
end

@inline function get_physical_point(::LogNormalSGS, χ1, χ2, μ_q, μ_T, σ_q, σ_T, corr)
    FT = typeof(μ_q)
    sqrt2 = sqrt(FT(2))
    ε = eps(FT)
    
    # T is Gaussian
    σ_c = sqrt(max(one(FT) - corr^2, zero(FT))) * σ_T
    μ_c = μ_T + sqrt2 * corr * σ_T * χ1
    T_hat = μ_c + sqrt2 * σ_c * χ2
    
    # q is log-normal: compute log-space params from mean/variance
    # For log-normal: μ_ln = log(μ²/√(σ² + μ²)), σ_ln = √log(1 + σ²/μ²)
    q_hat = ifelse(
        μ_q > ε && σ_q > zero(FT),
        begin
            σ_ln = sqrt(log(one(FT) + (σ_q / μ_q)^2))
            μ_ln = log(μ_q) - σ_ln^2 / 2
            exp(μ_ln + sqrt2 * σ_ln * χ1)
        end,
        μ_q,
    )
    
    return (T_hat, q_hat)
end

# BetaSGS: quadrature points χ1 are on [0,1], map to q ∈ [q_min, q_max]
@inline function get_physical_point(::BetaSGS, χ1, χ2, μ_q, μ_T, σ_q, σ_T, corr)
    FT = typeof(μ_q)
    sqrt2 = sqrt(FT(2))
    
    # T is still Gaussian
    σ_c = sqrt(max(one(FT) - corr^2, zero(FT))) * σ_T
    μ_c = μ_T + sqrt2 * corr * σ_T * χ1
    T_hat = μ_c + sqrt2 * σ_c * χ2
    
    # q is bounded: χ1 ∈ [0,1] maps linearly to [q_min, q_max]
    # Use ±2σ as bounds (captures 95% of distribution)
    q_min = max(zero(FT), μ_q - FT(2) * σ_q)
    q_max = μ_q + FT(2) * σ_q
    q_hat = q_min + χ1 * (q_max - q_min)
    
    return (T_hat, q_hat)
end

#####
##### Quadrature Integration
#####

"""
    sum_over_quadrature_points(f, get_x_hat, quad)

Compute the weighted sum of `f(T, q)` over quadrature points.

Approximates the integral:
```math
\\int\\int f(T, q) P(T, q) \\, dT \\, dq \\approx \\sum_{i,j} w_i w_j f(T_{ij}, q_{ij}) / \\pi
```

# Arguments
- `f`: Point-wise function `(T, q) -> result`
- `get_x_hat`: Function `(χ1, χ2) -> (T_hat, q_hat)` transforming quadrature points
- `quad`: `SGSQuadrature` struct

# Returns
Weighted sum with the same type as `f(T, q)`.
"""
function sum_over_quadrature_points(f, get_x_hat, quad)
    χ = quad.a
    weights = quad.w
    N = quadrature_order(quad)
    FT = eltype(χ)
    
    # Initialize with zero of correct type
    T = typeof(f(get_x_hat(χ[1], χ[1])...))
    outer_sum = recursive_zero(f(get_x_hat(χ[1], χ[1])...))
    
    inv_sqrt_pi = one(FT) / sqrt(FT(π))
    
    @inbounds for i in 1:N
        inner_sum = recursive_zero(outer_sum)
        for j in 1:N
            x_hat = get_x_hat(χ[i], χ[j])
            weighted = recursive_mul(f(x_hat...), weights[j] * inv_sqrt_pi)
            inner_sum = recursive_add(inner_sum, weighted)
        end
        outer_sum = recursive_add(outer_sum, recursive_mul(inner_sum, weights[i] * inv_sqrt_pi))
    end
    
    return outer_sum
end

"""
    integrate_over_sgs(f, quad, μ_q, μ_T, q′q′, T′T′, T′q′)

Integrate `f(T, q)` over SGS fluctuations.

Convenience function that handles covariance limiting and physical point
transformation internally.

# Arguments
- `f`: Point-wise function `(T, q) -> result`
- `quad`: `SGSQuadrature` struct
- `μ_q`, `μ_T`: Mean values
- `q′q′`, `T′T′`, `T′q′`: Covariances

# Returns
Weighted sum with the same type as `f(T, q)`.
"""
function integrate_over_sgs(f, quad, μ_q, μ_T, q′q′, T′T′, T′q′)
    σ_q, σ_T, corr = limit_covariances(q′q′, T′T′, T′q′, μ_q, quad)
    
    get_x_hat(χ1, χ2) = get_physical_point(quad.dist, χ1, χ2, μ_q, μ_T, σ_q, σ_T, corr)
    
    return sum_over_quadrature_points(f, get_x_hat, quad)
end
