"""
    SGS Quadrature Utilities

Subgrid-scale (SGS) quadrature infrastructure for integrating point-wise functions
over thermodynamic fluctuations. Supports multiple distribution types and provides
reusable utilities for cloud fraction, microphysics tendencies, and other SGS diagnostics.
"""

import StaticArrays as SA
import Thermodynamics as TD
import ClimaCore.RecursiveApply: rzero, ⊞, ⊠
import UnrolledUtilities: unrolled_reduce


# ============================================================================
# Gauss-Hermite Quadrature
# ============================================================================

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
        return (FT[-a, a], FT[sqrt(π) / 2, sqrt(π) / 2])
    elseif N == 3
        a = sqrt(FT(1.5))
        w0 = FT(2) * sqrt(FT(π)) / 3
        w1 = sqrt(FT(π)) / 6
        return (FT[-a, 0, a], FT[w1, w0, w1])
    elseif N == 4
        a1 = sqrt(FT(3) - sqrt(FT(6))) / sqrt(FT(2))
        a2 = sqrt(FT(3) + sqrt(FT(6))) / sqrt(FT(2))
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
        return (FT[-a, 0, a], FT[FT(5) / 9, FT(8) / 9, FT(5) / 9])
    elseif N == 4
        a1 = sqrt(FT(3) / FT(7) - FT(2) / FT(7) * sqrt(FT(6) / FT(5)))
        a2 = sqrt(FT(3) / FT(7) + FT(2) / FT(7) * sqrt(FT(6) / FT(5)))
        w1 = (FT(18) + sqrt(FT(30))) / FT(36)
        w2 = (FT(18) - sqrt(FT(30))) / FT(36)
        return (FT[-a2, -a1, a1, a2], FT[w2, w1, w1, w2])
    elseif N == 5
        a1 = FT(1) / FT(3) * sqrt(FT(5) - FT(2) * sqrt(FT(10) / FT(7)))
        a2 = FT(1) / FT(3) * sqrt(FT(5) + FT(2) * sqrt(FT(10) / FT(7)))
        w0 = FT(128) / FT(225)
        w1 = (FT(322) + FT(13) * sqrt(FT(70))) / FT(900)
        w2 = (FT(322) - FT(13) * sqrt(FT(70))) / FT(900)
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

# ============================================================================
# Distribution Types
# ============================================================================

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
    GridMeanSGS <: AbstractSGSDistribution

Grid-mean-only "quadrature" - evaluates at the grid mean without variance.

Uses a single quadrature point at ``(χ_1, χ_2) = (0, 0)`` with weight 1.
This is the 0th-order option: same code path as full quadrature, but only
evaluates at the grid mean. Use when SGS fluctuations should be ignored.
"""
struct GridMeanSGS <: AbstractSGSDistribution end

# SGS distribution types are scalar arguments in @. broadcast expressions
Base.broadcastable(x::AbstractSGSDistribution) = tuple(x)

# ============================================================================
# Quadrature Struct
# ============================================================================

"""
    SGSQuadrature{N, A, W, D, FT} <: AbstractSGSamplingType

Subgrid-scale quadrature configuration for integrating over thermodynamic fluctuations.

# Type Parameters
- `N`: Quadrature order
- `A`: Type of quadrature nodes (`SVector`)
- `W`: Type of quadrature weights (`SVector`)
- `D`: Distribution type (`<: AbstractSGSDistribution`)
- `FT`: Floating-point type

# Fields
- `a::A`: Quadrature nodes
- `w::W`: Quadrature weights
- `dist::D`: Distribution type
- `T_min::FT`: Minimum temperature for physical validity [K]. Used to clamp
  sampled temperatures in `get_physical_point` to prevent domain errors in
  thermodynamics calculations. Typically set from `Thermodynamics.Parameters.T_min`.

# Constructors

    SGSQuadrature(FT; quadrature_order=2, distribution=GaussianSGS(), T_min=150.0)

Create an `SGSQuadrature` with the specified floating-point type `FT`,
quadrature order, distribution type, and minimum temperature.

The T-q correlation coefficient is provided externally via `correlation_Tq(params)`
rather than being stored in this struct.

The minimum temperature `T_min` should be set from `Thermodynamics.Parameters.T_min(thermo_params)`
to ensure consistency with the thermodynamics package constraints.
"""
struct SGSQuadrature{N, A, W, D <: AbstractSGSDistribution, FT} <: AbstractSGSamplingType
    a::A             # quadrature points
    w::W             # quadrature weights
    dist::D          # distribution type
    T_min::FT        # minimum temperature for physical validity [K]
    function SGSQuadrature(
        ::Type{FT};
        quadrature_order = 3,
        distribution::D = GaussianSGS(),
        T_min = FT(150),  # Reasonable default for atmospheric applications
    ) where {FT, D <: AbstractSGSDistribution}
        # GridMeanSGS always uses N=1 (single point at origin)
        N = distribution isa GridMeanSGS ? 1 : quadrature_order
        a, w = get_quadrature_nodes_weights(distribution, FT, N)
        a, w = SA.SVector{N, FT}(a), SA.SVector{N, FT}(w)
        return new{N, typeof(a), typeof(w), D, FT}(
            a,
            w,
            distribution,
            FT(T_min),
        )
    end
end

"""
    quadrature_order(quad::SGSQuadrature)

Return the quadrature order `N`.
"""
@inline quadrature_order(::SGSQuadrature{N}) where {N} = N

# ============================================================================
# Quadrature Nodes and Weights
# ============================================================================

"""
    get_quadrature_nodes_weights(dist, FT, N)

Return quadrature nodes and weights for distribution type `dist`.

Dispatches to appropriate quadrature method based on distribution:
- `GaussianSGS`: Gauss-Hermite
- `LogNormalSGS`: Gauss-Hermite (log-space transform in `get_physical_point`)
"""
@inline get_quadrature_nodes_weights(::GaussianSGS, FT, N) = gauss_hermite(FT, N)

@inline function get_quadrature_nodes_weights(::LogNormalSGS, FT, N)
    # Log-normal uses Gauss-Hermite in log-space
    # Transformation is applied in get_physical_point
    gauss_hermite(FT, N)
end

@inline function get_quadrature_nodes_weights(::GridMeanSGS, FT, N)
    # Grid-mean-only: single point at origin with weight 1
    # Ignores N - always returns a single quadrature point
    ([FT(0)], [FT(1)])
end

# ============================================================================
# Helper for covariance Transformation
# ============================================================================

"""
    ∂T_∂θ_li(thermo_params, T, θ_li, q_liq, q_ice, q_tot, ρ)

Compute the Jacobian ∂T/∂θ_li for transforming θ_li fluctuations to T fluctuations.

# Derivation

Starting from `θ_li = T/Π × f(q_c)` where `f` encodes the condensate effect,
and taking the derivative at constant pressure:

For **unsaturated** air (q_c = 0):
```math
\\frac{\\partial T}{\\partial \\theta_{li}} = \\Pi
```

For **saturated** air, `q_c = q_{tot} - q_{sat}(T)` varies with T:
```math
\\frac{\\partial q_c}{\\partial \\theta_{li}} = -\\frac{\\partial q_{sat}}{\\partial T} \\frac{\\partial T}{\\partial \\theta_{li}}
```
Leading to:
```math
\\frac{\\partial T}{\\partial \\theta_{li}} = \\frac{\\Pi}{1 + \\Pi \\frac{L_v}{c_p} \\frac{\\partial q_{sat}}{\\partial T}}
```

The denominator > 1 in saturated conditions, so ∂T/∂θ_li < Π: latent heat
release partially buffers temperature changes (the "moist adiabatic" effect).

# Clausius-Clapeyron Approximation

```math
\\frac{\\partial q_{sat}}{\\partial T} \\approx q_{sat} \\frac{L_v}{R_v T^2}
```

# Usage
```julia
ᶜθ_li = @. lazy(TD.liquid_ice_pottemp(thp, ᶜT, ᶜρ, ᶜq_tot, ᶜq_liq, ᶜq_ice))
ᶜ∂T_∂θ = @. lazy(∂T_∂θ_li(thp, ᶜT, ᶜθ_li, ᶜq_liq, ᶜq_ice, ᶜq_tot, ᶜρ))
ᶜT′T′ = @. lazy(ᶜ∂T_∂θ^2 * ᶜθ′θ′)
ᶜT′q′ = @. lazy(ᶜ∂T_∂θ * ᶜθ′q′)
```

# Arguments
- `thermo_params`: Thermodynamics parameters (for L_v, c_p, R_v)
- `T`: Temperature [K]
- `θ_li`: Liquid-ice potential temperature [K]
- `q_liq`: Liquid water specific humidity [kg/kg]
- `q_ice`: Ice specific humidity [kg/kg]
- `q_tot`: Total water specific humidity [kg/kg]
- `ρ`: Air density [kg/m³]

# Returns
Derivative ∂T/∂θ_li [dimensionless]
"""
@inline function ∂T_∂θ_li(thermo_params, T, θ_li, q_liq, q_ice, q_tot, ρ)
    FT = typeof(T)
    L_v = TD.Parameters.LH_v0(thermo_params)
    c_p = TD.Parameters.cp_d(thermo_params)
    R_v = TD.Parameters.R_v(thermo_params)

    q_cond = q_liq + q_ice
    Π = T / max(θ_li, eps(FT))

    q_sat = TD.q_vap_saturation(thermo_params, T, ρ)
    is_saturated = q_tot > q_sat * (one(FT) + eps(FT))

    # Clausius-Clapeyron: dq_sat/dT ≈ q_sat × L_v / (R_v × T²)
    dqsat_dT = ifelse(
        is_saturated,
        q_sat * L_v / (R_v * max(T, eps(FT))^2),
        zero(FT),
    )

    denominator = one(FT) + dqsat_dT * L_v / c_p
    moist_correction = one(FT) + (L_v * q_cond) / (c_p * max(T, eps(FT)) * denominator)

    return Π * moist_correction
end

# ============================================================================
# T-q Correlation Coefficient
# ============================================================================

"""
    correlation_Tq(params)

Return the correlation coefficient between temperature and total water
perturbations for SGS quadrature.

Reads `Tq_correlation_coefficient` from the ClimaParams parameter set.
Atmospheric T-q correlations on sub-100 km scales relevant to GCM grid cells
are typically positive (warm air holds more moisture) with values around 0.6.

# Returns
Correlation coefficient corr(T′, q′) ∈ [-1, 1].
"""
@inline correlation_Tq(params) = CAP.Tq_correlation_coefficient(params)

# ============================================================================
# Covariance Limiting
# ============================================================================

"""
    limit_covariances(q′q′, T′T′, corr_Tq, q_mean, quad)

Limit variances and correlation to ensure physical validity.

Applies two constraints:
1. ``\\sigma_q`` is bounded to prevent negative ``q_{tot}``:
   ``\\sigma_q \\leq -q_{mean} / (\\sqrt{2} \\chi_1)``
2. Cauchy-Schwarz inequality: ``|\\rho| \\leq 1`` (enforced via `clamp`)

# Arguments
- `q′q′`: Variance of total water ``\\langle q'^2 \\rangle``
- `T′T′`: Variance of temperature ``\\langle T'^2 \\rangle``
- `corr_Tq`: Correlation coefficient ``\\rho(T', q')``
- `q_mean`: Mean total water
- `quad`: `SGSQuadrature` struct

# Returns
Tuple `(σ_q, σ_T, corr)` of limited standard deviations and correlation.
"""
@inline function limit_covariances(q′q′, T′T′, corr_Tq, q_mean, quad)
    FT = typeof(q_mean)

    # Limit σ_q to prevent negative q_tot_hat
    # For Gauss-Hermite, the most negative quadrature point has abscissa quad.a[1] < 0
    # For order=1, a[1]=0, so no limiting is needed (no variance displacement)
    sqrt2 = sqrt(FT(2))
    a_min = quad.a[1]  # Most negative abscissa (or zero for order=1)

    # Branchless limiting: q_hat = μ_q + sqrt2 * σ_q * a_min >= 0
    # => σ_q <= -μ_q / (sqrt2 * a_min)  when a_min < 0
    σ_q_raw = sqrt(max(FT(0), q′q′))
    σ_q_lim = ifelse(a_min < FT(0), -q_mean / (sqrt2 * a_min), σ_q_raw)
    σ_q = min(σ_q_raw, σ_q_lim)
    σ_T = sqrt(max(FT(0), T′T′))

    # Enforce Cauchy-Schwarz inequality: |corr| ≤ 1
    corr = clamp(corr_Tq, -one(FT), one(FT))

    return (σ_q, σ_T, corr)
end

# ============================================================================
# Physical Point Computation
# ============================================================================

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
- `dist`: Distribution type (`GaussianSGS` or `LogNormalSGS`)
- `χ1`, `χ2`: Quadrature points
- `μ_q`, `μ_T`: Mean values
- `σ_q`, `σ_T`: Standard deviations
- `corr`: Correlation coefficient

# Returns
Tuple `(T_hat, q_hat)` of physical values.
"""
@inline function get_physical_point(::GaussianSGS, χ1, χ2, μ_q, μ_T, σ_q, σ_T, corr, T_min)
    FT = typeof(μ_q)
    sqrt2 = sqrt(FT(2))

    # Conditional mean and std for T given q
    σ_c = sqrt(max(one(FT) - corr^2, zero(FT))) * σ_T
    μ_c = μ_T + sqrt2 * corr * σ_T * χ1

    # Clamp both T and q to physically valid ranges
    # T must be >= T_min for thermodynamics calculations (saturation vapor pressure uses log(T))
    # T_min comes from thermo_params for consistency with Thermodynamics.jl constraints
    T_hat = max(T_min, μ_c + sqrt2 * σ_c * χ2)
    q_hat = max(zero(FT), μ_q + sqrt2 * σ_q * χ1)

    return (T_hat, q_hat)
end

@inline function get_physical_point(::LogNormalSGS, χ1, χ2, μ_q, μ_T, σ_q, σ_T, corr, T_min)
    FT = typeof(μ_q)
    sqrt2 = sqrt(FT(2))
    ε = eps(FT)

    # Step 1: Generate correlated Gaussian variables using copula approach
    # This ensures proper correlation structure between T and q
    z_q = χ1
    z_T = corr * χ1 + sqrt(max(zero(FT), one(FT) - corr^2)) * χ2

    # Step 2: Transform z_q to log-normal for q
    # For log-normal: σ_ln = √log(1 + σ²/μ²), μ_ln = log(μ) - σ_ln²/2
    ratio = σ_q / max(μ_q, ε)
    σ_ln = sqrt(log(one(FT) + ratio^2))
    μ_ln = log(max(μ_q, ε)) - σ_ln^2 / 2
    q_lognormal = exp(μ_ln + sqrt2 * σ_ln * z_q)

    # Use log-normal only if both mean and variance are positive
    use_lognormal = (μ_q > ε) & (σ_q > zero(FT))
    q_hat = ifelse(use_lognormal, q_lognormal, μ_q)

    # Step 3: Keep Gaussian for T using correlated z_T, clamped to T_min
    # T_min comes from thermo_params for consistency with Thermodynamics.jl constraints
    T_hat = max(T_min, μ_T + sqrt2 * σ_T * z_T)

    return (T_hat, q_hat)
end



# GridMeanSGS: evaluates only at the grid mean, ignoring variance
@inline function get_physical_point(::GridMeanSGS, χ1, χ2, μ_q, μ_T, σ_q, σ_T, corr, T_min)
    # Return grid mean directly, ignoring quadrature points, variance, and T_min
    # χ1, χ2, σ_q, σ_T, corr, T_min are all ignored
    (μ_T, μ_q)
end

# ============================================================================
# Physical Point Transform Functor
# ============================================================================

"""
    PhysicalPointTransform

GPU-safe functor for transforming quadrature points to physical space.

Replaces closures like `(χ1, χ2) -> get_physical_point(dist, χ1, χ2, ...)` with
a struct that captures all necessary parameters, avoiding heap allocations.

Field order matches return order `(T, q)` for consistency.

# Fields
- `dist`: Distribution type (`GaussianSGS` or `LogNormalSGS`)
- `μ_T`: Mean temperature [K]
- `μ_q`: Mean specific humidity [kg/kg]
- `σ_T`: Standard deviation of T [K]
- `σ_q`: Standard deviation of q [kg/kg]
- `corr`: Correlation coefficient [-1, 1]
- `T_min`: Minimum temperature clamp [K]
"""
struct PhysicalPointTransform{D, FT}
    dist::D
    μ_T::FT
    μ_q::FT
    σ_T::FT
    σ_q::FT
    corr::FT
    T_min::FT
end

@inline function (t::PhysicalPointTransform)(χ1, χ2)
    return get_physical_point(t.dist, χ1, χ2, t.μ_q, t.μ_T, t.σ_q, t.σ_T, t.corr, t.T_min)
end

# ============================================================================
# Quadrature Integration
# ============================================================================

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
function sum_over_quadrature_points(f, get_x_hat, quad::SGSQuadrature{N}) where {N}
    χ = quad.a
    weights = quad.w
    FT = eltype(χ)

    inv_sqrt_pi = one(FT) / sqrt(FT(π))

    # Use ntuple for compile-time unrolling (type-stable)
    outer_sum = ntuple(Val(N)) do i
        inner_sum = ntuple(Val(N)) do j
            x_hat = get_x_hat(χ[i], χ[j])
            f(x_hat...) ⊠ (weights[j] * inv_sqrt_pi)
        end
        unrolled_reduce(⊞, inner_sum) ⊠ (weights[i] * inv_sqrt_pi)
    end

    return unrolled_reduce(⊞, outer_sum)
end

"""
    integrate_over_sgs(f, quad, μ_q, μ_T, q′q′, T′T′, corr_Tq)

Integrate `f(T, q)` over SGS fluctuations.

Convenience function that handles variance limiting and physical point
transformation internally.

# Arguments
- `f`: Point-wise function `(T, q) -> result`
- `quad`: `SGSQuadrature` struct
- `μ_q`, `μ_T`: Mean values
- `q′q′`, `T′T′`: Variances
- `corr_Tq`: Correlation coefficient ``ρ(T', q')``

# Returns
Weighted sum with the same type as `f(T, q)`.
"""
function integrate_over_sgs(f, quad, μ_q, μ_T, q′q′, T′T′, corr_Tq)
    σ_q, σ_T, corr = limit_covariances(q′q′, T′T′, corr_Tq, μ_q, quad)

    # Use functor instead of closure to avoid heap allocations
    # Field order is (T, q) to match return order of get_physical_point
    transform = PhysicalPointTransform(quad.dist, μ_T, μ_q, σ_T, σ_q, corr, oftype(μ_T, quad.T_min))

    return sum_over_quadrature_points(f, transform, quad)
end

"""
    integrate_over_sgs(f, ::GridMeanSGS, μ_q, μ_T, q′q′, T′T′, corr_Tq)

Simplified grid-mean integration: evaluates `f(μ_T, μ_q)` directly.

This allows using `GridMeanSGS()` directly without wrapping in `SGSQuadrature`,
avoiding the need to extract FT from the space. Variances and correlation are ignored.
"""
@inline function integrate_over_sgs(f, ::GridMeanSGS, μ_q, μ_T, q′q′, T′T′, corr_Tq)
    return f(μ_T, μ_q)
end
