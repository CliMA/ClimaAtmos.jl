# ============================================================================
# SGS Quadrature Utilities
# ============================================================================
# Subgrid-scale (SGS) quadrature infrastructure for integrating point-wise
# functions over thermodynamic fluctuations. Supports several distribution types
# and provides the utilities shared by cloud fraction, microphysics tendencies,
# and other SGS diagnostics.

import StaticArrays as SA
import Thermodynamics as TD
import ClimaCore.RecursiveApply: rzero, ⊞, ⊠
import UnrolledUtilities: unrolled_reduce

# ============================================================================
# Gauss-Hermite Quadrature
# ============================================================================

"""
    gauss_hermite(FT, N)

Return precomputed Gauss-Hermite quadrature nodes and weights of order `N`.

Nodes are roots of the physicists' Hermite polynomial ``H_N(x)``; weights are the
standard Gauss-Hermite weights for integration against ``e^{-x^2}``.

# Arguments

  - `FT`: Floating-point type.
  - `N::Int`: Quadrature order; `N ∈ {1, 2, 3, 4, 5}`, otherwise an error is thrown.

# Returns

Tuple `(nodes, weights)` of `Vector{FT}`.
"""
function gauss_hermite(::Type{FT}, N::Int) where {FT}
    # Precomputed values for common orders
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
    gauss_legendre_01(FT, N)

Return Gauss-Legendre quadrature nodes and weights of order `N` on ``[0, 1]``.

Mapped from the standard ``[-1, 1]`` rule via ``x = (t + 1)/2``, ``w_{01} = w/2``.

# Arguments

  - `FT`: Floating-point type.
  - `N::Int`: Quadrature order; `N ∈ {1, 2, 3, 4, 5}`, otherwise an error is thrown.

# Returns

Tuple `(nodes, weights)` of `Vector{FT}`.
"""
function gauss_legendre_01(::Type{FT}, N::Int) where {FT}
    half = FT(1) / FT(2)
    if N == 1
        return (FT[half], FT[1])
    elseif N == 2
        a = one(FT) / sqrt(FT(3))
        return (FT[(1 - a) * half, (1 + a) * half], FT[half, half])
    elseif N == 3
        a = sqrt(FT(3) / FT(5))
        return (
            FT[(1 - a) * half, half, (1 + a) * half],
            FT[FT(5) / 18, FT(4) / 9, FT(5) / 18],
        )
    elseif N == 4
        a1 = sqrt(FT(3) / FT(7) - FT(2) / FT(7) * sqrt(FT(6) / FT(5)))
        a2 = sqrt(FT(3) / FT(7) + FT(2) / FT(7) * sqrt(FT(6) / FT(5)))
        w1 = (FT(18) + sqrt(FT(30))) / FT(36)
        w2 = (FT(18) - sqrt(FT(30))) / FT(36)
        return (
            FT[(1 - a2) * half, (1 - a1) * half, (1 + a1) * half, (1 + a2) * half],
            FT[w2 * half, w1 * half, w1 * half, w2 * half],
        )
    elseif N == 5
        a1 = FT(1) / FT(3) * sqrt(FT(5) - FT(2) * sqrt(FT(10) / FT(7)))
        a2 = FT(1) / FT(3) * sqrt(FT(5) + FT(2) * sqrt(FT(10) / FT(7)))
        w0 = FT(128) / FT(225)
        w1 = (FT(322) + FT(13) * sqrt(FT(70))) / FT(900)
        w2 = (FT(322) - FT(13) * sqrt(FT(70))) / FT(900)
        return (
            FT[(1 - a2) * half, (1 - a1) * half, half, (1 + a1) * half, (1 + a2) * half],
            FT[w2 * half, w1 * half, w0 * half, w1 * half, w2 * half],
        )
    else
        error("Gauss-Legendre quadrature order $N not implemented. Use N ∈ {1,2,3,4,5}.")
    end
end

# ============================================================================
# Distribution Types
# ============================================================================

"""
    AbstractSGSDistribution

Joint subgrid-scale distribution of `(T, q)` assumed by the quadrature.

The distribution type selects how specific humidity is sampled from the quadrature
nodes; temperature is always Gaussian. Each subtype has a corresponding transform
functor built by [`create_physical_transform`](@ref).

Subtypes:

  - [`GaussianSGS`](@ref): correlated bivariate Gaussian.
  - [`LogNormalSGS`](@ref): log-normal `q`, Gaussian `T`.
  - [`GridMeanSGS`](@ref): degenerate, grid-mean-only.
"""
abstract type AbstractSGSDistribution end

"""
    GaussianSGS <: AbstractSGSDistribution

Bivariate Gaussian SGS distribution of `(T, q)`.

Humidity is sampled first, and temperature is drawn from its distribution
conditional on the sampled humidity, which reproduces the requested correlation.
Sampled `q` is clamped to `[0, q_max]` and sampled `T` is floored at `T_min`, so
extreme nodes cannot leave the physical domain.
"""
struct GaussianSGS <: AbstractSGSDistribution end

"""
    LogNormalSGS <: AbstractSGSDistribution

Log-normal SGS distribution for specific humidity, Gaussian for temperature.

Sampling `q` in log space makes it positive-definite by construction, so the
lower-tail truncation of `GaussianSGS` does not arise. The `T`-`q` correlation is
imposed with a Gaussian copula on the underlying normal variates. The log-normal
parameters degenerate for a vanishing mean or variance, in which case sampling
falls back to the mean humidity.
"""
struct LogNormalSGS <: AbstractSGSDistribution end



"""
    GridMeanSGS <: AbstractSGSDistribution

Degenerate SGS distribution: all mass at the grid mean.

A single node at ``(χ_1, χ_2) = (0, 0)`` with weight ``\\sqrt{\\pi}``, chosen so
that the ``1/\\pi`` normalization of the two-dimensional quadrature returns the
integrand evaluated at the mean. This is the zeroth-order option, taking the same
code path as full quadrature; use it when SGS fluctuations are to be ignored.
"""
struct GridMeanSGS <: AbstractSGSDistribution end

# SGS distribution types are scalar arguments in @. broadcast expressions
Base.broadcastable(x::AbstractSGSDistribution) = tuple(x)

# ============================================================================
# Quadrature Struct
# ============================================================================

"""
    SGSQuadrature{N, A, W, D, FT} <: AbstractSGSamplingType

Subgrid-scale quadrature configuration for integrating over thermodynamic
fluctuations of `(T, q_tot)`.

`N` is the quadrature order, `A` and `W` the node and weight `SVector` types, `D`
the [`AbstractSGSDistribution`](@ref) subtype, and `FT` the floating-point type.
The two-dimensional rule evaluates `N²` points.

# Fields

  - `a::A`: Quadrature nodes in standardized variables [-].
  - `w::W`: Quadrature weights [-].
  - `dist::D`: SGS distribution type.
  - `T_min::FT`: Floor applied to sampled temperatures [K], which keeps extreme
    nodes out of the domain-error region of the thermodynamics routines. Set from
    the ClimaParams `temperature_minimum`.
  - `q_max::FT`: Cap applied to sampled specific humidity [kg/kg], which keeps
    extreme supersaturation at a node from driving unphysically low temperatures
    through excessive latent heat. Set from the ClimaParams
    `specific_humidity_maximum`.

# Constructor

    SGSQuadrature(
        FT; quadrature_order = 3, distribution = GaussianSGS(),
        T_min = FT(150), q_max = FT(0.05),
    )

Build the quadrature for floating-point type `FT`. `GridMeanSGS` always overrides
`quadrature_order` to `N = 1`. The `T`-`q` correlation coefficient is deliberately
not stored here; it is supplied per call via `correlation_Tq(params)`.
"""
struct SGSQuadrature{N, A, W, D <: AbstractSGSDistribution, FT} <: AbstractSGSamplingType
    a::A             # quadrature points
    w::W             # quadrature weights
    dist::D          # distribution type
    T_min::FT        # minimum temperature for physical validity [K]
    q_max::FT        # maximum specific humidity [kg/kg]
    function SGSQuadrature(
        ::Type{FT};
        quadrature_order = 3,
        distribution::D = GaussianSGS(),
        T_min = FT(150),   # Reasonable default for atmospheric applications
        q_max = FT(0.05),  # Maximum humidity: 50 g/kg (above any physical value)
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
            FT(q_max),
        )
    end
end

"""
    quadrature_order(quad::SGSQuadrature)

Return the quadrature order `N`, so that the two-dimensional rule uses `N²` points.
"""
@inline quadrature_order(::SGSQuadrature{N}) where {N} = N

# ============================================================================
# Quadrature Nodes and Weights
# ============================================================================

"""
    get_quadrature_nodes_weights(dist, FT, N)

Return the quadrature nodes and weights appropriate to the SGS distribution `dist`.

  - `GaussianSGS`: Gauss-Hermite of order `N`.
  - `LogNormalSGS`: Gauss-Hermite of order `N`; the log-space transform is applied
    later, in the transform functor built by [`create_physical_transform`](@ref).
  - `GridMeanSGS`: the single node at the origin with weight ``\\sqrt{\\pi}``,
    which cancels the ``1/\\pi`` normalization in
    `sum_over_quadrature_points`.

Called from the [`SGSQuadrature`](@ref) constructor.
"""
@inline get_quadrature_nodes_weights(::GaussianSGS, FT, N) = gauss_hermite(FT, N)

@inline function get_quadrature_nodes_weights(::LogNormalSGS, FT, N)
    # Log-normal uses Gauss-Hermite in log-space
    # Transformation is applied in get_physical_point
    gauss_hermite(FT, N)
end

@inline function get_quadrature_nodes_weights(::GridMeanSGS, FT, N)
    # Grid-mean-only: single point at origin with weight sqrt(π)
    # The weight must be sqrt(π) because sum_over_quadrature_points divides by π
    # (assuming 2D quadrature), so (sqrt(π))^2 / π = 1.
    ([FT(0)], [sqrt(FT(π))])
end

# ============================================================================
# Helper for covariance Transformation
# ============================================================================

"""
    ∂T_∂θ_li(thermo_params, T, θ_li, q_liq, q_ice, q_tot, ρ)

Compute ∂T/∂θ_li, the factor converting θ_li fluctuations into T fluctuations.

The transformation is the local ratio ``\\Pi = T/\\theta_{li}`` times a moist
amplification factor:

```math
\\frac{\\partial T}{\\partial \\theta_{li}} = \\Pi \\left(1 +
  \\frac{L_v q_c}{c_p T \\left(1 + \\frac{L_v}{c_p}
  \\frac{\\partial q_{sat}}{\\partial T}\\right)}\\right)
```

Because `θ_li` has the latent heat of the condensate `q_c = q_liq + q_ice` already
removed, recovering `T` adds it back, so the factor exceeds `Π` wherever condensate
is present. In saturated air the condensate itself responds to temperature through
``q_c = q_{tot} - q_{sat}(T)``, and the Clausius-Clapeyron denominator
``\\partial q_{sat}/\\partial T \\approx q_{sat} L_v /(R_v T^2)`` damps that
amplification. The derivative is treated as unsaturated (denominator 1) wherever
`q_tot < q_sat`. For condensate-free air the factor reduces to `Π`.

The latent heat and heat capacity are the fixed reference values `LH_v0` and
`cp_d`, so this is a linearization, adequate for mapping SGS (co)variances between
the θ_li and T bases.

# Arguments

  - `thermo_params`: Thermodynamics parameters, supplying `L_v`, `c_p`, `R_v`.
  - `T`: Temperature [K].
  - `θ_li`: Liquid-ice potential temperature [K].
  - `q_liq`: Liquid water specific humidity [kg/kg].
  - `q_ice`: Ice specific humidity [kg/kg].
  - `q_tot`: Total water specific humidity [kg/kg].
  - `ρ`: Air density [kg/m³].

# Returns

∂T/∂θ_li [-].

# Examples

```julia
ᶜθ_li = @. lazy(TD.liquid_ice_pottemp(thp, ᶜT, ᶜρ, ᶜq_tot, ᶜq_lcl, ᶜq_icl))
ᶜ∂T_∂θ = @. lazy(∂T_∂θ_li(thp, ᶜT, ᶜθ_li, ᶜq_lcl, ᶜq_icl, ᶜq_tot, ᶜρ))
ᶜT′T′ = @. lazy(ᶜ∂T_∂θ^2 * ᶜθ′θ′)
```
"""
@inline function ∂T_∂θ_li(thermo_params, T, θ_li, q_liq, q_ice, q_tot, ρ)
    FT = typeof(T)
    L_v = TD.Parameters.LH_v0(thermo_params)
    c_p = TD.Parameters.cp_d(thermo_params)
    R_v = TD.Parameters.R_v(thermo_params)

    q_cond = q_liq + q_ice
    Π = T / max(θ_li, eps(FT))

    q_sat = TD.q_vap_saturation(thermo_params, T, ρ)
    is_saturated = q_tot >= q_sat

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

Return the SGS correlation coefficient between temperature and total water
perturbations.

Reads `Tq_correlation_coefficient` from the ClimaParams parameter set. Atmospheric
`T`-`q` correlations on the sub-100 km scales relevant to GCM grid cells are
typically positive, around 0.6, since warm air holds more moisture.

# Returns

Correlation coefficient corr(T′, q′) ∈ [-1, 1] [-].
"""
@inline correlation_Tq(params) = CAP.Tq_correlation_coefficient(params)

"""
    sgs_correlation_Tq(T′T′, q′q′, T′q′, fallback, r_max)

Diagnose the SGS T-q correlation from the (co)variances,
`corr = clamp(T′q′ / sqrt(T′T′ · q′q′), ±r_max)`. Where both variances are below
the numerical floor (`T′T′ · q′q′ ≤ ϵ_numerics²`, e.g. quiescent air with no
resolved gradients) the ratio is ill-conditioned, so the prescribed `fallback`
(`correlation_Tq`) is returned instead. Used by `DiagnosedTqCorrelation`.
"""
@inline function sgs_correlation_Tq(T′T′, q′q′, T′q′, fallback, r_max)
    FT = typeof(T′T′)
    denom = T′T′ * q′q′
    return ifelse(
        denom > ϵ_numerics(FT)^2,
        clamp(T′q′ / sqrt(denom), -r_max, r_max),
        fallback,
    )
end

# ============================================================================
# Extract standard deviations and correlation coefficient
# ============================================================================

"""
    sgs_stddevs_and_correlation(q′q′, T′T′, corr_Tq)

Convert variances to standard deviations, enforcing physical validity.

Variances are floored at zero before the square root, and the correlation is
clamped to ``|\\rho| \\leq 1`` as required by Cauchy-Schwarz, so that a diagnosed
covariance that slightly violates either constraint cannot produce a `NaN`.
Humidity samples that still land below zero at extreme nodes are clamped downstream
by the transform functor.

# Arguments

  - `q′q′`: Variance of total water ``\\langle q'^2 \\rangle`` [(kg/kg)²].
  - `T′T′`: Variance of temperature ``\\langle T'^2 \\rangle`` [K²].
  - `corr_Tq`: Correlation coefficient corr(T′, q′) [-].

# Returns

Tuple `(σ_q, σ_T, corr)` [kg/kg, K, -].
"""
@inline function sgs_stddevs_and_correlation(q′q′, T′T′, corr_Tq)
    FT = typeof(corr_Tq)
    σ_q = sqrt(max(FT(0), q′q′))
    σ_T = sqrt(max(FT(0), T′T′))
    # Enforce |corr| ≤ 1
    corr = clamp(corr_Tq, -one(FT), one(FT))
    return (σ_q, σ_T, corr)
end

# ============================================================================
# Physical Point Transform Functors
# ============================================================================

"""
    AbstractPhysicalPointTransform

Functor mapping a pair of standardized quadrature nodes `(χ1, χ2)` to a physical
state `(T_hat, q_hat)` [K, kg/kg].

One transform is built per grid cell by [`create_physical_transform`](@ref), which
precomputes every loop-invariant constant so the `N²` inner evaluations avoid
repeated `sqrt`, `log`, and division. Subtypes correspond one-to-one to the
[`AbstractSGSDistribution`](@ref) subtypes:
`GaussianPhysicalPointTransform`, `LogNormalPhysicalPointTransform`, and
`GridMeanPhysicalPointTransform`.
"""
abstract type AbstractPhysicalPointTransform end

"""
    GaussianPhysicalPointTransform{FT} <: AbstractPhysicalPointTransform

Transform for [`GaussianSGS`](@ref). Samples `q` from its marginal, then `T` from
the conditional distribution given `q`.

# Fields

  - `μ_T`, `μ_q`: Means of temperature [K] and specific humidity [kg/kg].
  - `σ_q`: Standard deviation of specific humidity [kg/kg].
  - `σ_c`: Conditional standard deviation of temperature,
    ``\\sigma_T \\sqrt{1 - \\rho^2}`` [K].
  - `fac`: Regression slope ``\\rho \\sigma_T / \\sigma_q`` of `T` on `q` [K kg/kg⁻¹].
  - `T_min`, `q_max`: Sampling bounds [K] and [kg/kg].
"""
struct GaussianPhysicalPointTransform{FT} <: AbstractPhysicalPointTransform
    μ_T::FT
    μ_q::FT
    σ_q::FT
    σ_c::FT      # precomputed conditional std of T: σ_T * sqrt(1 - corr^2)
    fac::FT      # precomputed factor: corr * σ_T / max(σ_q, ϵ)
    T_min::FT
    q_max::FT
end

"""
    (t::GaussianPhysicalPointTransform)(χ1, χ2)

Map Gauss-Hermite nodes `(χ1, χ2)` to `(T_hat, q_hat)` [K, kg/kg].

The ``\\sqrt{2}`` factors convert Gauss-Hermite nodes, which are defined for the
weight ``e^{-x^2}``, into standard normal deviates. `q_hat` is clamped to
`[0, q_max]` and `T_hat` is floored at `T_min`.
"""
@inline function (t::GaussianPhysicalPointTransform)(χ1, χ2)
    FT = typeof(t.μ_q)
    sqrt2 = sqrt(FT(2))

    q_hat = clamp(t.μ_q + sqrt2 * t.σ_q * χ1, zero(FT), t.q_max)
    μ_c = t.μ_T + t.fac * (q_hat - t.μ_q)
    T_hat = max(t.T_min, μ_c + sqrt2 * t.σ_c * χ2)

    return (T_hat, q_hat)
end

"""
    LogNormalPhysicalPointTransform{FT} <: AbstractPhysicalPointTransform

Transform for [`LogNormalSGS`](@ref). Samples `q` in log space and `T` from a
Gaussian correlated with it through a copula.

# Fields

  - `μ_T`, `μ_q`: Means of temperature [K] and specific humidity [kg/kg].
  - `σ_T`: Standard deviation of temperature [K].
  - `μ_ln`, `σ_ln`: Location and scale of the underlying normal in log space,
    matched to `μ_q` and `σ_q` [log kg/kg].
  - `c1`, `c2`: Copula coefficients ``\\rho`` and ``\\sqrt{1 - \\rho^2}`` [-].
  - `use_lognormal`: `false` where `μ_q` or `σ_q` is too small for the log-normal
    parameters to be meaningful; sampling then returns `μ_q`.
  - `T_min`, `q_max`: Sampling bounds [K] and [kg/kg].
"""
struct LogNormalPhysicalPointTransform{FT} <: AbstractPhysicalPointTransform
    μ_T::FT
    μ_q::FT
    σ_T::FT
    μ_ln::FT
    σ_ln::FT
    c1::FT       # corr
    c2::FT       # sqrt(1 - corr^2)
    use_lognormal::Bool
    T_min::FT
    q_max::FT
end

"""
    (t::LogNormalPhysicalPointTransform)(χ1, χ2)

Map Gauss-Hermite nodes `(χ1, χ2)` to `(T_hat, q_hat)` [K, kg/kg].

The node `χ1` drives the log-normal humidity; the temperature deviate is the copula
combination `c1·χ1 + c2·χ2`, which carries the requested correlation. `q_hat` is
capped at `q_max` and `T_hat` floored at `T_min`; no lower clamp on `q_hat` is
needed because the log-normal is positive by construction.
"""
@inline function (t::LogNormalPhysicalPointTransform)(χ1, χ2)
    FT = typeof(t.μ_q)
    sqrt2 = sqrt(FT(2))

    z_q = χ1
    z_T = t.c1 * χ1 + t.c2 * χ2

    q_lognormal = exp(t.μ_ln + sqrt2 * t.σ_ln * z_q)
    q_hat = clamp(ifelse(t.use_lognormal, q_lognormal, t.μ_q), zero(FT), t.q_max)
    T_hat = max(t.T_min, t.μ_T + sqrt2 * t.σ_T * z_T)

    return (T_hat, q_hat)
end

"""
    GridMeanPhysicalPointTransform{FT} <: AbstractPhysicalPointTransform

Transform for [`GridMeanSGS`](@ref): returns `(μ_T, μ_q)` [K, kg/kg] for any node,
ignoring the quadrature variables. No clamping is needed, since the grid-mean state
is already physical.

# Fields

  - `μ_T`: Mean temperature [K].
  - `μ_q`: Mean specific humidity [kg/kg].
"""
struct GridMeanPhysicalPointTransform{FT} <: AbstractPhysicalPointTransform
    μ_T::FT
    μ_q::FT
end

@inline function (t::GridMeanPhysicalPointTransform)(χ1, χ2)
    return (t.μ_T, t.μ_q)
end

"""
    create_physical_transform(dist, μ_q, μ_T, σ_q, σ_T, corr, T_min, q_max)

Build the [`AbstractPhysicalPointTransform`](@ref) functor for SGS distribution
`dist`.

All loop-invariant constants (conditional standard deviations, regression slopes,
log-space parameters) are computed here, so the `N²` inner evaluations avoid
repeated `sqrt`, `log`, and division. A functor rather than a closure is used to
keep the quadrature allocation-free on GPU.

# Arguments

  - `dist`: SGS distribution, dispatched on.
  - `μ_q`, `μ_T`: Means of specific humidity [kg/kg] and temperature [K].
  - `σ_q`, `σ_T`: Corresponding standard deviations [kg/kg] and [K].
  - `corr`: Correlation coefficient, already clamped to `[-1, 1]` [-].
  - `T_min`, `q_max`: Sampling bounds [K] and [kg/kg].

Called from [`integrate_over_sgs`](@ref).
"""
@inline function create_physical_transform(
    ::GaussianSGS, μ_q::FT, μ_T::FT, σ_q::FT, σ_T::FT, corr::FT, T_min::FT, q_max::FT,
) where {FT}
    σ_c = sqrt(max(one(FT) - corr^2, zero(FT))) * σ_T
    fac = corr * σ_T / max(σ_q, ϵ_numerics(FT))
    return GaussianPhysicalPointTransform(μ_T, μ_q, σ_q, σ_c, fac, T_min, q_max)
end

@inline function create_physical_transform(
    ::LogNormalSGS, μ_q::FT, μ_T::FT, σ_q::FT, σ_T::FT, corr::FT, T_min::FT, q_max::FT,
) where {FT}
    ε = ϵ_numerics(FT)
    c1 = corr
    c2 = sqrt(max(zero(FT), one(FT) - corr^2))

    ratio = σ_q / max(μ_q, ε)
    σ_ln = sqrt(log(one(FT) + ratio^2))
    μ_ln = log(max(μ_q, ε)) - σ_ln^2 / 2
    use_lognormal = (μ_q > ε) & (σ_q > zero(FT))

    return LogNormalPhysicalPointTransform(
        μ_T, μ_q, σ_T, μ_ln, σ_ln, c1, c2, use_lognormal, T_min, q_max,
    )
end

@inline function create_physical_transform(
    ::GridMeanSGS, μ_q::FT, μ_T::FT, σ_q::FT, σ_T::FT, corr::FT, T_min::FT, q_max::FT,
) where {FT}
    return GridMeanPhysicalPointTransform(μ_T, μ_q)
end

# ============================================================================
# Quadrature Integration
# ============================================================================

"""
    sum_over_quadrature_points(f, get_x_hat, quad)

Compute the weighted sum of `f(T, q)` over the `N²` quadrature points.

Approximates the expectation

```math
\\int\\!\\!\\int f(T, q) P(T, q) \\, dT \\, dq \\approx
  \\frac{1}{\\pi} \\sum_{i,j} w_i w_j f(T_{ij}, q_{ij})
```

with the ``1/\\pi`` normalization of the two-dimensional Gauss-Hermite rule applied
as ``1/\\sqrt{\\pi}`` per dimension. Accumulation uses `RecursiveApply`'s `⊞` and
`⊠`, so `f` may return a scalar or a `NamedTuple` of scalars.

# Arguments

  - `f`: Point-wise function `(T_hat, q_hat) -> result`.
  - `get_x_hat`: Transform functor `(χ1, χ2) -> (T_hat, q_hat)`; see
    [`create_physical_transform`](@ref).
  - `quad`: [`SGSQuadrature`](@ref) holding the nodes and weights.

# Returns

The weighted sum, of the same type as `f(T_hat, q_hat)`.
"""
function sum_over_quadrature_points(
    f,
    get_x_hat,
    quad::SGSQuadrature{N},
) where {N}
    χ = quad.a
    weights = quad.w
    FT = eltype(χ)

    inv_sqrt_pi = one(FT) / sqrt(FT(π))

    # Use loops (not ntuple) for register reuse across iterations: each loop
    # iteration releases registers from the previous one, dramatically reducing
    # peak register usage. Seed both accumulators from real (i, j) = (1, 1)
    # evaluations rather than a separate `rzero(f(...))` dummy call — that
    # saves one full evaluation of `f` per cell (≈ 11% of work at N = 3).
    @inbounds begin
        x_hat = get_x_hat(χ[1], χ[1])
        inner_sum = f(x_hat...) ⊠ (weights[1] * inv_sqrt_pi)
        for j in 2:N
            x_hat = get_x_hat(χ[1], χ[j])
            inner_sum = inner_sum ⊞ (f(x_hat...) ⊠ (weights[j] * inv_sqrt_pi))
        end
        outer_sum = inner_sum ⊠ (weights[1] * inv_sqrt_pi)

        for i in 2:N
            x_hat = get_x_hat(χ[i], χ[1])
            inner_sum = f(x_hat...) ⊠ (weights[1] * inv_sqrt_pi)
            for j in 2:N
                x_hat = get_x_hat(χ[i], χ[j])
                inner_sum = inner_sum ⊞ (f(x_hat...) ⊠ (weights[j] * inv_sqrt_pi))
            end
            outer_sum = outer_sum ⊞ (inner_sum ⊠ (weights[i] * inv_sqrt_pi))
        end
    end

    return outer_sum
end

"""
    integrate_over_sgs(f, quad, μ_q, μ_T, q′q′, T′T′, corr_Tq)

Integrate `f(T, q)` over the bivariate SGS distribution.

Converts the variances to standard deviations, builds the transform functor for
`quad.dist` (see [`create_physical_transform`](@ref)), and evaluates the
Gauss-Hermite rule. Temperature is always Gaussian; `quad.dist` determines only how
specific humidity is sampled. `μ_T` and `μ_q` are promoted to a common type so that
either may independently be a `Dual` under autodiff, when `ρe_tot` or `ρq_tot` is
perturbed.

# Arguments

  - `f`: Point-wise function `(T_hat, q_hat) -> result`.
  - `quad`: [`SGSQuadrature`](@ref) holding distribution type, nodes, and weights.
  - `μ_q`, `μ_T`: Mean specific humidity [kg/kg] and temperature [K].
  - `q′q′`, `T′T′`: Variances of `q` [(kg/kg)²] and `T` [K²].
  - `corr_Tq`: Correlation coefficient corr(T′, q′) [-].

# Returns

The weighted sum ``\\approx E[f(T, q)]``, of the same type as `f(T_hat, q_hat)`.
"""
function integrate_over_sgs(f, quad, μ_q, μ_T, q′q′, T′T′, corr_Tq)
    # Use functor instead of closure to avoid heap allocations.
    # Field order is (T, q) to match return order of get_physical_point.
    transform = build_physical_transform(quad, μ_q, μ_T, q′q′, T′T′, corr_Tq)
    return sum_over_quadrature_points(f, transform, quad)
end

"""
    build_physical_transform(quad, μ_q, μ_T, q′q′, T′T′, corr_Tq)

Build the [`AbstractPhysicalPointTransform`](@ref) for `quad` from the mean
state and (co)variances, converting variances to standard deviations and
clamping the correlation on the way (see `sgs_stddevs_and_correlation`).
Factored out of [`integrate_over_sgs`](@ref) so callers that need the
individual quadrature-point values (see `quadrature_point_values`)
sample from exactly the same transform.
"""
@inline function build_physical_transform(quad, μ_q, μ_T, q′q′, T′T′, corr_Tq)
    σ_q, σ_T, corr = sgs_stddevs_and_correlation(q′q′, T′T′, corr_Tq)

    # Promote μ_T and μ_q to the widest type: with autodiff, either may
    # independently be a Dual (when ρe_tot or ρq_tot is perturbed).
    μ_T_p, μ_q_p = promote(μ_T, μ_q)
    return create_physical_transform(
        quad.dist,
        μ_q_p,
        μ_T_p,
        oftype(μ_T_p, σ_q),
        oftype(μ_T_p, σ_T),
        oftype(μ_T_p, corr),
        oftype(μ_T_p, quad.T_min),
        oftype(μ_T_p, quad.q_max),
    )
end

"""
    quadrature_point_values(f, get_x_hat, quad)

Evaluate `f(T_hat, q_hat)` at each of the `N²` quadrature points and return
the values as an `SVector{N²}`, in the same point order as
`quadrature_prob_weights`. Unlike `sum_over_quadrature_points`,
which only accumulates the weighted sum, this materializes the individual
point values (in registers — `N` is a compile-time constant), for closures
that need the sampled distribution itself, such as the discrete
Lagrange-multiplier fit in `_compute_sgs_moments`.
"""
@inline function quadrature_point_values(
    f,
    get_x_hat,
    quad::SGSQuadrature{N},
) where {N}
    χ = quad.a
    return SA.SVector{N * N}(ntuple(Val(N * N)) do k
        j, i = fldmod1(k, N)
        f(get_x_hat(χ[i], χ[j])...)
    end)
end

"""
    quadrature_prob_weights(quad)

Return the `N²` probability weights of the two-dimensional quadrature rule as
an `SVector{N²}` in the same point order as `quadrature_point_values`:
the tensor products `wᵢ·wⱼ/π`, which sum to 1 for the Gauss-Hermite rule.
"""
@inline function quadrature_prob_weights(quad::SGSQuadrature{N}) where {N}
    w = quad.w
    FT = eltype(w)
    inv_pi = one(FT) / FT(π)
    return SA.SVector{N * N}(ntuple(Val(N * N)) do k
        j, i = fldmod1(k, N)
        w[i] * w[j] * inv_pi
    end)
end

"""
    integrate_over_sgs(f, ::GridMeanSGS, μ_q, μ_T, q′q′, T′T′, corr_Tq)

Evaluate `f(μ_T, μ_q)` directly, the grid-mean fast path.

Lets callers pass a bare `GridMeanSGS()` without wrapping it in an
[`SGSQuadrature`](@ref), which would require knowing `FT` from the space. The
variances and correlation are accepted for signature compatibility and ignored.
"""
@inline function integrate_over_sgs(f, ::GridMeanSGS, μ_q, μ_T, q′q′, T′T′, corr_Tq)
    return f(μ_T, μ_q)
end

"""
    not_quadrature(sgs_quad)

Return `true` when `sgs_quad` performs no SGS sampling.

That is the case for `nothing`, a bare [`GridMeanSGS`](@ref), or an
[`SGSQuadrature`](@ref) whose distribution is `GridMeanSGS`. Callers use this to
select the cheaper grid-mean code path.
"""
@inline function not_quadrature(sgs_quad)
    return isnothing(sgs_quad) || sgs_quad isa GridMeanSGS ||
           (sgs_quad isa SGSQuadrature && sgs_quad.dist isa GridMeanSGS)
end
