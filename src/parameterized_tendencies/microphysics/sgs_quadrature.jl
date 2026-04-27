"""
    SGS Quadrature Utilities

Subgrid-scale (SGS) quadrature infrastructure for integrating point-wise functions
over thermodynamic fluctuations. Supports multiple distribution types and provides
reusable utilities for cloud fraction, microphysics tendencies, and other SGS diagnostics.
"""

import StaticArrays as SA
import Thermodynamics as TD
import LinearAlgebra: SymTridiagonal, dot, eigen
import ClimaCore.Fields as Fields
import ClimaCore.Geometry as Geometry
import ClimaCore.RecursiveApply: rzero, ⊞, ⊠
import UnrolledUtilities: unrolled_reduce

# Golub–Welsch lets us support moderate orders beyond small precomputed tables (tests,
# convergence studies, offline calibration) without maintaining huge closed-form lists.
const _MAX_GAUSS_HERMITE_ORDER_GW = 64
const _MAX_GAUSS_LEGENDRE_ORDER_GW = 64

function _gauss_hermite_golub_welsch(::Type{FT}, n::Int) where {FT}
    (n >= 1 && n <= _MAX_GAUSS_HERMITE_ORDER_GW) ||
        error("Gauss-Hermite Golub–Welsch requires 1 ≤ n ≤ $_MAX_GAUSS_HERMITE_ORDER_GW; got $n.")
    if n == 1
        return (FT[0], FT[sqrt(FT(π))])
    end
    dv = zeros(FT, n)
    ev = [sqrt(FT(i) / FT(2)) for i in 1:(n - 1)]
    J = SymTridiagonal(dv, ev)
    F = eigen(J)
    nodes = F.values
    V = F.vectors
    w = zeros(FT, n)
    πv = sqrt(FT(π))
    @inbounds for j in 1:n
        w[j] = πv * V[1, j]^2
    end
    return (nodes, w)
end

function _gauss_legendre_neg1_1_golub_welsch(::Type{FT}, n::Int) where {FT}
    (n >= 1 && n <= _MAX_GAUSS_LEGENDRE_ORDER_GW) ||
        error("Gauss-Legendre Golub–Welsch requires 1 ≤ n ≤ $_MAX_GAUSS_LEGENDRE_ORDER_GW; got $n.")
    if n == 1
        return (FT[0], FT[2])
    end
    dv = zeros(FT, n)
    ev = [FT(k) / sqrt(FT(4 * k * k - 1)) for k in 1:(n - 1)]
    J = SymTridiagonal(dv, ev)
    F = eigen(J)
    nodes = F.values
    V = F.vectors
    w = zeros(FT, n)
    @inbounds for j in 1:n
        w[j] = 2 * V[1, j]^2
    end
    return (nodes, w)
end

"""
    gauss_hermite(FT, N)

Gauss-Hermite quadrature nodes and weights for order `N`.

Nodes are roots of the physicists' Hermite polynomial ``H_N(x)``.
Weights are standard Gauss-Hermite weights for integration against ``e^{-x^2}``.

# Arguments
- `FT`: Floating-point type
- `N::Int`: Quadrature order (`1 ≤ N ≤ 5` tabulated; `6 ≤ N ≤ 64` via Golub–Welsch)

# Returns
Tuple `(nodes, weights)` as `Vector{FT}`.
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
    elseif N <= _MAX_GAUSS_HERMITE_ORDER_GW
        return _gauss_hermite_golub_welsch(FT, N)
    else
        error(
            "Gauss-Hermite quadrature order $N not implemented. Use N ∈ {1,…,5} (tabulated) or N ∈ {6,…,$(_MAX_GAUSS_HERMITE_ORDER_GW)} (Golub–Welsch).",
        )
    end
end

"""
    gauss_legendre_01(FT, N)

Gauss-Legendre quadrature nodes and weights for order `N` on ``[0, 1]``.

Precomputed from standard ``[-1,1]`` quadrature via ``x = (t+1)/2``, ``w_{01} = w/2``.

# Arguments
- `FT`: Floating-point type
- `N::Int`: Quadrature order (`1 ≤ N ≤ 5` tabulated; `6 ≤ N ≤ 64` via Golub–Welsch)

# Returns
Tuple `(nodes, weights)` as `Vector{FT}`.
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
    elseif N <= _MAX_GAUSS_LEGENDRE_ORDER_GW
        t, w = _gauss_legendre_neg1_1_golub_welsch(FT, N)
        two = FT(2)
        x01 = map(tv -> (tv + one(FT)) / two, t)
        w01 = map(wv -> wv / two, w)
        return (x01, w01)
    else
        error(
            "Gauss-Legendre quadrature order $N not implemented. Use N ∈ {1,…,5} (tabulated) or N ∈ {6,…,$(_MAX_GAUSS_LEGENDRE_ORDER_GW)} (Golub–Welsch).",
        )
    end
end

"""
    gauss_legendre_neg1_1(FT, N)

Gauss–Legendre nodes `t_k` and weights `w_k` on `[-1, 1]` for `∫_{-1}^1 f(t) dt`.

Obtained from [`gauss_legendre_01`](@ref) via `t = 2x - 1`, `w = 2 w_{01}`.
"""
function gauss_legendre_neg1_1(::Type{FT}, N::Int) where {FT}
    x, w01 = gauss_legendre_01(FT, N)
    two = FT(2)
    t = map(xv -> two * xv - one(FT), x)
    w = map(wv -> two * wv, w01)
    return (t, w)
end

if !isdefined(@__MODULE__, :AbstractSGSDistribution)
    include(joinpath(@__DIR__, "sgs_distribution_types.jl"))
end

# ============================================================================
# Quadrature Struct
# ============================================================================

"""
    SGSQuadrature{N, A, W, D, FT, ZZ, ZW} <: AbstractSGSamplingType

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
  thermodynamics calculations. Set from ClimaParams `temperature_minimum`.
- `q_max::FT`: Maximum specific humidity [kg/kg]. Used to clamp sampled
  humidity values in `get_physical_point` to prevent extreme supersaturation
  from driving unphysically low temperatures through excessive latent heat.
  Set from ClimaParams `specific_humidity_maximum`.

# Constructors

    SGSQuadrature(FT; quadrature_order=3, distribution=GaussianSGS(), T_min=150.0, q_max=0.1)

Create an `SGSQuadrature` with the specified floating-point type `FT`,
quadrature order, distribution type, minimum temperature, and maximum humidity.

The T-q correlation coefficient is provided externally via `correlation_Tq(params)`
rather than being stored in this struct.

`T_min` and `q_max` are read from ClimaParams (`temperature_minimum` and
`specific_humidity_maximum`) and passed through at construction time.
"""
struct SGSQuadrature{N, A, W, D <: AbstractSGSDistribution, FT, ZZ, ZW} <:
       AbstractSGSamplingType
    a::A       # Gauss–Hermite nodes (standardized)
    w::W       # Gauss–Hermite weights
    dist::D    # distribution type
    T_min::FT  # minimum temperature for physical validity [K]
    q_max::FT  # maximum specific humidity [kg/kg]
    z_t::ZZ    # Gauss–Legendre nodes on [-1,1] for layer height (`nothing` unless column-tensor)
    z_w::ZW    # matching weights for `(1/2) ∫_{-1}^1` outer average
    function SGSQuadrature(
        ::Type{FT};
        quadrature_order = 3,
        distribution::D = GaussianSGS(),
        T_min = FT(150),  # Reasonable default for atmospheric applications
        q_max = FT(0.1),  # Maximum humidity: ~100 g/kg (well above physical max)
    ) where {FT, D <: AbstractSGSDistribution}
        # GridMeanSGS always uses N=1 (single point at origin)
        N = distribution isa GridMeanSGS ? 1 : quadrature_order
        a, w = get_quadrature_nodes_weights(distribution, FT, N)
        a, w = SA.SVector{N, FT}(a), SA.SVector{N, FT}(w)
        # Vertically resolved SGS types that are NOT profile–Rosenblatt need
        # Gauss–Legendre z-nodes for their column-tensor / LHS / principal-axis /
        # Voronoi / barycentric-seeds inner loops.
        needs_z_nodes =
            distribution isa AbstractVerticallyResolvedSGS &&
            !(distribution isa VerticallyResolvedSGS{<:SubgridProfileRosenblatt})
        if needs_z_nodes
            z_t_vec, z_w_vec = gauss_legendre_neg1_1(FT, N)
            zt = SA.SVector{N, FT}(z_t_vec)
            zw = SA.SVector{N, FT}(z_w_vec)
            return new{N, typeof(a), typeof(w), D, FT, typeof(zt), typeof(zw)}(
                a,
                w,
                distribution,
                FT(T_min),
                FT(q_max),
                zt,
                zw,
            )
        else
            return new{N, typeof(a), typeof(w), D, FT, Nothing, Nothing}(
                a,
                w,
                distribution,
                FT(T_min),
                FT(q_max),
                nothing,
                nothing,
            )
        end
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

@inline get_quadrature_nodes_weights(::VerticallyResolvedSGS, FT, N) =
    gauss_hermite(FT, N)

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
ᶜθ_li = @. lazy(TD.liquid_ice_pottemp(thp, ᶜT, ᶜρ, ᶜq_tot, ᶜq_lcl, ᶜq_icl))
ᶜ∂T_∂θ = @. lazy(∂T_∂θ_li(thp, ᶜT, ᶜθ_li, ᶜq_lcl, ᶜq_icl, ᶜq_tot, ᶜρ))
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
# Extract standard deviations and correlation coefficient
# ============================================================================

"""
    sgs_stddevs_and_correlation(q′q′, T′T′, corr_Tq)

Compute standard deviations from variances and enforce physical validity.

Applies two constraints:
1. Variances are floored at zero before taking the square root
2. Cauchy-Schwarz inequality: ``|\\rho| \\leq 1`` (enforced via `clamp`)

Negative `q_hat` values at extreme quadrature points are handled downstream
by `max(0, q_hat)` clamping in `get_physical_point`.

# Arguments
- `q′q′`: Variance of total water ``\\langle q'^2 \\rangle``
- `T′T′`: Variance of temperature ``\\langle T'^2 \\rangle``
- `corr_Tq`: Correlation coefficient ``\\rho(T', q')``

# Returns
Tuple `(σ_q, σ_T, corr)` of standard deviations and clamped correlation.
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
# Inner Distribution Accessor
# ============================================================================

"""
    _inner_dist(dist::VerticallyResolvedSGS{S, I})

Return the inner bivariate distribution instance (`GaussianSGS()` or `LogNormalSGS()`)
from a [`VerticallyResolvedSGS`](@ref). Used by both the cell-center
[`integrate_over_sgs`](@ref) overload (bivariate path) and the layer-profile overload
defined in `subgrid_layer_profile_quadrature.jl`.
"""
@inline _inner_dist(::VerticallyResolvedSGS{<:Any, GaussianSGS}) = GaussianSGS()
@inline _inner_dist(::VerticallyResolvedSGS{<:Any, LogNormalSGS}) = LogNormalSGS()

# ============================================================================
# Physical Point Computation
# ============================================================================

"""
    get_physical_point(dist, χ1, χ2, μ_q, μ_T, σ_q, σ_T, corr, T_min, q_max)

Transform Gauss–Hermite abscissae ``(\\chi_1, \\chi_2)`` to physical ``(T, q)`` for
distributions that use **only** the Gaussian layer (no structural uniform).

**`GaussianSGS`**: same Hermite nodes map to the same correlated Gaussian law as in
Gauss–Hermite quadrature for a bivariate normal with s.d. ``σ_q``, ``σ_T`` and correlation
``corr`` (then bounds):
```math
q = \\clamp(\\mu_q + \\sqrt{2} \\sigma_q \\chi_1, \\; 0, \\; q_{max})
```
```math
T = \\max(T_{min}, \\; \\mu_T + \\sqrt{2} \\sigma_T (\\rho \\chi_1^{\\mathrm{eff}} + \\sqrt{1-\\rho^2} \\chi_2))
```
with ``\\chi_1^{\\mathrm{eff}}`` inferred from the **clamped** `q` so `T` is conditioned on
the realized `q` fluctuation.

**`LogNormalSGS`**: log-normal for `q`, Gaussian for `T`, Gaussian copula
```math
q = \\min(q_{max}, \\; \\exp(\\mu_{\\ln} + \\sqrt{2} \\sigma_{\\ln} z_q))
```
with ``z_q = \\chi_1``, ``z_T = \\rho \\chi_1 + \\sqrt{1-\\rho^2} \\chi_2``.

# Arguments
- `dist`: e.g. `GaussianSGS`, `LogNormalSGS`, `GridMeanSGS`, or gridscale wrappers that
  delegate to these
- `χ1`, `χ2`: Hermite abscissae
- `μ_q`, `μ_T`: Mean specific humidity [kg/kg] and temperature [K]
- `σ_q`, `σ_T`: Standard deviations of `q` and `T` for the Gaussian layer
- `corr`: Correlation coefficient ``\\rho`` for that layer
- `T_min`, `q_max`: Physical bounds

# Returns
Tuple `(T_hat, q_hat)` of physical values.
"""
@inline function get_physical_point(
    ::GaussianSGS,
    χ1,
    χ2,
    μ_q,
    μ_T,
    σ_q,
    σ_T,
    corr,
    T_min,
    q_max,
)
    FT = typeof(μ_q)
    sqrt2 = sqrt(FT(2))

    # Clamp q to physically valid ranges
    q_hat = clamp(μ_q + sqrt2 * σ_q * χ1, zero(FT), q_max)

    # Re-infer effective χ1 from clamped q to maintain physical T-q correlation.
    # If a negative q fluctuation was truncated to 0, T should only be conditioned
    # on the q=0 state, not the "phantom" negative q.
    χ1_eff = (q_hat - μ_q) / (sqrt2 * max(σ_q, ϵ_numerics(FT)))

    # Conditional mean and std for T given *clamped* q
    σ_c = sqrt(max(one(FT) - corr^2, zero(FT))) * σ_T
    μ_c = μ_T + sqrt2 * corr * σ_T * χ1_eff

    # Clamp T to physically valid ranges
    T_hat = max(T_min, μ_c + sqrt2 * σ_c * χ2)

    return (T_hat, q_hat)
end

@inline function get_physical_point(
    ::LogNormalSGS,
    χ1,
    χ2,
    μ_q,
    μ_T,
    σ_q,
    σ_T,
    corr,
    T_min,
    q_max,
)
    FT = typeof(μ_q)
    sqrt2 = sqrt(FT(2))
    ε = ϵ_numerics(FT)

    # Step 1: Generate correlated Gaussian variables using copula approach
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
    q_hat = clamp(ifelse(use_lognormal, q_lognormal, μ_q), zero(FT), q_max)

    # Step 3: Keep Gaussian for T using correlated z_T, clamped to T_min
    T_hat = max(T_min, μ_T + sqrt2 * σ_T * z_T)

    return (T_hat, q_hat)
end

@inline function get_physical_point(
    ::VerticallyResolvedSGS{<:Any, GaussianSGS},
    χ1,
    χ2,
    μ_q,
    μ_T,
    σ_q,
    σ_T,
    corr,
    T_min,
    q_max,
)
    return get_physical_point(
        GaussianSGS(),
        χ1,
        χ2,
        μ_q,
        μ_T,
        σ_q,
        σ_T,
        corr,
        T_min,
        q_max,
    )
end

@inline function get_physical_point(
    ::VerticallyResolvedSGS{<:Any, LogNormalSGS},
    χ1,
    χ2,
    μ_q,
    μ_T,
    σ_q,
    σ_T,
    corr,
    T_min,
    q_max,
)
    return get_physical_point(
        LogNormalSGS(),
        χ1,
        χ2,
        μ_q,
        μ_T,
        σ_q,
        σ_T,
        corr,
        T_min,
        q_max,
    )
end

# GridMeanSGS: evaluates only at the grid mean, ignoring variance
@inline function get_physical_point(
    ::GridMeanSGS,
    χ1,
    χ2,
    μ_q,
    μ_T,
    σ_q,
    σ_T,
    corr,
    T_min,
    q_max,
)
    # Return grid mean directly, ignoring quadrature points, variance, and bounds
    # χ1, χ2, σ_q, σ_T, corr, T_min, q_max are all ignored
    (μ_T, μ_q)
end

# ============================================================================
# Physical Point Transform Functor
# ============================================================================

"""
    PhysicalPointTransform

GPU-safe functor wrapping `get_physical_point` to avoid heap-allocated closures.

Captures all parameters needed by `get_physical_point(dist, χ1, χ2, ...)` in a
struct. Field order matches return order `(T, q)` for consistency.

# Fields
- `dist`: Distribution type (`GaussianSGS`, `LogNormalSGS`, or `GridMeanSGS`)
- `μ_T`: Mean temperature [K]
- `μ_q`: Mean specific humidity [kg/kg]
- `σ_T`: Standard deviation of T [K]
- `σ_q`: Standard deviation of q [kg/kg]
- `corr`: Correlation coefficient [-1, 1]
- `T_min`: Minimum temperature floor [K]
- `q_max`: Maximum specific humidity ceiling [kg/kg]
"""
struct PhysicalPointTransform{D, FT}
    dist::D
    μ_T::FT
    μ_q::FT
    σ_T::FT
    σ_q::FT
    corr::FT
    T_min::FT
    q_max::FT
end

@inline function (t::PhysicalPointTransform)(χ1, χ2)
    return get_physical_point(
        t.dist,
        χ1,
        χ2,
        t.μ_q,
        t.μ_T,
        t.σ_q,
        t.σ_T,
        t.corr,
        t.T_min,
        t.q_max,
    )
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
- `get_x_hat`: Function `(χ1, χ2) -> (T_hat, q_hat)`
- `quad`: `SGSQuadrature` struct

# Returns
Weighted sum with the same type as `f(T, q)`.
"""
function sum_over_quadrature_points(f, get_x_hat, quad::SGSQuadrature{N}) where {N}
    χ = quad.a
    weights = quad.w
    FT = eltype(χ)
    inv_sqrt_pi = one(FT) / sqrt(FT(π))

    outer_sum = rzero(f(get_x_hat(χ[1], χ[1])...))
    @inbounds for i in 1:N
        inner_sum = rzero(f(get_x_hat(χ[1], χ[1])...))
        @inbounds for j in 1:N
            x_hat = get_x_hat(χ[i], χ[j])
            contribution = f(x_hat...) ⊠ (weights[j] * inv_sqrt_pi)
            inner_sum = inner_sum ⊞ contribution
        end
        weighted_inner = inner_sum ⊠ (weights[i] * inv_sqrt_pi)
        outer_sum = outer_sum ⊞ weighted_inner
    end
    return outer_sum
end

"""
    tensor_gh_row_split(N::Int) -> (n1::Int, n2::Int)

`n1 = N ÷ 2`, `n2 = N - n1` with `n1 + n2 == N`. For **two-branch mixture** tensor
Gauss–Hermite, partition the first Hermite factor ``χ_1`` (the ``j`` index in
`gauss_hermite_nodes_bivariate_chi1_slice`), giving `N^2` nodes per height instead of `2 N^2`.

[`sum_over_quadrature_points`](@ref) for [`SubgridColumnTensor`](@ref) evaluates **one**
bivariate Gaussian per height with a full `N × N` tensor (`N^2` evaluations per `z`);
the `2 N^2` budget story applies to the toy mixture DN/UP stack, not that inner loop.
"""
function tensor_gh_row_split(N::Int)
    N < 2 && throw(ArgumentError("tensor_gh_row_split requires N >= 2"))
    n1 = N ÷ 2
    return n1, N - n1
end

"""
    integrate_over_sgs(f, quad, μ_q, μ_T, q′q′, T′T′, corr_Tq)

Integrate `f(T, q)` over the bivariate SGS distribution.

Converts variances to standard deviations, constructs a `PhysicalPointTransform`
for the distribution type in `quad`, and evaluates the Gauss-Hermite quadrature.
Temperature is always Gaussian; the distribution of specific humidity is
determined by `quad.dist` (see [`get_physical_point`](@ref)).

# Relation to layer-profile `integrate_over_sgs`

Both approximate ``\\mathbb{E}[f]`` over SGS noise. The layer-profile overload (defined in
`subgrid_layer_profile_quadrature.jl`) takes additional arguments: **layer thickness,
`LocalGeometry`, and subcell gradients** as part of the public contract for vertically
resolved SGS.

# Vertically resolved ``quad.dist``

For [`AbstractVerticallyResolvedSGS`](@ref), the branch below applies only a
**bivariate (T, q)** map at cell center (inner `GaussianSGS` / `LogNormalSGS`); the
**layer-profile** type parameter `S` is not used. Layer-mean, gradient-aware rules
use the layer-profile `integrate_over_sgs` overload (0M saturation, and 1M via
[`microphysics_tendencies_1m_sgs_row`](@ref); column tensor, LHS, principal axis,
Voronoi/barycentric seeds, profile–Rosenblatt, …). Bivariate **1M** only dispatches
[`microphysics_tendencies_1m`](@ref) for base
[`GaussianSGS`](@ref) / [`LogNormalSGS`](@ref) / [`GridMeanSGS`](@ref); it does not accept
vertically resolved `SGSQuadrature` at the type level. This
path remains for **base** `GaussianSGS` / `LogNormalSGS` and other cell-center bivariate
uses.

# Arguments
- `f`: Point-wise function `(T, q) -> result`
- `quad`: `SGSQuadrature` struct (contains distribution type, nodes, weights)
- `μ_q`, `μ_T`: Mean specific humidity [kg/kg] and temperature [K]
- `q′q′`, `T′T′`: Variances of `q` and `T`
- `corr_Tq`: Correlation coefficient ``ρ(T', q')``

# Returns
Weighted sum ``\\approx E[f(T, q)]`` with the same type as `f(T, q)`.
"""
function integrate_over_sgs(f, quad::SGSQuadrature, μ_q, μ_T, q′q′, T′T′, corr_Tq)
    if _is_vertically_resolved_sgs(quad.dist)
        σ_q, σ_T, corr = sgs_stddevs_and_correlation(q′q′, T′T′, corr_Tq)
        μ_T_p, μ_q_p = promote(μ_T, μ_q)
        inner = _inner_dist(quad.dist)
        transform = PhysicalPointTransform(
            inner,
            μ_T_p,
            μ_q_p,
            oftype(μ_T_p, σ_T),
            oftype(μ_T_p, σ_q),
            oftype(μ_T_p, corr),
            oftype(μ_T_p, quad.T_min),
            oftype(μ_T_p, quad.q_max),
        )
        inner_quad = SGSQuadrature(
            typeof(μ_T_p);
            quadrature_order = quadrature_order(quad),
            distribution = inner,
            T_min = quad.T_min,
            q_max = quad.q_max,
        )
        return sum_over_quadrature_points(f, transform, inner_quad)
    end
    σ_q, σ_T, corr = sgs_stddevs_and_correlation(q′q′, T′T′, corr_Tq)
    μ_T_p, μ_q_p = promote(μ_T, μ_q)
    transform = PhysicalPointTransform(
        quad.dist,
        μ_T_p,
        μ_q_p,
        oftype(μ_T_p, σ_T),
        oftype(μ_T_p, σ_q),
        oftype(μ_T_p, corr),
        oftype(μ_T_p, quad.T_min),
        oftype(μ_T_p, quad.q_max),
    )
    return sum_over_quadrature_points(f, transform, quad)
end

function integrate_over_sgs(
    f,
    quad::SGSQuadrature,
    μ_q,
    μ_T,
    q′q′,
    T′T′,
    corr_Tq,
    δq_half,
    δT_half,
)
    return integrate_over_sgs(f, quad, μ_q, μ_T, q′q′, T′T′, corr_Tq)
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

"""
    not_quadrature(sgs_quad)

Return `true` if no quadrature sampling is chosen.
"""
@inline function not_quadrature(sgs_quad)
    return isnothing(sgs_quad) || sgs_quad isa GridMeanSGS ||
           (sgs_quad isa SGSQuadrature && sgs_quad.dist isa GridMeanSGS)
end

"""
    sgs_quadrature_moments_at_point(
        dist::AbstractSGSDistribution,
        ρ_param, ε, Δz, w_grad_q_sq, w_grad_θ_sq, ∂T∂θ_li, wq_dot_wθ,
        T′T′, q′q′,
    )

Per-cell **scalar** moments for SGS quadrature: returns
`(T_var, q_var, ρ_corr, δq_half, δT_half)` for use with [`integrate_over_sgs`](@ref).

- Base distributions ([`GaussianSGS`](@ref), [`LogNormalSGS`](@ref)): raw turbulent
  variances, `ρ_param`, zero segment half-widths.
- [`VerticallyResolvedSGS`](@ref): raw
  turbulent variances and `ρ_param` (layer-profile quadrature uses gradients separately);
  `δq_half` and `δT_half` are zero.
"""
@inline function sgs_quadrature_moments_at_point(
    dist::AbstractSGSDistribution,
    ρ_param,
    ε,
    Δz,
    w_grad_q_sq,
    w_grad_θ_sq,
    ∂T∂θ_li,
    wq_dot_wθ,
    T′T′,
    q′q′,
)
    FT = typeof(q′q′)
    z = zero(FT)
    if dist isa AbstractVerticallyResolvedSGS
        return T′T′, q′q′, ρ_param, z, z
    end
    return T′T′, q′q′, ρ_param, z, z
end

"""
    sgs_quadrature_moments_from_gradients(
        dist::AbstractSGSDistribution,
        ρ_param, ε, Δz, local_geometry, grad_q, grad_θ, ∂T∂θ_li, T′T′, q′q′,
    )

Like [`sgs_quadrature_moments_at_point`](@ref) but forms W-vector squared norms and
`dot(∇q, ∇θ)` from `grad_q`, `grad_θ` (e.g. `ᶜgradᵥ` outputs) using
`WVector(·, local_geometry)` (ClimaCore requires `LocalGeometry` for covariant → W).
"""
@inline function sgs_quadrature_moments_from_gradients(
    dist::AbstractSGSDistribution,
    ρ_param,
    ε,
    Δz,
    local_geometry,
    grad_q,
    grad_θ,
    ∂T∂θ_li,
    T′T′,
    q′q′,
)
    wvq = Geometry.WVector(grad_q, local_geometry)
    wvθ = Geometry.WVector(grad_θ, local_geometry)
    wgq = dot(wvq, wvq)
    wgθ = dot(wvθ, wvθ)
    wqdot = dot(wvq, wvθ)
    return sgs_quadrature_moments_at_point(
        dist,
        ρ_param,
        ε,
        Δz,
        wgq,
        wgθ,
        ∂T∂θ_li,
        wqdot,
        T′T′,
        q′q′,
    )
end

include(joinpath(@__DIR__, "convolution_quantile_chebyshev_tables.jl"))
include(joinpath(@__DIR__, "subgrid_layer_profile_quadrature.jl"))
