# ============================================================================
# SGS distribution types (loaded before microphysics_cache.jl in ClimaAtmos.jl)
# ============================================================================

"""
    AbstractSGSDistribution

Abstract supertype for subgrid-scale probability distributions.

Subtypes determine how specific humidity `q` is sampled in `get_physical_point`.
Temperature is always sampled from a Gaussian, regardless of distribution type.
"""
abstract type AbstractSGSDistribution end

"""
    GaussianSGS <: AbstractSGSDistribution

Bivariate Gaussian distribution for SGS fluctuations of `(T, q)`.

Both temperature and specific humidity are sampled from correlated Gaussians.
Negative `q` values at extreme quadrature points are clamped to zero.
"""
struct GaussianSGS <: AbstractSGSDistribution end

"""
    LogNormalSGS <: AbstractSGSDistribution

Log-normal distribution for specific humidity; Gaussian for temperature.

Specific humidity `q` is sampled from a log-normal distribution (positive-definite
by construction), while temperature remains Gaussian. Correlation between
`T` and `q` is maintained via a Gaussian copula.
"""
struct LogNormalSGS <: AbstractSGSDistribution end

"""
    AbstractSubgridLayerProfileQuadrature

Selects how layer-mean SGS expectations are discretized for gridscale-corrected
Gaussian / lognormal distributions when the conditional layer mean varies linearly
in height and turbulent fluctuations are homogeneous Gaussian.

See [`SubgridColumnTensor`](@ref), [`SubgridProfileRosenblatt`](@ref).
"""
abstract type AbstractSubgridLayerProfileQuadrature end

"""Reference tensor quadrature: Gauss–Legendre in layer height × inner fluctuation rule."""
struct SubgridColumnTensor <: AbstractSubgridLayerProfileQuadrature end

"""
    AbstractConvolutionQuantileMethod

Numerical method for `F^{-1}(p)` with `F` the CDF of a uniform–Gaussian convolution
along the layer-mean-gradient direction (used inside profile–Rosenblatt quadrature).
"""
abstract type AbstractConvolutionQuantileMethod end

"""Bracketed root of `F(u) - p` (e.g. Brent via RootSolvers)."""
struct ConvolutionQuantilesBracketed <: AbstractConvolutionQuantileMethod end

"""
One **Halley** correction to `F(u) = p` after a closed-form Cornish–Fisher / uniform blend initial `u`
(see `_convolution_quantile_halley`). **Not** used by Chebyshev. **No** Brent and **no** Halley iteration loop.
"""
struct ConvolutionQuantilesHalley <: AbstractConvolutionQuantileMethod end

"""
Chebyshev surrogate for `u/L` in mapped `log10(η)` (see [`chebyshev_convolution_coeffs`](@ref)).
**No root iteration** at runtime — fixed-cost polynomial eval; error budget **O(1e-3)** on ``u`` vs bracketed inversion at degree 12 (see [`chebyshev_convolution_coeffs`](@ref)).

Requires ``η = s/L`` above `ϵ_numerics` and Gauss–Legendre order `N_GL ∈ {1,…,5}` with `1 ≤ i_node ≤ N_GL`; otherwise **errors**.
"""
struct ConvolutionQuantilesChebyshevLogEta <: AbstractConvolutionQuantileMethod end

"""
    SubgridProfileRosenblatt{B} <: AbstractSubgridLayerProfileQuadrature

Profile–Rosenblatt factorization: Gaussian outer quadrature × inverse-CDF Gauss–Legendre
on the uniform–Gaussian convolution marginal, with quantiles computed per `B`.
"""
struct SubgridProfileRosenblatt{B <: AbstractConvolutionQuantileMethod} <:
       AbstractSubgridLayerProfileQuadrature end

"""Default gridscale-corrected discretization: profile–Rosenblatt with bracketed quantiles."""
const DefaultGridscaleProfileQuadrature =
    SubgridProfileRosenblatt{ConvolutionQuantilesBracketed}

"""
    AbstractGridscaleCorrectedSGS <: AbstractSGSDistribution

Subtypes use vertical subcell geometry for SGS quadrature and saturation adjustment
(see [`AbstractSubgridLayerProfileQuadrature`](@ref) on Gaussian / lognormal gridscale types).
"""
abstract type AbstractGridscaleCorrectedSGS <: AbstractSGSDistribution end

"""
    GaussianGridscaleCorrectedSGS{S} <: AbstractGridscaleCorrectedSGS

Gridscale-corrected Gaussian SGS. Type parameter `S` selects the layer-profile
quadrature (default [`DefaultGridscaleProfileQuadrature`](@ref) = profile–Rosenblatt + Brent).
"""
struct GaussianGridscaleCorrectedSGS{S <: AbstractSubgridLayerProfileQuadrature} <:
       AbstractGridscaleCorrectedSGS end

GaussianGridscaleCorrectedSGS() = GaussianGridscaleCorrectedSGS{DefaultGridscaleProfileQuadrature}()

"""
    LogNormalGridscaleCorrectedSGS{S} <: AbstractGridscaleCorrectedSGS

Gridscale-corrected log-normal `q` / Gaussian `T`. Default matches [`GaussianGridscaleCorrectedSGS`](@ref)
(profile–Rosenblatt + Brent). [`SubgridColumnTensor`](@ref) is also available for explicit vertical quadrature.
"""
struct LogNormalGridscaleCorrectedSGS{S <: AbstractSubgridLayerProfileQuadrature} <:
       AbstractGridscaleCorrectedSGS end

LogNormalGridscaleCorrectedSGS() = LogNormalGridscaleCorrectedSGS{DefaultGridscaleProfileQuadrature}()

@inline function _is_nonuniform_gridscale_corrected(
    ::Union{GaussianGridscaleCorrectedSGS, LogNormalGridscaleCorrectedSGS},
)
    return true
end
@inline _is_nonuniform_gridscale_corrected(::AbstractSGSDistribution) = false

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
