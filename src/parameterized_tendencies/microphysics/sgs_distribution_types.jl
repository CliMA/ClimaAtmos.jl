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

See [`SubgridColumnTensor`](@ref), [`SubgridProfileRosenblatt`](@ref),
[`SubgridLatinHypercubeZ`](@ref), [`SubgridPrincipalAxisLayer`](@ref),
[`SubgridVoronoiRepresentatives`](@ref), [`SubgridBarycentricSeeds`](@ref).
"""
abstract type AbstractSubgridLayerProfileQuadrature end

"""Reference tensor quadrature: Gauss–Legendre in layer height × inner fluctuation rule."""
struct SubgridColumnTensor <: AbstractSubgridLayerProfileQuadrature end

"""
    SubgridLatinHypercubeZ <: AbstractSubgridLayerProfileQuadrature

Structured ``N^2`` pairing: permute which Gauss–Legendre ``z`` level is tied to each
Hermite fluctuation node in ``(T,q)`` (LHS-style staggering).
"""
struct SubgridLatinHypercubeZ <: AbstractSubgridLayerProfileQuadrature end

"""
    SubgridPrincipalAxisLayer <: AbstractSubgridLayerProfileQuadrature

Cheap inner fluctuation rule: one Gauss–Hermite line along the dominant correlation axis
instead of a full ``N \\times N`` tensor (see calibration README §3.5).
"""
struct SubgridPrincipalAxisLayer <: AbstractSubgridLayerProfileQuadrature end

"""
    SubgridVoronoiRepresentatives <: AbstractSubgridLayerProfileQuadrature

``N^2`` representatives chosen from a dense index pool via Voronoi-style clustering
(see calibration README §3.3).
"""
struct SubgridVoronoiRepresentatives <: AbstractSubgridLayerProfileQuadrature end

"""
    SubgridBarycentricSeeds <: AbstractSubgridLayerProfileQuadrature

Deterministic ``N^2`` seeds in ``(z, \\text{index})`` space with barycentric mass
accumulation from streamed candidates (see calibration README §3.4; `quadrature_order`
must match seed layout in the kernel).
"""
struct SubgridBarycentricSeeds <: AbstractSubgridLayerProfileQuadrature end

"""
    AbstractConvolutionQuantileMethod

Numerical method for `F^{-1}(p)` with `F` the CDF of a uniform–Gaussian convolution
along the layer-mean-gradient direction (used inside profile–Rosenblatt quadrature).
"""
abstract type AbstractConvolutionQuantileMethod end

"""
Bracketed root of `F(u) - p` via Brent on the two-component
uniform–Gaussian mixture CDF (see
[`mixture_convolution_quantile_brent`](@ref)). Robust safety net and
validation reference. Costs O(10–20) CDF evaluations per quantile node;
prefer `ConvolutionQuantilesHalley` in production.
"""
struct ConvolutionQuantilesBracketed <: AbstractConvolutionQuantileMethod end

"""
Production default. One **Halley** correction to `F(u) = p` from a
closed-form Cornish–Fisher initial guess on the moment-matched Gaussian of
the two-component uniform–Gaussian mixture (mean `(L_+ − L_−)/4`,
variance `(5L_+² + 6 L_+ L_− + 5 L_−²)/48 + (s_−² + s_+²)/2`). Total cost
≈ 4 `erf`/`exp` per quantile, no iteration loop. See
[`mixture_convolution_quantile_halley`](@ref).
"""
struct ConvolutionQuantilesHalley <: AbstractConvolutionQuantileMethod end

"""
Chebyshev polynomial in `τ` for `u/L` at fixed Gauss–Legendre nodes in `p`, fit
offline for **one** law: centered `uniform[-L/2,L/2] ⊛ N(0,s²)` with `η = s/L`
(mapped `log10(η)` → `τ`). See
[`centered_uniform_gaussian_convolution_quantile_chebyshev`](@ref) and
`chebyshev_convolution_coeffs` / `gen_convolution_chebyshev_tables.jl`.

For the **two-component** layer mixture, `F^{-1}(p)` depends on
`(L_−, L_+, s_−, s_+)`; no checked-in Chebyshev surrogate exists for that
four-parameter family. [`_mixture_quantile_u`](@ref) with this type therefore
`error`s at runtime so YAML that selects it fails loudly instead of silently
using the single-law table. The struct remains so a mixture-tabulated dispatch
can be added later without renaming the API.
"""
struct ConvolutionQuantilesChebyshevLogEta <: AbstractConvolutionQuantileMethod end

"""
    SubgridProfileRosenblatt{B} <: AbstractSubgridLayerProfileQuadrature

Profile–Rosenblatt factorization: Gaussian outer quadrature in `v` × inverse-CDF
Gauss–Legendre in `p` on the inner `u` marginal. That marginal is the **two-component**
uniform–Gaussian mixture (one half-cell below center, one above), with **piecewise-linear**
conditional mean along the half in physical space and **face-anchored** conditional
standard deviation `σ_{u|v}` on each half (constant on that half in `u`). Quantiles use
`B <: AbstractConvolutionQuantileMethod` (Halley default, Brent bracketed, or Chebyshev
placeholder for a future mixture surrogate).
"""
struct SubgridProfileRosenblatt{B <: AbstractConvolutionQuantileMethod} <:
       AbstractSubgridLayerProfileQuadrature end

"""
Default gridscale-corrected discretization: profile–Rosenblatt with
**Halley** quantiles (one-step, closed-form guess). Avoids rootfinding
in the inner loop; Brent is available as a safety-net fallback.
"""
const DefaultGridscaleProfileQuadrature =
    SubgridProfileRosenblatt{ConvolutionQuantilesHalley}

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
