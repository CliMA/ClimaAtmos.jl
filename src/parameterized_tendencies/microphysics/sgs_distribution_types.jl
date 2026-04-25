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

"""
Reference tensor quadrature: Gauss–Legendre in layer height × inner fluctuation rule.
See the column-tensor branch of [`integrate_over_sgs_linear_profile`](@ref).
"""
struct SubgridColumnTensor <: AbstractSubgridLayerProfileQuadrature end

"""
    SubgridLatinHypercubeZ <: AbstractSubgridLayerProfileQuadrature

Structured ``N^2`` pairing: permute which Gauss–Legendre ``z`` level is tied to each
Hermite fluctuation node in ``(T,q)`` (LHS-style staggering). See the
Latin-hypercube-``z`` branch of [`integrate_over_sgs_linear_profile`](@ref).
"""
struct SubgridLatinHypercubeZ <: AbstractSubgridLayerProfileQuadrature end

"""
    SubgridPrincipalAxisLayer <: AbstractSubgridLayerProfileQuadrature

Cheap inner fluctuation rule: one Gauss–Hermite line along the dominant correlation axis
instead of a full ``N \\times N`` inner tensor. Implemented in the principal-axis branch
of [`integrate_over_sgs_linear_profile`](@ref).
"""
struct SubgridPrincipalAxisLayer <: AbstractSubgridLayerProfileQuadrature end

"""
    SubgridVoronoiRepresentatives <: AbstractSubgridLayerProfileQuadrature

``N^2`` representatives chosen from a dense index pool via Voronoi-style clustering
(see the Voronoi branch of [`integrate_over_sgs_linear_profile`](@ref)).
"""
struct SubgridVoronoiRepresentatives <: AbstractSubgridLayerProfileQuadrature end

"""
    SubgridBarycentricSeeds <: AbstractSubgridLayerProfileQuadrature

Deterministic ``N^2`` seeds in ``(z, \\text{index})`` space with barycentric mass
accumulation from streamed candidates. See the barycentric-seeds branch of
[`integrate_over_sgs_linear_profile`](@ref); `quadrature_order` must match seed layout
in the kernel.
"""
struct SubgridBarycentricSeeds <: AbstractSubgridLayerProfileQuadrature end

"""
    AbstractConvolutionQuantileMethod

Selects the **per-leg** uniform–Gaussian quantile rule inside profile–Rosenblatt.
The inner marginal is always discretized as a **composite** of lower and upper
half-cell laws (½ weight each, same Gauss–Legendre nodes on ``[0,1]`` per leg); this
type chooses Brent, one-step Halley, or Chebyshev **on each single shifted**
`uniform ⊛ Gaussian` leg—not a scalar inversion of the mixture CDF `F_{mix}`.
"""
abstract type AbstractConvolutionQuantileMethod end

"""
Per-leg **Brent** inverse of each shifted `uniform ⊛ Gaussian` half law (exact bracketed
root on the single-component CDF). Slower than [`ConvolutionQuantilesHalley`](@ref) but
maximally robust for validation.
"""
struct ConvolutionQuantilesBracketed <: AbstractConvolutionQuantileMethod end

"""
Default / production per-leg rule: one **Halley** step on each single centered
`uniform[-L/2,L/2] ⊛ N(0,s²)` law (then map to the DN/UP shifted `u` coordinate).
"""
struct ConvolutionQuantilesHalley <: AbstractConvolutionQuantileMethod end

"""
Chebyshev surrogate in `τ` for **each** leg’s centered `uniform[-L/2,L/2] ⊛ N(0,s²)`
quantile at fixed Gauss–Legendre node index (same `N_gl` and node order as
[`gauss_legendre_01`](@ref)). See [`centered_uniform_gaussian_convolution_quantile_chebyshev`](@ref).
"""
struct ConvolutionQuantilesChebyshevLogEta <: AbstractConvolutionQuantileMethod end

"""
    SubgridProfileRosenblatt{B} <: AbstractSubgridLayerProfileQuadrature

Profile–Rosenblatt factorization: Gaussian outer quadrature in `v` × composite inner
rule on the two-component uniform–Gaussian **layer** marginal: for each outer `v`,
``\\tfrac12\\sum_i w_i h(u_{dn,i}) + \\tfrac12\\sum_i w_i h(u_{up,i})`` with `u_{dn,i}`,
`u_{up,i}` from the DN/UP single-law inverses at the same GL abscissas. `B` selects
Brent ([`ConvolutionQuantilesBracketed`](@ref)), one-step Halley per leg
([`ConvolutionQuantilesHalley`](@ref)), or Chebyshev per leg ([`ConvolutionQuantilesChebyshevLogEta`](@ref)).
"""
struct SubgridProfileRosenblatt{B <: AbstractConvolutionQuantileMethod} <:
       AbstractSubgridLayerProfileQuadrature end

"""
Default gridscale-corrected discretization: profile–Rosenblatt with per-leg **Halley**
inners (composite split marginal).
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
quadrature (default [`DefaultGridscaleProfileQuadrature`](@ref) = profile–Rosenblatt + per-leg Halley).
"""
struct GaussianGridscaleCorrectedSGS{S <: AbstractSubgridLayerProfileQuadrature} <:
       AbstractGridscaleCorrectedSGS end

GaussianGridscaleCorrectedSGS() = GaussianGridscaleCorrectedSGS{DefaultGridscaleProfileQuadrature}()

"""
    LogNormalGridscaleCorrectedSGS{S} <: AbstractGridscaleCorrectedSGS

Gridscale-corrected log-normal `q` / Gaussian `T`. Default matches [`GaussianGridscaleCorrectedSGS`](@ref)
(profile–Rosenblatt + per-leg Halley). [`SubgridColumnTensor`](@ref) is also available for explicit vertical quadrature.
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
