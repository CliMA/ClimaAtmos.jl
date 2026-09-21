"""
    AbstractSGSCondensateDistribution

How a cloud-condensate species (liquid or ice) is distributed over the SGS
quadrature nodes when the 1-moment microphysics is evaluated
(`Microphysics1MEvaluator`). Each species has its own choice:

  - [`ExcessCondensateDistribution`](@ref) (default for both): the species is its
    liquid-fraction share of the reconstructed saturation excess at every node
    (`λ` for liquid, `1 − λ` for ice), so it sits only where the node is
    saturated. Bitwise identical to the previous behaviour.
  - [`UniformCondensateDistribution`](@ref): the species is the subdomain mean
    (`q_lcl` or `q_icl`) at every node, held fixed across the quadrature like
    rain and snow. It then condenses/deposits at supersaturated nodes and
    evaporates/sublimates at subsaturated ones against the local vapour,
    instead of being tied to the local saturation excess.

Any combination conserves the quadrature mean of the local cloud condensate in
cells with condensate (`λ q_c = q_lcl`, `(1 − λ) q_c = q_icl`). Selected by the
`sgs_liquid_distribution` and `sgs_ice_distribution` configuration keys.
"""
abstract type AbstractSGSCondensateDistribution end

"""
    ExcessCondensateDistribution()

The species at a quadrature node is its liquid-fraction share of the
reconstructed saturation excess (the historical split).
"""
struct ExcessCondensateDistribution <: AbstractSGSCondensateDistribution end

"""
    UniformCondensateDistribution()

The species at every quadrature node is its subdomain mean, uniform across the
SGS distribution like rain and snow.
"""
struct UniformCondensateDistribution <: AbstractSGSCondensateDistribution end
Base.broadcastable(x::AbstractSGSCondensateDistribution) = tuple(x)

"""
    sgs_local_species(dist, share, q_mean)

Local amount of one cloud-condensate species at a quadrature node: `share` is
its liquid-fraction share of the reconstructed excess (`λ·excess` or
`(1 − λ)·excess`), `q_mean` its subdomain mean.
"""
@inline sgs_local_species(::ExcessCondensateDistribution, share, q_mean) = share
@inline sgs_local_species(::UniformCondensateDistribution, share, q_mean) = q_mean

"""
    sgs_local_cloud_condensate(liq, ice, λ, shifted_excess, q_lcl, q_icl)

Local cloud liquid and ice `(q_lcl_hat, q_icl_hat)` [kg/kg] at one quadrature
node from the reconstructed `shifted_excess = max(0, λ_lagrange + α S′)`, the
liquid fraction `λ` of the cell, the subdomain-mean cloud liquid and ice, and
the two species distributions.
"""
@inline function sgs_local_cloud_condensate(
    liq::AbstractSGSCondensateDistribution,
    ice::AbstractSGSCondensateDistribution,
    λ, shifted_excess, q_lcl, q_icl,
)
    q_lcl_hat = sgs_local_species(liq, λ * shifted_excess, q_lcl)
    q_icl_hat = sgs_local_species(ice, (one(λ) - λ) * shifted_excess, q_icl)
    return (q_lcl_hat, q_icl_hat)
end
