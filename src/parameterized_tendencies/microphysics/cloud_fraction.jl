import NVTX
import StaticArrays as SA
import ClimaCore.RecursiveApply: rzero, ⊞, ⊠

"""
    set_covariance_cache_and_cloud_fraction!(Y, p)

Update the covariance cache and cloud fraction in a way that is consistent with
the coupling between cloud fraction, buoyancy gradient, and mixing length.

The buoyancy gradient depends on the cloud fraction, while the cloud fraction
depends on the covariance cache, whose mixing length depends on the buoyancy
gradient. This circular dependency is resolved by performing two Picard
iterations on cloud fraction and then applying a guarded Aitken Δ²
acceleration,

    cₐ = c₀ - (c₁ - c₀)^2 / (c₂ - 2c₁ + c₀),

where `c₀` is the initial cloud fraction, `c₁ = f(c₀)`, and `c₂ = f(c₁)`.

The accelerated update is only applied when the first two Picard increments
change sign, since in that case the Aitken value lies between the previously
computed iterates. Otherwise, the second Picard iterate is retained.

For reproducible restart, the initial cloud fraction is first recomputed using
`GridScaleCloud()` so that the starting iterate is deterministic.

Note: Vertical gradients (`ᶜgradᵥ_q_tot`, `ᶜgradᵥ_θ_liq_ice`) are always computed
from grid-mean variables. Ideally PrognosticEDMFX would use environmental
gradients since the covariances represent sub-grid fluctuations within the
environment, but this is a current approximation.
"""
function set_covariance_cache_and_cloud_fraction!(Y, p)
    (; cloud_model, microphysics_model) = p.atmos
    (; ᶜgradᵥ_q_tot, ᶜgradᵥ_θ_liq_ice, ᶜcloud_fraction) = p.precomputed
    (; ᶜlinear_buoygrad, ᶜT, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice) = p.precomputed
    thermo_params = CAP.thermodynamics_params(p.params)
    ᶜlg = Fields.local_geometry_field(Y.c)

    # Precompute gradients
    @. ᶜgradᵥ_q_tot = ᶜgradᵥ(ᶠinterp(ᶜq_tot_nonneg))
    @. ᶜgradᵥ_θ_liq_ice = ᶜgradᵥ(
        ᶠinterp(
            TD.liquid_ice_pottemp(
                thermo_params,
                ᶜT,
                Y.c.ρ,
                ᶜq_tot_nonneg,
                ᶜq_liq,
                ᶜq_ice,
            ),
        ),
    )

    # The buoyancy gradient depends on cloud fraction, and cloud fraction depends
    # on the covariance cache through the mixing length. For reproducible restart,
    # first reconstruct the initial cloud fraction deterministically.
    if p.atmos.numerics.reproducible_restart isa ReproducibleRestart
        set_cloud_fraction!(Y, p, microphysics_model, GridScaleCloud())
    end

    # One Picard step: use the current cloud fraction to update buoyancy
    # gradient and covariance cache, then recompute cloud fraction.
    #
    # The hybrid cloud fraction reads `(μ_S, σ_S²)` from a Gauss-Hermite
    # pre-pass, but those moments are computed inline inside the CF broadcast
    # (see `compute_cloud_fraction_hybrid`) — there is no separate
    # `set_sgs_moments!` call during Picard. The microphysics-only moments
    # `(M_l, M_i)` are written once after Picard converges, in
    # `set_sgs_moments_mp!`.
    function picard_step!()
        @. ᶜlinear_buoygrad = buoyancy_gradients(
            BuoyGradMean(), # TODO: modify for NonEq + 1M tracers if needed
            thermo_params,
            ᶜT,
            Y.c.ρ,
            ᶜq_tot_nonneg,
            ᶜq_liq,
            ᶜq_ice,
            ᶜcloud_fraction,
            C3,
            ᶜgradᵥ_q_tot,
            ᶜgradᵥ_θ_liq_ice,
            ᶜlg,
        )

        # Cache SGS covariances (no-op for dry/0M/GridScaleCloud configs).
        # For EDMF: gradients are precomputed above.
        # For non-EDMF: gradients are computed inside set_covariance_cache!.
        set_covariance_cache!(Y, p, thermo_params)
        set_cloud_fraction!(Y, p, microphysics_model, cloud_model)
        return nothing
    end

    # Scratch storage for Picard/Aitken iterates:
    #   c0 = initial cloud fraction
    #   c1 = first Picard iterate
    #   c2 = second Picard iterate
    # ᶜtemp_scalar, ᶜtemp_scalar_2, ᶜtemp_scalar_3, ᶜtemp_scalar_5, ᶜtemp_scalar_6 might
    # change inside the functions that are called in picard_step!() and should not be used
    # here to store variables before calling picard_step!
    c0 = p.scratch.ᶜtemp_scalar_4
    c1 = p.scratch.ᶜtemp_scalar_7
    c2 = p.scratch.ᶜtemp_scalar

    # Picard iterates: c1 = f(c0), c2 = f(c1)
    @. c0 = ᶜcloud_fraction
    picard_step!()
    @. c1 = ᶜcloud_fraction
    picard_step!()
    @. c2 = ᶜcloud_fraction

    # Apply aitken Δ² acceleration for better convergence
    @. ᶜcloud_fraction = _aitken_picard_helper(c0, c1, c2)

    # Recompute buoyancy gradient and covariance cache with the final cloud fraction.
    @. ᶜlinear_buoygrad = buoyancy_gradients(
        BuoyGradMean(), # TODO: modify for NonEq + 1M tracers if needed
        thermo_params,
        ᶜT,
        Y.c.ρ,
        ᶜq_tot_nonneg,
        ᶜq_liq,
        ᶜq_ice,
        ᶜcloud_fraction,
        C3,
        ᶜgradᵥ_q_tot,
        ᶜgradᵥ_θ_liq_ice,
        ᶜlg,
    )
    set_covariance_cache!(Y, p, thermo_params)

    # Write the microphysics moments `(M_l, M_i)` once against the post-Aitken
    # variance closure. The downstream microphysics tendency reads
    # `ᶜsgs_moments_mp` for the shape-function partition. No-op when that
    # cache is not allocated (e.g. 0M / dry).
    set_sgs_moments_mp!(Y, p)

    return nothing
end

"""
Guarded Aitken Δ² acceleration:
  c_acc = c0 - (c1 - c0)^2 / (c2 - 2c1 + c0)

Apply Aitken only when the Picard increments change sign, i.e. when the
Picard iterates oscillate around the fixed point. In that case, the
accelerated value is expected to remain between previously computed iterates.
Otherwise, retain the second Picard iterate.
"""
@inline function _aitken_picard_helper(c0, c1, c2)
    FT = typeof(c0)
    Δ1 = c1 - c0
    Δ2 = c2 - c1
    denom = c2 - 2c1 + c0
    tol = eps(FT)
    return ifelse(
        (Δ1 * Δ2 < zero(FT)) & (abs(denom) > tol),
        c0 - Δ1^2 / denom,
        c2,
    )
end

# ============================================================================
# Utility Functions
# ============================================================================


"""
    compute_∂T_∂θ!(dest, Y, p, thermo_params)

Materialize the θ→T Jacobian (∂T/∂θ_li) into `dest`.

Always uses grid-mean variables, consistent with the gradient computation
(see `set_covariance_cache!`).
"""
function compute_∂T_∂θ!(dest, Y, p, thermo_params)
    (; ᶜT) = p.precomputed
    ᶜρ = Y.c.ρ
    if p.atmos.microphysics_model isa Union{DryModel, EquilibriumMicrophysics0M}
        (; ᶜq_liq, ᶜq_ice, ᶜq_tot_nonneg) = p.precomputed
        ᶜq_tot = ᶜq_tot_nonneg
    else
        ᶜq_liq = @. lazy(specific(Y.c.ρq_lcl, Y.c.ρ))
        ᶜq_ice = @. lazy(specific(Y.c.ρq_icl, Y.c.ρ))
        ᶜq_tot = @. lazy(specific(Y.c.ρq_tot, Y.c.ρ))
    end
    ᶜθ_li = @. lazy(
        TD.liquid_ice_pottemp(thermo_params, ᶜT, ᶜρ, ᶜq_tot, ᶜq_liq, ᶜq_ice),
    )
    @. dest = ∂T_∂θ_li(
        thermo_params, ᶜT, ᶜθ_li, ᶜq_liq, ᶜq_ice, ᶜq_tot, ᶜρ,
    )
    return dest
end

"""
    set_covariance_cache!(Y, p, thermo_params)

Materializes T-based SGS covariances into cached fields for use by downstream
computations (SGS quadrature, cloud fraction). Populates `p.precomputed.(ᶜT′T′, ᶜq′q′)`.

Pipeline:
1. Compute mixing length via `compute_gm_mixing_length` or `ᶜmixing_length`
2. Materialize θ-based covariances from gradients
3. Transform θ→T using `compute_∂T_∂θ!`
"""
function set_covariance_cache!(Y, p, thermo_params)
    # Covariance fields are only allocated when microphysics needs the
    # quadrature API or QuadratureCloud/MLCloud is active.
    # No-op otherwise (e.g. EquilMoist + 0M + GridScaleCloud).
    uses_covariances =
        !isnothing(p.atmos.sgs_quadrature) ||
        p.atmos.microphysics_model isa
        Union{NonEquilibriumMicrophysics1M, NonEquilibriumMicrophysics2M} ||
        p.atmos.cloud_model isa Union{QuadratureCloud, MLCloud}
    uses_covariances || return nothing

    (; ᶜT′T′, ᶜq′q′) = p.precomputed

    coeff = CAP.diagnostic_covariance_coeff(p.params)
    turbconv_model = p.atmos.turbconv_model
    (; ᶜgradᵥ_q_tot, ᶜgradᵥ_θ_liq_ice) = p.precomputed

    # NOTE: gradients must be precomputed when using compute_gm_mixing_length
    # compute_gm_mixing_length materializes into p.scratch.ᶜtemp_scalar
    ᶜmixing_length_field =
        turbconv_model isa PrognosticEDMFX || turbconv_model isa DiagnosticEDMFX ?
        ᶜmixing_length(Y, p) :
        compute_gm_mixing_length(Y, p)

    # Compute θ-based covariances from gradients and mixing length
    cov_from_grad(C, L, ∇Φ, ∇Ψ) = 2 * C * L^2 * dot(∇Φ, ∇Ψ)

    # Materialize q′q′ into cache (same in θ and T basis)
    @. ᶜq′q′ = cov_from_grad(
        coeff,
        ᶜmixing_length_field,
        Geometry.WVector(ᶜgradᵥ_q_tot),
        Geometry.WVector(ᶜgradᵥ_q_tot),
    )
    # Materialize θ′θ′ into ᶜT′T′ temporarily
    @. ᶜT′T′ = cov_from_grad(
        coeff,
        ᶜmixing_length_field,
        Geometry.WVector(ᶜgradᵥ_θ_liq_ice),
        Geometry.WVector(ᶜgradᵥ_θ_liq_ice),
    )
    # Transform θ′θ′ → T′T′ in-place using Jacobian ∂T/∂θ
    ᶜ∂T_∂θ = p.scratch.ᶜtemp_scalar_2
    compute_∂T_∂θ!(ᶜ∂T_∂θ, Y, p, thermo_params)
    @. ᶜT′T′ = ᶜ∂T_∂θ^2 * ᶜT′T′  # θ′θ′ → T′T′
    return nothing
end


# ============================================================================
# SGS Moments — pre-pass quadrature
# ============================================================================
#
# A Gauss-Hermite pass over the SGS PDF (no BMT calls) supplies the moments
# consumed by the cloud fraction and microphysics tendency closures. Two
# specialized evaluators run separately so each consumer pays for only the
# moments it uses:
#
#   - `SGSMomentsCFEvaluator` returns `(μ_S, S²)` for the smooth logistic-CDF
#     cloud fraction (used inline in the CF broadcast — never materialized
#     to a Field).
#
#   - `SGSMomentsMPEvaluator` returns `(M_l, M_i)` for the
#     `Microphysics1MEvaluator` shape-function partition. Cached once per
#     time step in `p.precomputed.ᶜsgs_moments_mp`.
#
# Both consumers share the same SGS PDF, so the closures remain mutually
# consistent.

"""
    SGSMomentsType{FT}

NamedTuple type alias for the full per-cell SGS moments NamedTuple returned
by [`compute_sgs_moments`](@ref). Field semantics:

- `mu_S`: Mean of the saturation variable `S(ξ)` over the SGS PDF.
- `sigma_S_sq`: Variance of `S(ξ)` over the SGS PDF.
- `M_l`: Mean of the equilibrium cloud-liquid condensate over the SGS PDF
  (`⟨max(0, λ·(q_tot_hat − q_sat(T_hat)) − q_rai)⟩`).
- `M_i`: Mean of the equilibrium cloud-ice condensate over the SGS PDF.

The saturation variable `S` depends on the SGS distribution:
- `GaussianSGS`, `GridMeanSGS`: `S = q_tot − q_sat(T)` (linear saturation excess; units of `q`).
- `LogNormalSGS`: `S = log(q_tot / q_sat(T))` (dimensionless log-ratio).

`M_l` and `M_i` are always in linear units of specific humidity, regardless
of the SGS distribution.
"""
const SGSMomentsType{FT} = @NamedTuple{
    mu_S::FT, sigma_S_sq::FT, M_l::FT, M_i::FT,
} where {FT}

"""
    SGSMomentsMPType{FT}

NamedTuple type alias for the per-cell SGS microphysics-moments cache used
by `Microphysics1MEvaluator`. Fields:

- `M_l`: SGS-mean equilibrium cloud-liquid condensate `⟨q_lcl_eq⟩`.
- `M_i`: SGS-mean equilibrium cloud-ice condensate `⟨q_icl_eq⟩`.
"""
const SGSMomentsMPType{FT} = @NamedTuple{M_l::FT, M_i::FT} where {FT}

# Linear saturation excess (Gaussian / GridMean SGS distributions)
@inline _saturation_variable(
    ::Union{GaussianSGS, GridMeanSGS}, q_tot, q_sat, q_min,
) = q_tot - q_sat
# Log saturation ratio (Lognormal SGS distribution); arguments regularized
# at `q_min` to prevent `log(0)` in dry / extreme states.
@inline _saturation_variable(::LogNormalSGS, q_tot, q_sat, q_min) =
    log(max(q_tot, q_min) / max(q_sat, q_min))

"""
    SGSMomentsCFEvaluator(dist, tps, ρ, q_min)

GPU-safe functor for the cloud-fraction-only quadrature integrand. Returns
`(mu_S = S(ξ), s_sq = S(ξ)²)` at each quadrature point. The CF closure does
not need the equilibrium-condensate moments, so this evaluator skips the
`(M_l, M_i)` arithmetic entirely.
"""
struct SGSMomentsCFEvaluator{D, TPS, FT}
    dist::D
    tps::TPS
    ρ::FT
    q_min::FT
end

@inline function (eval::SGSMomentsCFEvaluator)(T_hat, q_tot_hat)
    q_sat_hat = TD.q_vap_saturation(eval.tps, T_hat, eval.ρ)
    s = _saturation_variable(eval.dist, q_tot_hat, q_sat_hat, eval.q_min)
    return (mu_S = s, s_sq = s * s)
end

"""
    SGSMomentsMPEvaluator(tps, ρ, λ, q_rai, q_sno)

GPU-safe functor for the microphysics-only quadrature integrand. Returns
`(M_l, M_i) = (q_lcl_eq(ξ), q_icl_eq(ξ))` at each quadrature point. The
partition is linear in the saturation excess (independent of `dist`), so the
log-saturation arithmetic is skipped.
"""
struct SGSMomentsMPEvaluator{TPS, FT}
    tps::TPS
    ρ::FT
    λ::FT
    q_rai::FT
    q_sno::FT
end

@inline function (eval::SGSMomentsMPEvaluator)(T_hat, q_tot_hat)
    FT = typeof(eval.ρ)
    q_sat_hat = TD.q_vap_saturation(eval.tps, T_hat, eval.ρ)
    excess = max(zero(FT), q_tot_hat - q_sat_hat)
    M_l = max(zero(FT), eval.λ * excess - eval.q_rai)
    M_i = max(zero(FT), (one(FT) - eval.λ) * excess - eval.q_sno)
    return (; M_l, M_i)
end

"""
    SGSMomentsEvaluator(dist, tps, ρ, λ, q_rai, q_sno, q_min)

Combined functor returning all four moment integrands
`(mu_S, s_sq, M_l, M_i)` for `compute_sgs_moments`. Retained mainly for the
test surface — production paths use the specialized CF or MP evaluators.
"""
struct SGSMomentsEvaluator{D, TPS, FT}
    dist::D
    tps::TPS
    ρ::FT
    λ::FT
    q_rai::FT
    q_sno::FT
    q_min::FT
end

@inline function (eval::SGSMomentsEvaluator)(T_hat, q_tot_hat)
    FT = typeof(eval.ρ)
    q_sat_hat = TD.q_vap_saturation(eval.tps, T_hat, eval.ρ)
    s = _saturation_variable(eval.dist, q_tot_hat, q_sat_hat, eval.q_min)
    excess = max(zero(FT), q_tot_hat - q_sat_hat)
    q_lcl_eq = max(zero(FT), eval.λ * excess - eval.q_rai)
    q_icl_eq = max(zero(FT), (one(FT) - eval.λ) * excess - eval.q_sno)
    return (mu_S = s, s_sq = s * s, M_l = q_lcl_eq, M_i = q_icl_eq)
end

"""
    _cf_moments_inline(thp, ρ, T_mean, q_tot_mean, sgs_quad, T′T′, q′q′,
                       corr_Tq, q_min)

Compute `(mu_S, sigma_S_sq)` via a single quadrature pass over the SGS PDF.
Used inline by `compute_cloud_fraction_hybrid` so the CF broadcast fuses the
moments calculation with the CDF evaluation into a single GPU kernel and never
materializes a moments Field.
"""
@inline function _cf_moments_inline(
    thp, ρ, T_mean, q_tot_mean,
    sgs_quad, T′T′, q′q′, corr_Tq, q_min,
)
    FT = typeof(ρ)
    sgs_quad_eff = isnothing(sgs_quad) ? GridMeanSGS() : sgs_quad
    dist = sgs_quad_eff isa SGSQuadrature ? sgs_quad_eff.dist : sgs_quad_eff
    evaluator = SGSMomentsCFEvaluator(dist, thp, ρ, q_min)
    raw = integrate_over_sgs(
        evaluator, sgs_quad_eff, q_tot_mean, T_mean, q′q′, T′T′, corr_Tq,
    )
    sigma_S_sq = max(raw.s_sq - raw.mu_S * raw.mu_S, ϵ_numerics(FT))
    return (mu_S = raw.mu_S, sigma_S_sq = sigma_S_sq)
end

"""
    compute_sgs_moments_mp(thp, ρ, T_mean, q_tot_mean,
                           q_lcl, q_icl, q_rai, q_sno,
                           sgs_quad, T′T′, q′q′, corr_Tq)

Compute the microphysics moments `(M_l, M_i)` via one quadrature pass.
Linear in the saturation excess, independent of the SGS distribution.
Used by [`set_sgs_moments_mp!`](@ref) to populate `p.precomputed.ᶜsgs_moments_mp`
once per time step.
"""
@inline function compute_sgs_moments_mp(
    thp, ρ, T_mean, q_tot_mean,
    q_lcl, q_icl, q_rai, q_sno,
    sgs_quad, T′T′, q′q′, corr_Tq,
)
    FT = typeof(ρ)
    sgs_quad_eff = isnothing(sgs_quad) ? GridMeanSGS() : sgs_quad
    q_lcl_nn = max(zero(FT), q_lcl)
    q_icl_nn = max(zero(FT), q_icl)
    q_rai_nn = max(zero(FT), q_rai)
    q_sno_nn = max(zero(FT), q_sno)
    λ = TD.liquid_fraction(thp, T_mean, q_lcl_nn, q_icl_nn)
    evaluator = SGSMomentsMPEvaluator(thp, ρ, λ, q_rai_nn, q_sno_nn)
    return integrate_over_sgs(
        evaluator, sgs_quad_eff, q_tot_mean, T_mean, q′q′, T′T′, corr_Tq,
    )::SGSMomentsMPType{FT}
end

"""
    compute_sgs_moments(thp, ρ, T_mean, q_tot_mean,
                          q_lcl, q_icl, q_rai, q_sno,
                          sgs_quad, T′T′, q′q′, corr_Tq, q_min)

Compute all four SGS moments `(mu_S, sigma_S_sq, M_l, M_i)` via one quadrature
pass. Retained as a single building block for tests and external callers;
production code uses [`_cf_moments_inline`](@ref) (for the cloud fraction) and
[`compute_sgs_moments_mp`](@ref) (for microphysics) instead, so neither pays for
unused moments.
"""
@inline function compute_sgs_moments(
    thp, ρ, T_mean, q_tot_mean,
    q_lcl, q_icl, q_rai, q_sno,
    sgs_quad, T′T′, q′q′, corr_Tq, q_min,
)
    FT = typeof(ρ)
    sgs_quad_eff = isnothing(sgs_quad) ? GridMeanSGS() : sgs_quad
    dist = sgs_quad_eff isa SGSQuadrature ? sgs_quad_eff.dist : sgs_quad_eff
    q_lcl_nn = max(zero(FT), q_lcl)
    q_icl_nn = max(zero(FT), q_icl)
    q_rai_nn = max(zero(FT), q_rai)
    q_sno_nn = max(zero(FT), q_sno)
    λ = TD.liquid_fraction(thp, T_mean, q_lcl_nn, q_icl_nn)
    evaluator = SGSMomentsEvaluator(dist, thp, ρ, λ, q_rai_nn, q_sno_nn, q_min)
    raw = integrate_over_sgs(
        evaluator, sgs_quad_eff, q_tot_mean, T_mean, q′q′, T′T′, corr_Tq,
    )
    sigma_S_sq = max(raw.s_sq - raw.mu_S * raw.mu_S, ϵ_numerics(FT))
    return (
        mu_S = raw.mu_S,
        sigma_S_sq = sigma_S_sq,
        M_l = raw.M_l,
        M_i = raw.M_i,
    )::SGSMomentsType{FT}
end


# ============================================================================
# Cloud Fraction: Hybrid (quadrature-moment + analytic CDF)
# ============================================================================
#
# Smooth logistic-CDF approximation evaluated against the cached SGS moments
# `(μ_S, σ_S²)` produced by `compute_sgs_moments`. The variance and saturation
# mean come directly from the Gauss-Hermite quadrature over the same SGS PDF
# that microphysics integrates over, giving consistency between the cloud
# fraction and the microphysics tendencies by construction. 
#
# Dispatched on the SGS distribution via `_effective_excess_hybrid`:
#   - Gaussian/GridMean: linear coords, `Q_eff = q_c + min(0, μ_S)`.
#   - Lognormal: log coords, `Q_eff = log(1 + q_c/q_v_sat) + min(0, μ_S)`.
# The lognormal branch handles the heavy upper tail of `q_tot` in a
# principled way.

"""
    compute_cloud_fraction_hybrid(
        thermo_params, T, ρ, q_tot, q_liq, q_ice,
        moments, cf_guess, cf_steepness_scale, q_min, sgs_dist,
    )

Compute cloud fraction from cached SGS moments. The variance and mean of
the saturation variable are taken from the pre-pass quadrature
(`compute_sgs_moments`); only the smooth CDF approximation and the
activation factor are evaluated here.

# Algorithm
1. Activation factor on linear excess from cached moments:
   `α = clamp(q_c / √((μ_{lin})_+² + q_min²), 0, 1)` where `μ_{lin}` is
   the saturation excess (kg/kg) derived from `μ_S`.
2. Effective activation-scaled SGS standard deviation:
   `σ_qc = α √(σ_S²)`,
   - Gaussian/GridMean (linear coordinates):
   `σ_eff = √(σ_qc^2 + c_f (1-c_f) (q_c / c_f - min(0, μ_S) / (1-c_f))^2)`
   - Lognormal:
   `σ_eff = √(σ_qc^2 + c_f (1-c_f) (log(1 + q_c/q_v_sat) / c_f - min(0, μ_S) / (1-c_f))^2)`
3. Effective excess `Q_eff` from cached `μ_S`, dispatched on `sgs_dist`:
   - Gaussian/GridMean: `Q_eff = q_c + min(0, μ_S)` (linear coordinates).
   - Lognormal: `Q_eff = log(1 + q_c/q_v_sat) + min(0, μ_S)` (log coordinates).
4. Cloud fraction via variance-matched logistic CDF
   (`erf(x/√2) ≈ tanh((π/(2√3)) x)`):
   `c_f = ½ [1 + tanh(coeff · Q_eff / σ_eff)]` with
   `coeff = (π/(2√3)) · cf_steepness_scale`.
5. Enforce zero cloud fraction when no condensate exists.

# Arguments
- `thermo_params`: Thermodynamics parameters.
- `T`: Grid-mean temperature [K].
- `ρ`: Air density [kg/m³].
- `q_tot`: Grid-mean total specific humidity [kg/kg].
- `q_liq`, `q_ice`: Grid-mean cloud condensate [kg/kg].
- `moments`: Cached `SGSMomentsType` `(mu_S, sigma_S_sq, M_l, M_i)`.
- `cf_guess`: Previous-step cloud fraction used in the effective variance estimate.
- `cf_steepness_scale`: Tunable steepness factor `coeff` (default 1).
- `q_min`: Minimum specific humidity threshold [kg/kg].
- `sgs_dist`: SGS distribution (`GaussianSGS`, `LogNormalSGS`, `GridMeanSGS`).

# Returns
Cloud fraction ∈ [0, 1].
"""
@inline function compute_cloud_fraction_hybrid(
    thermo_params,
    T,
    ρ,
    q_tot,
    q_liq,
    q_ice,
    moments,
    cf_guess,
    cf_steepness_scale,
    q_min,
    sgs_dist::AbstractSGSDistribution,
)
    FT = typeof(T)

    q_c = q_liq + q_ice

    # --- 1. Activation factor: linear excess from cached quadrature moments
    # We recover the linear excess (kg/kg) from the distribution-specific μ_S.
    q_sat = TD.q_vap_saturation(thermo_params, T, ρ)
    linear_mu = _linear_mu(sgs_dist, moments.mu_S, q_sat)
    excess_eq = max(zero(FT), linear_mu)
    α = min(one(FT), q_c / sqrt(excess_eq * excess_eq + q_min * q_min))

    # --- 2. Effective activation-scaled SGS standard deviation
    σ_S = sqrt(max(moments.sigma_S_sq, ϵ_numerics(FT)))
    σ_qc = α * σ_S
    σ_eff = _effective_std_hybrid(σ_qc, cf_guess, q_c, q_sat, moments.mu_S, q_min, sgs_dist)

    # --- 3. Effective excess Q_eff in the coordinate of the SGS distribution
    Q_eff =
        _effective_excess_hybrid(q_c, q_sat, moments.mu_S, q_min, sgs_dist)

    # --- 4. Logistic-CDF approximation
    coeff = (FT(π) / (FT(2) * sqrt(FT(3)))) * cf_steepness_scale
    σ_safe = max(σ_eff, ϵ_numerics(FT))
    cf = FT(0.5) * (one(FT) + tanh(coeff * Q_eff / σ_safe))

    # --- 5. No condensate → no cloud (branchless) ---
    has_cond = TD.has_condensate(thermo_params, q_c)
    return ifelse(has_cond, cf, zero(FT))
end

# Recover the linear saturation excess (kg/kg) from the distribution-specific
# saturation variable. Used by the hybrid cloud fraction activation factor.
@inline _linear_mu(::Union{GaussianSGS, GridMeanSGS}, μ_S, q_sat) = μ_S
@inline _linear_mu(::LogNormalSGS, μ_S, q_sat) =
    q_sat * (exp(μ_S) - one(typeof(μ_S)))

"""
    compute_cloud_fraction_hybrid(
        thermo_params, T, ρ, q_tot, q_liq, q_ice,
        sgs_quad, T′T′, q′q′, corr_Tq, cf_guess,
        cf_steepness_scale, q_min, sgs_dist,
    )

Fused production overload: compute the hybrid cloud fraction in a single
inlined call that runs the `(μ_S, σ_S²)` quadrature pass and the logistic-CDF
evaluation in one broadcast kernel. Used by `set_cloud_fraction!(QuadratureCloud)`
so the moments are never materialized to a Field.
"""
@inline function compute_cloud_fraction_hybrid(
    thermo_params,
    T,
    ρ,
    q_tot,
    q_liq,
    q_ice,
    sgs_quad,
    T′T′,
    q′q′,
    corr_Tq,
    cf_guess,
    cf_steepness_scale,
    q_min,
    sgs_dist::AbstractSGSDistribution,
)
    moments = _cf_moments_inline(
        thermo_params, ρ, T, q_tot, sgs_quad, T′T′, q′q′, corr_Tq, q_min,
    )
    return compute_cloud_fraction_hybrid(
        thermo_params, T, ρ, q_tot, q_liq, q_ice, moments,
        cf_guess, cf_steepness_scale, q_min, sgs_dist,
    )
end

"""
    _effective_excess_hybrid(q_c, q_sat, μ_S, q_min, sgs_dist)

Distribution-specific effective excess `Q_eff` used by the hybrid cloud
fraction. Combines the prognostic condensate with the quadrature-derived
mean of the saturation variable so that `cf → 0` smoothly in subsaturated
states even when a numerical trace of prognostic condensate remains.

- `GaussianSGS`, `GridMeanSGS`: linear coordinates,
  `Q_eff = q_c + min(0, μ_S)`.
- `LogNormalSGS`: log coordinates,
  `Q_eff = log(1 + q_c/q_v_sat) + min(0, μ_S)` (regularized at `q_min`).
"""
@inline _effective_excess_hybrid(
    q_c, q_sat, μ_S, q_min, ::Union{GaussianSGS, GridMeanSGS},
) = q_c + min(zero(typeof(q_c)), μ_S)

@inline function _effective_excess_hybrid(
    q_c, q_sat, μ_S, q_min, ::LogNormalSGS,
)
    FT = typeof(q_c)
    q_v_sat_safe = max(q_sat, q_min)
    return log(FT(1) + q_c / q_v_sat_safe) + min(zero(FT), μ_S)
end

"""
    _effective_std_hybrid(σ_qc, cf, q_c, q_sat, μ_S, q_min, sgs_dist)

Distribution-specific effective standard deviation `σ_eff` used by the hybrid
cloud fraction. The formulation augments the activation-scaled SGS variance
with a between-state variance term arising from the separation between the
prognostic condensate contribution and the quadrature-derived saturation-state
mean.

- Gaussian/GridMean (linear coordinates):
`σ_eff = √(σ_qc^2 + c_f (1-c_f) (qc / c_f - min(0, μ_S) / (1-c_f))^2)`
- Lognormal:
`σ_eff = √(σ_qc^2 + c_f (1-c_f) (log(1 + q_c/q_v_sat) / c_f - min(0, μ_S) / (1-c_f))^2)`
"""
@inline function _effective_std_hybrid(
    σ_qc, cf, q_c, q_sat, μ_S, q_min, ::Union{GaussianSGS, GridMeanSGS},
)
    FT = typeof(q_c)
    # clamp cf between 0.01 and 0.99 to avoid division by small numbers
    cf = clamp(cf, FT(0.01), FT(0.99))
    return sqrt(σ_qc^2 + cf * (1 - cf) * (q_c / cf - min(zero(FT), μ_S) / (1 - cf))^2)
end

@inline function _effective_std_hybrid(
    σ_qc, cf, q_c, q_sat, μ_S, q_min, ::LogNormalSGS,
)
    FT = typeof(q_c)
    # clamp cf between 0.01 and 0.99 to avoid division by small numbers
    cf = clamp(cf, FT(0.01), FT(0.99))
    q_v_sat_safe = max(q_sat, q_min)
    return sqrt(
        σ_qc^2 +
        cf * (1 - cf) *
        (log(FT(1) + q_c / q_v_sat_safe) / cf - min(zero(FT), μ_S) / (1 - cf))^2,
    )
end

# ============================================================================
# Cloud Fraction Dispatch Methods
# ============================================================================

"""
    set_cloud_fraction!(Y, p, microphysics_model, cloud_model)

Compute and store grid-scale cloud fraction based on sub-grid scale properties.

Dispatches on `microphysics_model` and `cloud_model`:
- `DryModel`: Cloud fraction and cloud condensate are zero.
- `GridScaleCloud`: Cloud fraction is 1 if grid-scale condensate exists, 0 otherwise.
- `QuadratureCloud`: Cloud fraction from the hybrid quadrature-moment formula
  (`compute_cloud_fraction_hybrid`).
- `MLCloud`: Cloud fraction from neural network.

For EDMF turbulence models, updraft contributions are added to the environment values.
"""
NVTX.@annotate function set_cloud_fraction!(Y, p, ::DryModel, _)
    FT = eltype(p.params)
    p.precomputed.ᶜcloud_fraction .= FT(0)
end
NVTX.@annotate function set_cloud_fraction!(
    Y,
    p,
    ::MoistMicrophysics,
    ::GridScaleCloud,
)
    (; ᶜq_liq, ᶜq_ice) = p.precomputed
    thermo_params = CAP.thermodynamics_params(p.params)
    FT = eltype(p.params)
    @. p.precomputed.ᶜcloud_fraction =
        ifelse(
            TD.has_condensate(thermo_params, ᶜq_liq + ᶜq_ice),
            FT(1),
            FT(0),
        )
end
NVTX.@annotate function set_cloud_fraction!(
    Y,
    p,
    ::MoistMicrophysics,
    ::QuadratureCloud,
)
    thermo_params = CAP.thermodynamics_params(p.params)
    turbconv_model = p.atmos.turbconv_model
    microphysics_model = p.atmos.microphysics_model

    # Get environment density, temperature, and total specific humidity
    ᶜρ_env, ᶜT_mean, ᶜq_mean = _get_env_ρ_T_q(Y, p, thermo_params, turbconv_model)

    # Get condensate means (dispatches on microphysics_model)
    ᶜq_lcl, ᶜq_icl = _get_condensate_means(Y, p, turbconv_model, microphysics_model)

    sgs_quad = p.atmos.sgs_quadrature
    sgs_dist =
        isnothing(sgs_quad) ? GaussianSGS() : sgs_quad.dist

    cf_steepness_scale = CAP.cloud_fraction_steepness_scale(p.params)
    q_min = CAP.q_min(p.params)
    corr_Tq = correlation_Tq(p.params)

    (; ᶜT′T′, ᶜq′q′) = p.precomputed
    cf_guess = p.precomputed.ᶜcloud_fraction
    # Hybrid cloud fraction: the `(μ_S, σ_S²)` quadrature pass is fused into
    # this broadcast kernel via the production `compute_cloud_fraction_hybrid`
    # overload, so the moments stay in registers and are never written to a
    # Field. The `Microphysics1MEvaluator` consumes a separately-cached
    # `(M_l, M_i)` populated once per time step by `set_sgs_moments_mp!`.
    @. p.precomputed.ᶜcloud_fraction = compute_cloud_fraction_hybrid(
        thermo_params,
        ᶜT_mean,
        ᶜρ_env,
        ᶜq_mean,
        ᶜq_lcl,
        ᶜq_icl,
        $(sgs_quad),
        ᶜT′T′,
        ᶜq′q′,
        corr_Tq,
        cf_guess,
        cf_steepness_scale,
        q_min,
        $(sgs_dist),
    )

    _apply_edmf_cloud_weighting!(Y, p, turbconv_model, thermo_params)
end

NVTX.@annotate function set_cloud_fraction!(
    Y,
    p,
    ::MoistMicrophysics,
    qc::MLCloud,
)
    thermo_params = CAP.thermodynamics_params(p.params)
    turbconv_model = p.atmos.turbconv_model
    microphysics_model = p.atmos.microphysics_model

    # Get environment state, condensate, and covariances
    ᶜρ_env, ᶜT_mean, ᶜq_mean, ᶜθ_mean, ᶜq_lcl, ᶜq_icl, ᶜT′T′, ᶜq′q′ =
        _compute_cloud_state(Y, p, thermo_params, turbconv_model, microphysics_model)

    set_ml_cloud_fraction!(
        Y,
        p,
        qc,
        thermo_params,
        turbconv_model,
        ᶜρ_env,
        ᶜT_mean,
        ᶜq_mean,
        ᶜθ_mean,
    )
    _apply_edmf_cloud_weighting!(Y, p, turbconv_model, thermo_params)
end

"""
    set_sgs_moments_mp!(Y, p)

Cache the microphysics moments `(M_l, M_i)` into `p.precomputed.ᶜsgs_moments_mp`
via one Gauss-Hermite pass over the SGS PDF (no BMT calls). The cached moments
are consumed by `Microphysics1MEvaluator` (via `microphysics_tendencies_1m`):
the shape-function partition `q_lcl_hat = q_lcl_mean + q_lcl_mean · γ_l ·
(q_lcl_eq_hat − M_l)` exactly conserves `⟨q_lcl_hat⟩ = q_lcl_mean` provided
`M_l = ⟨q_lcl_eq⟩` over the same PDF.

Called once per time step after the Picard-accelerated cloud fraction update
in `set_covariance_cache_and_cloud_fraction!`. The cloud-fraction-specific
moments `(μ_S, σ_S²)` are computed inline inside the CF broadcast and are
never written to a Field, so this function does not compute them.

No-op when `ᶜsgs_moments_mp` is not allocated (i.e. configurations without
1M / 2M microphysics — 0M and dry).

Uses grid-mean `(ρ, T, q_tot, q_rai, q_sno)`; for EDMF configurations this
currently uses grid-mean covariances and state. A future refinement could use
environment-only covariances and state.
"""
NVTX.@annotate function set_sgs_moments_mp!(Y, p)
    hasproperty(p.precomputed, :ᶜsgs_moments_mp) || return nothing

    thermo_params = CAP.thermodynamics_params(p.params)
    (; ᶜT, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice) = p.precomputed
    (; ᶜT′T′, ᶜq′q′, ᶜsgs_moments_mp) = p.precomputed
    sgs_quad = p.atmos.sgs_quadrature
    corr_Tq = correlation_Tq(p.params)

    ᶜq_rai = @. lazy(specific(Y.c.ρq_rai, Y.c.ρ))
    ᶜq_sno = @. lazy(specific(Y.c.ρq_sno, Y.c.ρ))
    @. ᶜsgs_moments_mp = compute_sgs_moments_mp(
        thermo_params, Y.c.ρ, ᶜT, ᶜq_tot_nonneg,
        ᶜq_liq, ᶜq_ice, ᶜq_rai, ᶜq_sno,
        $(sgs_quad), ᶜT′T′, ᶜq′q′, corr_Tq,
    )
    return nothing
end

# ============================================================================
# Internal Helper Functions
# ============================================================================

"""
    _get_env_ρ_T_q(Y, p, thermo_params, turbconv_model)

Get environment density, temperature, and specific humidity for cloud fraction.
Lightweight alternative to `_compute_cloud_state` when only ρ, T, and q are needed.
"""
function _get_env_ρ_T_q(Y, p, thermo_params, turbconv_model)
    (; ᶜp, ᶜT, ᶜq_tot_nonneg) = p.precomputed
    if turbconv_model isa PrognosticEDMFX
        (; ᶜT⁰, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰) = p.precomputed
        ᶜρ_env = @. lazy(
            TD.air_density(
                thermo_params,
                ᶜT⁰,
                ᶜp,
                ᶜq_tot_nonneg⁰,
                ᶜq_liq⁰,
                ᶜq_ice⁰,
            ),
        )
        return ᶜρ_env, ᶜT⁰, ᶜq_tot_nonneg⁰
    else
        return Y.c.ρ, ᶜT, ᶜq_tot_nonneg
    end
end

"""
    _compute_cloud_state(Y, p, thermo_params, turbconv_model, microphysics_model)

Compute environment state, condensate means, and variances for cloud fraction.

For PrognosticEDMFX, uses environment (⁰) fields; otherwise uses grid-scale fields.

# Returns
Tuple: `(ᶜρ_env, ᶜT_mean, ᶜq_mean, ᶜθ_mean, ᶜq_lcl, ᶜq_icl, ᶜT′T′, ᶜq′q′)`
"""
function _compute_cloud_state(Y, p, thermo_params, turbconv_model, microphysics_model)
    (; ᶜp, ᶜT, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice) = p.precomputed

    if turbconv_model isa PrognosticEDMFX
        (; ᶜT⁰, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰) = p.precomputed
        ᶜρ_env = @. lazy(
            TD.air_density(
                thermo_params,
                ᶜT⁰,
                ᶜp,
                ᶜq_tot_nonneg⁰,
                ᶜq_liq⁰,
                ᶜq_ice⁰,
            ),
        )
        ᶜT_mean = ᶜT⁰
        ᶜq_mean = ᶜq_tot_nonneg⁰
        ᶜθ_mean = @. lazy(
            TD.liquid_ice_pottemp(
                thermo_params,
                ᶜT⁰,
                ᶜρ_env,
                ᶜq_tot_nonneg⁰,
                ᶜq_liq⁰,
                ᶜq_ice⁰,
            ),
        )
    else
        ᶜρ_env = Y.c.ρ
        ᶜT_mean = ᶜT
        ᶜq_mean = ᶜq_tot_nonneg
        ᶜθ_mean = @. lazy(
            TD.liquid_ice_pottemp(
                thermo_params,
                ᶜT,
                Y.c.ρ,
                ᶜq_tot_nonneg,
                ᶜq_liq,
                ᶜq_ice,
            ),
        )
    end

    # Get condensate means
    ᶜq_lcl, ᶜq_icl = _get_condensate_means(Y, p, turbconv_model, microphysics_model)

    # Get T-based variances from cache
    (; ᶜT′T′, ᶜq′q′) = p.precomputed

    return ᶜρ_env, ᶜT_mean, ᶜq_mean, ᶜθ_mean, ᶜq_lcl, ᶜq_icl, ᶜT′T′, ᶜq′q′
end

"""
    _get_condensate_means(Y, p, turbconv_model, microphysics_model)

Dispatch condensate mean retrieval based on microphysics model.
"""
_get_condensate_means(Y, p, turbconv_model, ::EquilibriumMicrophysics0M) =
    _get_condensate_means_equil(p, turbconv_model)
_get_condensate_means(Y, p, turbconv_model, ::NonEquilibriumMicrophysics) =
    _get_condensate_means_nonequil(Y, p, turbconv_model)

"""
    _get_condensate_means_equil(p, turbconv_model)

Retrieve grid-mean cloud condensate for EquilibriumMicrophysics0M.

For PrognosticEDMFX, uses environment condensate fields (ᶜq_liq⁰, ᶜq_ice⁰).
Otherwise (including DiagnosticEDMFX), uses grid-scale precomputed condensate.

# Returns
Tuple: `(ᶜq_lcl_mean, ᶜq_icl_mean)` as lazy field expressions.
"""
function _get_condensate_means_equil(p, turbconv_model)
    if turbconv_model isa PrognosticEDMFX
        (; ᶜq_liq⁰, ᶜq_ice⁰) = p.precomputed
        return ᶜq_liq⁰, ᶜq_ice⁰
    else
        (; ᶜq_liq, ᶜq_ice) = p.precomputed
        return ᶜq_liq, ᶜq_ice
    end
end

"""
    _get_condensate_means_nonequil(Y, p, turbconv_model)

Retrieve grid-mean cloud condensate for NonEquilibriumMicrophysics.

For PrognosticEDMFX, uses environment condensate fields (ᶜq_liq⁰, ᶜq_ice⁰).
Otherwise (including DiagnosticEDMFX), computes cloud-only condensate from prognostic variables.

# Returns
Tuple: `(ᶜq_lcl_mean, ᶜq_icl_mean)` as lazy field expressions.
"""
function _get_condensate_means_nonequil(Y, p, turbconv_model)
    if turbconv_model isa PrognosticEDMFX # TODO Shouldn't we do this for DiagnosticEDMFX too?
        (; ᶜq_liq⁰, ᶜq_ice⁰) = p.precomputed
        return ᶜq_liq⁰, ᶜq_ice⁰
    else
        ᶜq_lcl_mean = @. lazy(specific(Y.c.ρq_lcl, Y.c.ρ))
        ᶜq_icl_mean = @. lazy(specific(Y.c.ρq_icl, Y.c.ρ))
        return ᶜq_lcl_mean, ᶜq_icl_mean
    end
end

"""
    _apply_edmf_cloud_weighting!(Y, p, turbconv_model, thermo_params)

Apply EDMF-specific adjustments to cloud diagnostics.

For PrognosticEDMFX:
1. Weights environment cloud diagnostics by environment area fraction
2. Adds updraft contributions weighted by their respective area fractions

For DiagnosticEDMFX:
1. Adds updraft contributions (environment area fraction assumed = 1)

Updraft cloud fraction is binary: 1 if updraft contains condensate, 0 otherwise.
"""
function _apply_edmf_cloud_weighting!(Y, p, turbconv_model, thermo_params)
    (; ᶜp) = p.precomputed

    # Weight by environment area fraction if using PrognosticEDMFX (assumed 1 otherwise)
    if turbconv_model isa PrognosticEDMFX
        ᶜρa⁰ = @. lazy(ρa⁰(Y.c.ρ, Y.c.sgsʲs, turbconv_model))
        (; ᶜT⁰, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰) = p.precomputed
        ᶜρ⁰ = @. lazy(
            TD.air_density(
                thermo_params,
                ᶜT⁰,
                ᶜp,
                ᶜq_tot_nonneg⁰,
                ᶜq_liq⁰,
                ᶜq_ice⁰,
            ),
        )
        @. p.precomputed.ᶜcloud_fraction *= draft_area(ᶜρa⁰, ᶜρ⁰)
    end

    # Add contributions from the updrafts if using EDMF
    if turbconv_model isa PrognosticEDMFX || turbconv_model isa DiagnosticEDMFX
        n = n_mass_flux_subdomains(turbconv_model)
        (; ᶜρʲs, ᶜq_liqʲs, ᶜq_iceʲs) = p.precomputed
        for j in 1:n
            ᶜρaʲ =
                turbconv_model isa PrognosticEDMFX ? Y.c.sgsʲs.:($j).ρa :
                p.precomputed.ᶜρaʲs.:($j)

            @. p.precomputed.ᶜcloud_fraction +=
                ifelse(
                    TD.has_condensate(
                        thermo_params,
                        ᶜq_liqʲs.:($$j) + ᶜq_iceʲs.:($$j),
                    ),
                    draft_area(ᶜρaʲ, ᶜρʲs.:($$j)),
                    0,
                )
        end
    end
end

# ============================================================================
# Machine Learning Cloud Fraction
# ============================================================================

"""
    set_ml_cloud_fraction!(Y, p, ml_cloud, thermo_params, turbconv_model,
                          ᶜρ_env, ᶜT_mean, ᶜq_mean, ᶜθ_mean)

Overwrite the environment cloud fraction with ML-predicted values.

The ML model uses non-dimensional π groups derived from thermodynamic state
and turbulence quantities. Only the cloud fraction is replaced by the ML
prediction; condensate is computed from the grid-mean thermodynamic state.

# Arguments
- `Y`: State vector
- `p`: Cache/parameters
- `ml_cloud`: MLCloud configuration with trained neural network
- `thermo_params`: Thermodynamics parameters
- `turbconv_model`: Turbulence-convection model type
- `ᶜρ_env`: Environment air density [kg/m³]
- `ᶜT_mean`: Mean temperature [K]
- `ᶜq_mean`: Mean total specific humidity [kg/kg]
- `ᶜθ_mean`: Mean liquid-ice potential temperature [K]
"""
function set_ml_cloud_fraction!(
    Y,
    p,
    ml_cloud::MLCloud,
    thermo_params,
    turbconv_model,
    ᶜρ_env,
    ᶜT_mean,
    ᶜq_mean,
    ᶜθ_mean,
)
    # compute_gm_mixing_length materializes into p.scratch.ᶜtemp_scalar
    ᶜmixing_length_lazy =
        turbconv_model isa PrognosticEDMFX || turbconv_model isa DiagnosticEDMFX ?
        ᶜmixing_length(Y, p) :
        compute_gm_mixing_length(Y, p)

    # Materialize mixing length into scratch field to break the lazy broadcast
    # chain. For PrognosticEDMFX, ᶜmixing_length returns a lazy broadcast
    # carrying mixing_length_lopez_gomez_2020 with its parameter structs,
    # which would exceed the 4 KiB GPU kernel parameter limit if nested.
    ᶜmixing_length_field = p.scratch.ᶜtemp_scalar_6
    ᶜmixing_length_field .= ᶜmixing_length_lazy

    # Vertical gradients of q_tot and θ_liq_ice
    ᶜ∇q = p.scratch.ᶜtemp_scalar_2
    ᶜ∇q .=
        projected_vector_data.(
            C3,
            p.precomputed.ᶜgradᵥ_q_tot,
            Fields.level(Fields.local_geometry_field(Y.c)),
        )
    ᶜ∇θ = p.scratch.ᶜtemp_scalar_3
    ᶜ∇θ .=
        projected_vector_data.(
            C3,
            p.precomputed.ᶜgradᵥ_θ_liq_ice,
            Fields.level(Fields.local_geometry_field(Y.c)),
        )

    p.precomputed.ᶜcloud_fraction .=
        compute_ml_cloud_fraction.(
            Ref(ml_cloud.model),
            ᶜmixing_length_field,
            ᶜ∇q,
            ᶜ∇θ,
            ᶜρ_env,
            ᶜT_mean,
            ᶜq_mean,
            ᶜθ_mean,
            thermo_params,
        )
end

"""
    compute_ml_cloud_fraction(nn_model, mixing_length, ∇q, ∇θ, ρ, T, q_tot, θli, thermo_params)

Compute ML-predicted cloud fraction at a single grid point using non-dimensional π groups.

The ML model was trained on four non-dimensional features:
- π₁: Saturation deficit `(q_sat - q_tot) / q_sat`
- π₂: Normalized distance to saturation `Δθ / θ_sat`
- π₃: Moisture gradient term `((dq_sat/dθ × ∇θ - ∇q) × L) / q_sat`
- π₄: Temperature gradient term `(∇θ × L) / θ_sat`

# Arguments
- `nn_model`: Trained neural network model
- `mixing_length`: Turbulent mixing length [m]
- `∇q`: Vertical gradient of total specific humidity [kg/kg/m]
- `∇θ`: Vertical gradient of liquid-ice potential temperature [K/m]
- `ρ`: Air density [kg/m³]
- `T`: Temperature [K]
- `q_tot`: Total specific humidity [kg/kg]
- `θli`: Liquid-ice potential temperature [K]
- `thermo_params`: Thermodynamics parameters

# Returns
- Cloud fraction ∈ [0, 1]
"""
function compute_ml_cloud_fraction(
    nn_model,
    mixing_length,
    ∇q,
    ∇θ,
    ρ,
    T,
    q_tot,
    θli,
    thermo_params,
)
    FT = eltype(thermo_params)

    # Finite difference step size [K] for computing ∂q_sat/∂θ
    Δθ_fd_step = FT(0.1)

    # Compute saturation using functional API
    q_sat = TD.q_vap_saturation(thermo_params, T, ρ)

    # Distance to saturation in θ-space (needed for π groups)
    Δθli, θli_sat, dqsatdθli =
        saturation_distance(q_tot, q_sat, T, ρ, θli, thermo_params, Δθ_fd_step)

    # Form non-dimensional π groups
    π_1 = (q_sat - q_tot) / q_sat
    π_2 = Δθli / θli_sat
    π_3 = ((dqsatdθli * ∇θ - ∇q) * mixing_length) / q_sat
    π_4 = (∇θ * mixing_length) / θli_sat

    return apply_cf_nn(nn_model, π_1, π_2, π_3, π_4)
end

"""
    saturation_distance(q_tot, q_sat, T, ρ, θli, thermo_params, Δθ_fd)

Compute the distance to saturation in θ-space using finite differences.

This function estimates how far the current state is from saturation
by computing a Newton step in θ_liq_ice space. Used for ML feature engineering.

# Arguments
- `q_tot`: Total specific humidity [kg/kg]
- `q_sat`: Saturation specific humidity [kg/kg]
- `T`: Temperature [K]
- `ρ`: Air density [kg/m³]
- `θli`: Liquid-ice potential temperature [K]
- `thermo_params`: Thermodynamics parameters
- `Δθ_fd`: Finite difference step size for computing ∂q_sat/∂θ [K]

# Returns
- `Δθli`: Distance to saturation in θ-space [K]
- `θli_sat`: θ value at saturation [K]
- `dq_sat_dθli`: Sensitivity of saturation humidity to θ [kg/kg/K]
"""
function saturation_distance(q_tot, q_sat, T, ρ, θli, thermo_params, Δθ_fd)
    FT = typeof(T)

    # Estimate perturbed temperature from perturbed θ
    # Using chain rule: ΔT ≈ (∂T/∂θ) × Δθ ≈ (T/θ) × Δθ (Exner factor approximation)
    ∂T_∂θ = T / max(θli, eps(FT))
    T_perturbed = T + ∂T_∂θ * Δθ_fd

    # Compute perturbed saturation using functional API
    q_sat_perturbed = TD.q_vap_saturation(thermo_params, T_perturbed, ρ)

    # Finite-difference derivative ∂q_sat / ∂θli
    dq_sat_dθli = (q_sat_perturbed - q_sat) / Δθ_fd

    # Newton step to saturation distance in θli-space
    # Avoids division by zero when derivative is very small
    Δθli = ifelse(
        abs(dq_sat_dθli) > eps(FT),
        (q_sat - q_tot) / dq_sat_dθli,
        FT(0),
    )
    θli_sat = θli + Δθli

    return Δθli, θli_sat, dq_sat_dθli
end

"""
    apply_cf_nn(model, π_1, π_2, π_3, π_4) -> FT

Apply the neural network model to compute cloud fraction from π groups.

# Arguments
- `model`: Trained neural network (callable with SVector input)
- `π_1`: Saturation deficit `(q_sat - q_tot) / q_sat`
- `π_2`: Normalized distance to saturation `Δθ / θ_sat`
- `π_3`: Moisture gradient term `((dq_sat/dθ × ∇θ - ∇q) × L) / q_sat`
- `π_4`: Temperature gradient term `(∇θ × L) / θ_sat`

# Returns
Cloud fraction clamped to [0, 1].
"""
function apply_cf_nn(model, π_1::FT, π_2::FT, π_3::FT, π_4::FT) where {FT}
    return clamp((model(SA.SVector(π_1, π_2, π_3, π_4))[]), FT(0.0), FT(1.0))
end
