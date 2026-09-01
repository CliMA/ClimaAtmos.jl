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
    (; ᶜbuoygrad, ᶜT, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice) = p.precomputed
    thermo_params = CAP.thermodynamics_params(p.params)
    ᶜlg = Fields.local_geometry_field(Y.c)

    # Materialize the pointwise buoyancy-gradient chain-rule coefficients,
    # exact face gradients, and centered vertical gradients of (θ_li, q_tot) once:
    # the coefficients and materialized θ_li carry all of the expensive saturation
    # thermodynamics and are independent of cloud fraction, so within the Picard
    # iteration every buoyancy-gradient stencil reduces to a cheap `blended_N²` FMA broadcast.
    set_buoyancy_gradient_inputs!(Y, p, thermo_params)
    (; ᶜbg_coeffs) = p.precomputed

    # The buoyancy gradient depends on cloud fraction, and cloud fraction depends
    # on the covariance cache through the mixing length. For reproducible restart,
    # first reconstruct the initial cloud fraction deterministically.
    if p.atmos.numerics.reproducible_restart isa ReproducibleRestart
        set_cloud_fraction!(Y, p, microphysics_model, GridScaleCloud())
    end


    # One Picard step: use the current cloud fraction to update buoyancy
    # gradient and covariance cache, then recompute cloud fraction.
    #
    # The hybrid cloud fraction fuses the σ_S² quadrature pass inline inside
    # the CF broadcast (see `_compute_cloud_fraction`) — no separate moments
    # materialization during Picard. CF, μ_S, and λ are all written once after
    # Picard converges, in the final `set_sgs_moments_and_cloud_fraction!` call.
    function picard_step!()
        @. ᶜbuoygrad = blended_N²(
            ᶜbg_coeffs,
            ᶜcloud_fraction,
            projected_vector_data(C3, ᶜgradᵥ_θ_liq_ice, ᶜlg),
            projected_vector_data(C3, ᶜgradᵥ_q_tot, ᶜlg),
        )

        # Stability-biased buoyancy gradient for the mixing-length and
        # Pr_t(Ri) closures (max of one-sided estimates; registers
        # unresolved inversions that the centered gradient dilutes).
        set_stability_buoyancy_gradient!(Y, p, thermo_params)

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
    @. ᶜbuoygrad = blended_N²(
        ᶜbg_coeffs,
        ᶜcloud_fraction,
        projected_vector_data(C3, ᶜgradᵥ_θ_liq_ice, ᶜlg),
        projected_vector_data(C3, ᶜgradᵥ_q_tot, ᶜlg),
    )
    set_stability_buoyancy_gradient!(Y, p, thermo_params)
    set_covariance_cache!(Y, p, thermo_params)

    # Final post-Aitken update: one quadrature pass refreshes both CF and the
    # microphysics SGS moments (σ_S, λ_lagrange) using the final covariance.
    set_sgs_moments_and_cloud_fraction!(Y, p)

    return nothing
end

"""
    _aitken_picard_helper(c0, c1, c2)

Guarded Aitken Δ² acceleration of the cloud-fraction Picard iterates,
`c_acc = c0 - (c1 - c0)^2 / (c2 - 2c1 + c0)`.

The accelerated value is used only when the two Picard increments change sign,
i.e. when the iterates oscillate about the fixed point and the Aitken value is
expected to lie between them, and when the denominator is above roundoff.
Otherwise the second Picard iterate `c2` is retained.

Called from `set_covariance_cache_and_cloud_fraction!`.
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
    uses_covariances(atmos)

Whether SGS (co)variances of `(T, q_tot)` are needed by the configuration:
either the microphysics quadrature API, 1M/2M non-equilibrium microphysics,
or a `QuadratureCloud`/`MLCloud` model requires them. This is the single
source of truth shared by the two places that must agree on it: the
covariance-cache no-op guard (`set_covariance_cache!`) and the
`ᶜl_mix` caching in `set_explicit_precomputed_quantities!`. When `true`,
`ᶜl_mix` is materialized inside the covariance/cloud-fraction iteration and
the explicit stage skips its redundant recompute; when `false`, only the
explicit stage writes `ᶜl_mix`. The two must never disagree, or `ᶜl_mix`
would go stale (wrong physics) or be computed twice (wasteful).
"""
uses_covariances(atmos) =
    !isnothing(atmos.sgs_quadrature) ||
    atmos.microphysics_model isa
    Union{NonEquilibriumMicrophysics1M, NonEquilibriumMicrophysics2M} ||
    atmos.cloud_model isa Union{QuadratureCloud, MLCloud}

"""
    materialized_mixing_length!(Y, p)

Materialize the SGS master mixing length into a center field and return it.
For an `AbstractEDMF` this writes the persistent `ᶜl_mix` cache (computed once,
reused by TKE dissipation, the covariance closure, and diagnostics); every
`AbstractEDMF` carries `Y.c.ρtke`, so `ᶜmixing_length` is well defined. Otherwise
it falls back to the grid-mean Smagorinsky-Lilly length, materialized in
`p.scratch.ᶜtemp_scalar` by `compute_gm_mixing_length`. Always
materializing (rather than passing the lazy `ᶜmixing_length` broadcast, which
carries `mixing_length_lopez_gomez_2020` with its parameter structs) avoids
recomputing the closure for each covariance and keeps GPU kernel parameters
under the size limit.
"""
function materialized_mixing_length!(Y, p)
    turbconv_model = p.atmos.turbconv_model
    if turbconv_model isa AbstractEDMF
        # Every AbstractEDMF carries Y.c.ρtke, so ᶜmixing_length is well
        # defined; materialize it into the persistent ᶜl_mix cache.
        ᶜl_mix = p.precomputed.ᶜl_mix
        ᶜl_mix .= ᶜmixing_length(Y, p)
        return ᶜl_mix
    else
        # compute_gm_mixing_length materializes into p.scratch.ᶜtemp_scalar
        return compute_gm_mixing_length(Y, p)
    end
end

"""
    set_covariance_cache!(Y, p, thermo_params)

Materializes T-based SGS covariances into cached fields for use by downstream
computations (SGS quadrature, cloud fraction). Populates `p.precomputed.(ᶜT′T′, ᶜq′q′)`.

Pipeline:

 1. Compute mixing length via `materialized_mixing_length!`
 2. Materialize θ-based covariances from gradients
 3. Transform θ→T using `compute_∂T_∂θ!`
"""
function set_covariance_cache!(Y, p, thermo_params)
    # Covariance fields are only allocated when the configuration needs them.
    # No-op otherwise (e.g. EquilMoist + 0M + GridScaleCloud).
    uses_covariances(p.atmos) || return nothing

    (; ᶜT′T′, ᶜq′q′) = p.precomputed

    coeff = CAP.diagnostic_covariance_coeff(p.params)
    (; ᶜgradᵥ_q_tot, ᶜgradᵥ_θ_liq_ice) = p.precomputed

    # Materialize once (see materialized_mixing_length!) to avoid repeating
    # the closure broadcast across the ᶜq′q′ and ᶜT′T′ calculations.
    ᶜmixing_length_field = materialized_mixing_length!(Y, p)

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
# A Gauss-Hermite pass over the SGS PDF computes σ_S via `_sgs_saturation_moments`,
# which drives the truncated-Gaussian cloud-fraction and λ_lagrange closures.
#
# The saturation variable is ALWAYS defined as the linear excess
#
#     S = q_tot − q_sat
#
# regardless of whether the SGS distribution of (T, q_tot) is Gaussian or
# lognormal.  The distribution type (GaussianSGS / LogNormalSGS) controls only
# how the quadrature points (T̂, q̂_tot) are sampled from the joint SGS PDF —
# it does not change the definition of S.  This ensures consistency with the
# Lagrange-multiplier evaluator, which always uses the centred linear excess
# S′ = q_tot_hat − q_sat_hat − μ_S.
#
# (A lognormal closure for S = log(q_tot/q_sat) has no clean closed-form
# inversion for the Lagrange multiplier, so the linear S is used universally.)

"""
    SGSVarianceEvaluator(tps, ρ, mu_S)

GPU-safe functor returning `(S(ξ) − μ_S)²` at each quadrature point, where
S = q_tot_hat − q_sat_hat is the linear saturation excess and `μ_S` is the
linearized SGS mean of S (see `_sgs_saturation_moments`). Used to
accumulate σ_S² = E[(S − μ_S)²] in one quadrature pass; avoids catastrophic
cancellation in `E[S²] − (E[S])²` in Float32 when Var[S] ≪ (E[S])².
"""
struct SGSVarianceEvaluator{TPS, FT}
    tps::TPS
    ρ::FT
    mu_S::FT
end

@inline function (eval::SGSVarianceEvaluator)(T_hat, q_tot_hat)
    q_sat_hat = TD.q_vap_saturation(eval.tps, T_hat, eval.ρ)
    s = q_tot_hat - q_sat_hat
    return (s - eval.mu_S)^2
end

"""
    SGSExcessEvaluator(tps, ρ, mu_S)

GPU-safe functor returning the centred saturation excess
`S′ = q_tot_hat − q_sat(T_hat, ρ) − μ_S` at each quadrature point. This is the
quantity from which the `Microphysics1MEvaluator` reconstructs condensate. It is
used with `quadrature_point_values` to materialize the sampled S′ distribution for
the discrete Lagrange-multiplier fit in `_compute_sgs_moments`.
"""
struct SGSExcessEvaluator{TPS, FT}
    tps::TPS
    ρ::FT
    mu_S::FT
end

@inline (eval::SGSExcessEvaluator)(T_hat, q_tot_hat) =
    q_tot_hat - TD.q_vap_saturation(eval.tps, T_hat, eval.ρ) - eval.mu_S


"""
    _sgs_saturation_moments(thp, ρ, T_mean, q_tot_mean,
                            sgs_quad, T′T′, q′q′, corr_Tq)

Compute μ_S and σ_S of the linear saturation excess S = q_tot − q_sat over
the SGS PDF.

The mean is set analytically to the linearized value

    μ_S = q_tot_mean − q_sat(T_mean, ρ),

which is exact under the same linearization of `q_sat(T)` that makes S
Gaussian to begin with: any difference from `E[S]` evaluated by quadrature
is the q_sat-curvature term that is already discarded in the truncated-
Gaussian closure. Using this analytic μ_S lets us accumulate σ_S² as
`E[(S − μ_S)²]` in a single pass.

`σ_S` is floored at `ϵ_numerics(FT)` so the normalised closure
`C = q_c / (α·σ_S)` stays well-conditioned.

Returns `(; mu_S, sigma_S)`.
"""
@inline function _sgs_saturation_moments(
    thp, ρ, T_mean, q_tot_mean,
    sgs_quad, T′T′, q′q′, corr_Tq,
)
    FT = typeof(ρ)
    mu_S = q_tot_mean - TD.q_vap_saturation(thp, T_mean, ρ)
    sgs_quad_eff = isnothing(sgs_quad) ? GridMeanSGS() : sgs_quad
    evaluator = SGSVarianceEvaluator(thp, ρ, mu_S)
    sigma_S_sq = integrate_over_sgs(
        evaluator, sgs_quad_eff, q_tot_mean, T_mean, q′q′, T′T′, corr_Tq,
    )
    return (;
        mu_S,
        sigma_S = max(sqrt(sigma_S_sq), ϵ_numerics(FT)),
    )
end

# ============================================================================
# Cloud Fraction: truncated-Gaussian closure with Lagrange-multiplier inversion
# ============================================================================
#
# **Physical model**: the subgrid saturation excess S = q_tot − q_sat is
# assumed Gaussian: S ~ N(μ_S, σ_S²), with μ_S set analytically to the
# linearized mean μ_S = q_tot_mean − q_sat(T_mean, ρ) (exact under the
# same linearization that justifies Gaussianity of S) and σ_S accumulated
# in one pass as E[(S − μ_S)²] (see `_sgs_saturation_moments`).
# Working with the centred excess S′ = S − μ_S ~ N(0, σ_S²), we seek a
# Lagrange multiplier λ that enforces mass conservation:
#
#     E[max(0, λ + α·S′)] = q_c,                                     (*)
#
# where α = `sgs_variance_fidelity` controls how much of the SGS variance is
# propagated into the local condensate (currently α = 1). The effective
# scale `σ_S_eff = α · σ_S` uses the standard deviation returned by
# `_sgs_saturation_moments` (already floored at `ϵ_numerics(FT)` to keep
# the normalised problem well-conditioned for tiny variances).
#
# Introducing  z = λ / σ_S_eff  and  C = q_c / σ_S_eff,
# the truncated-Gaussian expectation in (*) evaluates to:
#
#     C = z·Φ(z) + φ(z),
#
# where Φ is the standard normal CDF and φ is its PDF.
#
# The microphysics, however, uses (*) via the N²-point quadrature
# rule (not the continuous Gaussian), and the discrete rule represents the
# kinked integrand max(0, ·) imperfectly (and samples the q ≥ 0 clamp).
# The analytic inverse of C = z·Φ(z) + φ(z) therefore only *seeds*
# `λ_lagrange`; the multiplier is then fitted to the discrete constraint
# Σᵢ wᵢ·max(0, λ + α·S′ᵢ) = q_c  over the same sampled points the
# microphysics evaluator integrates (see `_fit_discrete_lagrange`),
# so mass conservation holds exactly under the quadrature measure.
#
# For the *cloud fraction* we use an augmented variance with a fixed
# non-equilibrium floor `σ_S_floor` (defined inside
# `_compute_cloud_fraction`):
#
#     σ_aug = α · sqrt(σ_S² + σ_S_floor²),
#
# which keeps CF well-behaved in the singular limit (q_c, σ_S²) → 0.  CF is
# always computed by solving the truncated-Gaussian closure with `σ_aug`,
# i.e. `C_aug = q_c / σ_aug`, `z_aug = _compute_z(C_aug)`, `CF = Φ(z_aug)`;
# it is a probability under the continuous Gaussian model. `λ_lagrange` is
# *not* used to recover CF (the natural shortcut `Φ(λ/σ_aug)` would be
# inconsistent because `λ` is fitted with the equilibrium `σ_S_eff`, not `σ_aug`).
#
# Inverting C = z·Φ(z)+φ(z) for z uses Newton iteration on
# F(z) = z·Φ(z)+φ(z)−C with F′(z) = Φ(z):
#
#     z_{n+1} = (C − φ(z_n)) / Φ(z_n).
#
# **Algorithm** (two Newton steps with a fitted initial guess):
# 1. Initial guess:  Φ(z₀) = tanh(1.35·C)  [least-squares fit to exact solution],
#                    z₀    = Φ⁻¹(Φ(z₀))  via `normal_cdf_inv` (A&S 26.2.22).
#    The A&S inverse correctly scales as −√(−2 ln Φ(z₀)) in the tail, avoiding
#    the extreme underestimate of the tanh-based inverse that causes divergence.
# 2. Two Newton steps:  z₁ = (C − φ(z₀)) / Φ(z₀), then likewise z₂ from z₁
#    (skipped where Φ(z₁) ≤ ϵ_numerics; see `_compute_z`).
# 3. CF = Φ(z₂) via `normal_cdf` (A&S 26.2.17, max error ≈ 7.5×10⁻⁸).

"""
    sgs_variance_fidelity(cf_steepness_coeff)

Return the variance fidelity parameter `α = 1 / cf_steepness_coeff`, where
`cf_steepness_coeff = CAP.cloud_fraction_steepness_scale(params)`.

`α` controls how much of the SGS saturation variance enters the local
condensate computation:

    E[max(0, λ + α·S′)] = q_c,

where `S′ = S − μ_S ~ N(0, σ_S²)`.  The effective standard deviation used by
the Lagrange-multiplier closure is `σ_S_eff = α · σ_S` (with `σ_S` clipped at
`ϵ_numerics(FT)` inside `_sgs_saturation_moments`), so `C = q_c / σ_S_eff`
and `λ = z · σ_S_eff`.

With the default `cf_steepness_coeff = 1` this gives `α = 1` (full variance
propagation). Increasing the steepness coefficient sharpens the CF transition
and reduces `α`.

!!! note "Preliminary approximation"

    This is a first-order approximation that ties `α` to the existing
    steepness parameter. Future calibrations may replace or extend this
    relationship to account for the local turbulence state, grid resolution,
    or other physical controls on subgrid variance propagation.
"""
sgs_variance_fidelity(cf_steepness_coeff::FT) where {FT} = one(FT) / cf_steepness_coeff

"""
    _compute_z(C)

Compute the normalised threshold `z` that satisfies the truncated-Gaussian
condensate relation `C = z·Φ(z) + φ(z)` via two Newton steps seeded with an
analytic initial guess.

`C = q_c / σ_eff` is the normalised condensate; the caller is responsible
for computing it (typically with `σ_eff = α · σ_S` or `σ_eff = σ_aug` for
the smooth-floored CF formula) so this helper stays free of parameter-
dependent logic.

Accuracy: the residual `z·Φ(z) + φ(z) − C` after two steps is below 0.7 %
of `C` for `C ≥ 0.05` and below 4 % for `C ≥ 0.01`; a single step leaves
7–36 % in that range. Because the relation is convex, each Newton iterate
overshoots `z` from above, so the (small) remaining error biases `Φ(z)`
high.
"""
@inline function _compute_z(C)
    FT = typeof(C)
    inv_sqrt2π = one(FT) / sqrt(FT(2) * FT(π))

    # 1. Initial guess: Φ(z₀) = tanh(1.35 · C).
    Φz0 = tanh(FT(1.35) * C)
    # Upper bound must be representably less than 1: 1 - eps(FT) is the
    # largest Float strictly below 1, avoiding normal_cdf_inv(1) → log(0) → NaN.
    Φz0_safe = clamp(Φz0, ϵ_numerics(FT), one(FT) - eps(FT))

    # z₀ = Φ⁻¹(Φz0) via A&S 26.2.22
    z0 = normal_cdf_inv(Φz0_safe)

    # 2. First Newton step: z₁ = (C − φ(z₀)) / Φ(z₀)
    φz0 = exp(-z0 * z0 / 2) * inv_sqrt2π
    z1 = (C - φz0) / Φz0_safe

    # 3. Second Newton step. Applied only where Φ(z₁) is above
    # the ϵ_numerics clamp; below that (z₁ ≲ −7 at Float32) the clamped
    # denominator corrupts the ratio φ(z₁)/Φ(z₁), and the result is
    # condensate- and cloud-free to machine precision either way.
    φz1 = exp(-z1 * z1 / 2) * inv_sqrt2π
    Φz1 = normal_cdf(z1)
    z2 = (C - φz1) / max(Φz1, ϵ_numerics(FT))
    return ifelse(Φz1 > ϵ_numerics(FT), z2, z1)
end

"""
    _compute_cloud_fraction(q_c, mu_S, sigma_S, q_sat, α, floor)

Cloud fraction `CF = Φ(z)`, where `z` solves the truncated-Gaussian condensate
relation `q_c/σ_aug = z·Φ(z) + φ(z)` (see `_compute_z`) with the
augmented standard deviation `σ_aug = α · sqrt(σ_S² + σ_S_floor²)`. The `floor`
argument is a `CloudFractionFloorParams` bundling the floor magnitude and
release-shape parameters (see `cloud_fraction_floor_params`).

Scale-aware non-equilibrium floor. We assume the local condensate `q_c`
fluctuates partly through the equilibrium variations of (T, q_tot) captured by
the quadrature (`σ_S`), and partly through non-equilibrium variations not
captured by the equilibrium SGS PDF. `σ_S_floor` models the latter and is scaled
with the saturation specific humidity,

    σ_S_floor² = (D · ε_rel · q_sat)² + σ_abs².

The q_sat-scaling implies saturation-excess fluctuations scale with q_sat.
The parameter `ε_rel` is a condensate-patchiness scale: it indicates the
condensate-to-q_sat ratio at which a subdomain saturates over its full area
(`q_c ≫ ε_rel·q_sat ⇒ CF→1`; `q_c ≪ ε_rel·q_sat ⇒` patchy, `CF→0`).
This is loosely the critical-relative-humidity idea (subgrid humidity
variance ∝ q_sat, `ε_rel ~ 1 − RH_crit`; Quaas, 2012,
doi:10.1029/2012JD017495). However, this is an inexact analogy because
the PDF here is Gaussian (rather than a bounded top-hat as in Quaas 2012),
so there is no sharp onset, and `ε_rel` is the *intra*-subdomain width
only; the inter-subdomain (convective) spread is carried explicitly by the
drafts. The parameter `ε_rel` is a q_sat-scaling whose magnitude should grow
with grid spacing.

Saturation-margin release `D`. Non-equilibrium patchiness is a property of
*partially* saturated air: in a subdomain whose equilibrium PDF sits
inside saturation (an equilibrated overcast deck maintained by radiative
cooling), the patchiness the floor parameterizes is near zero, and an
undamped floor spuriously caps CF well below 1 whenever `q_c ≲ ε_rel·q_sat`.
(This would cut cloud-top longwave cooling in the stratocumulus regime.)
The relative floor is therefore released only where the mean saturation
excess `μ_S = q_tot − q_sat` is positive by a margin relative to the
release width `w`:

    w  = sqrt((c_w·α)²·(σ_S² + σ_abs²) + (c_a·ε_rel·q_sat)²),
    x  = max(μ_S, 0) / w,
    D  = D_min + (1 − D_min) · (1 + x²)^(−s/2),

with four calibratable shape parameters (see
`cloud_fraction_floor_params`):

  - `margin` `c_w`: saturation margin in equilibrium PDF widths at which
    the release transitions. With `abs_margin = 0`, `w` is the equilibrium
    width itself, so the floor is released exactly where the unfloored
    closure already predicts overcast (under saturation adjustment
    `μ_S = q_c`, so `x` is the unfloored normalized condensate `C`).
  - `abs_margin` `c_a`: absolute margin in floor units, added in
    quadrature. Guards against release driven by a spuriously small
    quadrature `σ_S`: for `c_a > 0` (the default), release additionally
    requires `μ_S ≳ c_a·ε_rel·q_sat` regardless of how small the equilibrium
    width is.
  - `sharpness` `s`: transition exponent; larger `s` approaches a switch
    at `x ≈ 1`, smaller `s` gives a gentler algebraic release.
  - `residual` `D_min`: fraction of the relative floor retained deep
    inside a saturated deck, bounding CF below 1 even at full release.

For any parameter values, the floor is fully active (`D = 1`) for a
subsaturated or marginally saturated mean (cumulus, cloud edges — `μ_S ≤ 0`
retains the constant-floor behavior exactly).

The floor enters *only* the CF computation; the Lagrange multiplier `λ` (in
`_compute_sgs_moments`) uses the equilibrium `σ_S`, so mass conservation
`E[max(0, λ + α·S′)] = q_c` is exactly preserved for the microphysics tendencies.
"""
@inline function _compute_cloud_fraction(q_c, mu_S, sigma_S, q_sat, α, floor)
    FT = typeof(q_c)
    (; ε_rel, σ_abs, margin, abs_margin, sharpness, residual) = floor
    # Release the relative floor only where the mean is saturated by a
    # margin relative to the release width `w` — by default the
    # *equilibrium* PDF width, so the floor is released where the unfloored
    # closure already predicts overcast (x is then the unfloored normalized
    # condensate when μ_S ≈ q_c). Subsaturated or marginally saturated means
    # (μ_S ≤ 0: cumulus, cloud edges) keep the full floor for any parameter
    # values. The denominator guard covers only the w → 0 limit: the
    # smallness of the equilibrium width relative to ε_rel·q_sat is exactly
    # what drives the release in a quiescent deck, so it must not be floored
    # away.
    w = sqrt(
        (margin * α)^2 * (sigma_S^2 + σ_abs^2) +
        (abs_margin * ε_rel * q_sat)^2,
    )
    x = max(mu_S, zero(FT)) / max(w, ϵ_numerics(FT))
    # `sharpness == 1` is the default release profile; the fast path keeps
    # it identical to the plain saturation-margin release.
    D_shape =
        sharpness == one(FT) ? 1 / sqrt(1 + x^2) : (1 + x^2)^(-sharpness / 2)
    D = residual + (1 - residual) * D_shape
    σ_S_floor_sq = (D * ε_rel * q_sat)^2 + σ_abs^2
    σ_aug = α * sqrt(sigma_S^2 + σ_S_floor_sq)
    C = q_c / σ_aug
    z = _compute_z(C)
    return normal_cdf(z)
end

"""
    _compute_cloud_fraction(
        thermo_params, T, ρ, q_tot, q_liq, q_ice,
        sgs_quad, T′T′, q′q′, corr_Tq, α, floor,
    )

Fused production overload: compute the hybrid cloud fraction in a single
inlined call that runs the `σ_S²` quadrature pass and the truncated-Gaussian
approximate closure in one broadcast kernel. Used by
`set_cloud_fraction!(QuadratureCloud)` so the moments are never
materialized to a Field.
"""
@inline function _compute_cloud_fraction(
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
    α,
    floor,
)
    moments = _sgs_saturation_moments(
        thermo_params, ρ, T, q_tot, sgs_quad, T′T′, q′q′, corr_Tq,
    )
    q_sat = TD.q_vap_saturation(thermo_params, T, ρ, q_liq, q_ice)
    return _compute_cloud_fraction(
        q_liq + q_ice, moments.mu_S, moments.sigma_S, q_sat, α, floor,
    )
end

"""
    CloudFractionFloorParams{FT}

Augmented-σ floor magnitude and release-shape parameters for
`_compute_cloud_fraction`, bundled into one isbits broadcast scalar
(a struct rather than a NamedTuple, which Base reserves from broadcasting):

  - `ε_rel`, `σ_abs`: relative and absolute floor magnitudes,
  - `margin`, `abs_margin`, `sharpness`, `residual`: release shape (see the
    saturation-margin release section of `_compute_cloud_fraction`).
"""
Base.@kwdef struct CloudFractionFloorParams{FT}
    ε_rel::FT
    σ_abs::FT
    margin::FT
    abs_margin::FT
    sharpness::FT
    residual::FT
end
Base.broadcastable(x::CloudFractionFloorParams) = tuple(x)

"""
    cloud_fraction_floor_params(params)

Build the `CloudFractionFloorParams` bundle from the model
parameter set.
"""
cloud_fraction_floor_params(params) = CloudFractionFloorParams(;
    ε_rel = CAP.cloud_fraction_eps_rel(params),
    σ_abs = CAP.cloud_fraction_sigma_abs(params),
    margin = CAP.cloud_fraction_floor_release_margin(params),
    abs_margin = CAP.cloud_fraction_floor_release_abs_margin(params),
    sharpness = CAP.cloud_fraction_floor_release_sharpness(params),
    residual = CAP.cloud_fraction_floor_residual(params),
)

"""
    _fit_discrete_lagrange(λ0, q_c, α, S′s, ws)

Solve the mass-conservation constraint under the discrete quadrature measure,

    g(λ) = Σᵢ wᵢ · max(0, λ + α·S′ᵢ) = q_c,

for the Lagrange multiplier `λ`, given the sampled centred excesses `S′s` and
probability weights `ws` (summing to 1). `g` is convex, piecewise linear, and
nondecreasing, with kinks at `λ = −α·S′ᵢ`; Newton with the exact one-sided
slope solves the active segment exactly, and, by convexity, subsequent iterates
lie on the root's side, so the iteration reaches the root within
`length(S′s) + 2` fixed trips (GPU-safe: no data-dependent loop bound).

`λ0` is the analytic truncated-Gaussian seed (see `_compute_z`); `q_c ≤ 0`
returns the largest `λ` with `g(λ) = 0`.

Fitting `λ` to the discrete rule rather than the continuous Gaussian makes
`⟨q_c^local⟩ = q_c` hold exactly for the microphysics evaluator, which
uses the same quadrature points.
"""
@inline function _fit_discrete_lagrange(λ0, q_c, α, S′s, ws)
    FT = typeof(q_c)
    λ_max_inactive = -α * maximum(S′s)  # largest λ with g(λ) = 0
    q_c <= zero(FT) && return λ_max_inactive
    λ = λ0
    for _ in 1:(length(S′s) + 2)
        g = zero(FT)
        dg = zero(FT)
        @inbounds for i in eachindex(S′s)
            r = λ + α * S′s[i]
            g += ifelse(r > zero(FT), ws[i] * r, zero(FT))
            dg += ifelse(r > zero(FT), ws[i], zero(FT))
        end
        if dg == zero(FT)
            # Below every kink (g = 0, slope 0): jump into the first active
            # segment; the point with the largest S′ then has r = q_c > 0.
            λ = λ_max_inactive + q_c
        else
            λ_new = λ + (q_c - g) / dg
            λ_new == λ && return λ
            λ = λ_new
        end
    end
    return λ
end

"""
    discrete_cloudy_weight_width_coeff(FT)

Smoothing width of the discrete cloudy weight `sᵢ`, in units of the
equilibrium PDF width `α·σ_S` (see `_discrete_cloud_fraction`).

A hard indicator `1[shifted_excess > 0]` would make each point's
cloudy/clear assignment jump as it crosses the cloud threshold between
steps, injecting noise into any tendency conditioned on `CF_d`. `0.25`
keeps the transition narrow compared with the Gauss-Hermite point spacing
(`≈ 1.7 σ_S` at order 3), so `CF_d` stays close to the hard count, while
still spreading each crossing over a finite range of states.

TODO: promote to a calibratable parameter if the precipitation-overlap closure
turns out to be sensitive to it.
"""
@inline discrete_cloudy_weight_width_coeff(::Type{FT}) where {FT} = FT(0.25)

"""
    discrete_cloudy_weight_width(α, sigma_S)

Smoothing width `ε_w = c_w·α·σ_S` of the discrete cloudy weight, floored at
`ϵ_numerics` so the zero-variance limit stays finite.
"""
@inline discrete_cloudy_weight_width(α, sigma_S) =
    max(
        discrete_cloudy_weight_width_coeff(typeof(sigma_S)) * α * sigma_S,
        ϵ_numerics(typeof(sigma_S)),
    )

"""
    discrete_cloudy_weight(shifted_excess, ε_w)

Smooth cloudy weight `s = sigmoid(shifted_excess / ε_w) ∈ [0, 1]` of one
quadrature point, where `shifted_excess = λ_lagrange + α·S′` is the (signed)
quantity whose positive part is the local condensate.

Evaluated in `tanh` form, which cannot overflow.

`Microphysics1MEvaluator` conditions precipitation on this same weight, and the
exact conservation of that assignment holds only because `CF_d` is accumulated
with it (see `_discrete_cloud_fraction`) — so both must keep calling this one
function.
"""
@inline discrete_cloudy_weight(shifted_excess, ε_w) =
    (1 + tanh(shifted_excess / (2 * ε_w))) / 2

"""
    _discrete_cloud_fraction(q_c, λ_lagrange, α, sigma_S, S′s, ws)

Discrete cloudy mass of the quadrature measure,

    shifted_excessᵢ = λ_lagrange + α·S′ᵢ
    sᵢ              = sigmoid(shifted_excessᵢ / ε_w),
                      ε_w = c_w·α·σ_S  (floored at ϵ_numerics)
    CF_d            = Σᵢ wᵢ·sᵢ

`shifted_excessᵢ` is the quantity whose positive part is the local condensate
(`Microphysics1MEvaluator`), so `sᵢ` is a smoothed indicator of "point `i` is
cloudy" under exactly the measure the microphysics evaluator integrates over.

`CF_d` is the normalizer that makes a per-point assignment of a cell quantity to
cloudy and clear points sum back to the cell mean under the discrete measure.
It therefore cannot be replaced by `ᶜcloud_fraction`, which is continuous,
carries the augmented-σ floor, and (under EDMF) is grid-box rather than
environment weighted.

Condensate-free cells (`q_c ≤ 0`) return `CF_d = 0`. There the fit parks
`λ_lagrange` exactly on the largest kink (`g(λ) = 0`), so the moistest point
sits precisely on the cloud threshold and the smooth weight would report half
its quadrature weight (≈ 1.4% at order 3) as cloudy in every clear cell — the
one place where `CF_d = 0` is load-bearing, since it is what marks a layer as
below cloud base.

The sigmoid is evaluated in `tanh` form, which cannot overflow.
"""
@inline function _discrete_cloud_fraction(q_c, λ_lagrange, α, sigma_S, S′s, ws)
    FT = typeof(λ_lagrange)
    q_c <= zero(FT) && return zero(FT)
    ε_w = discrete_cloudy_weight_width(α, sigma_S)
    CF_d = zero(FT)
    @inbounds for i in eachindex(S′s)
        shifted_excess = λ_lagrange + α * S′s[i]
        CF_d += ws[i] * discrete_cloudy_weight(shifted_excess, ε_w)
    end
    return CF_d
end

"""
    _compute_sgs_moments(thp, ρ, T, q_tot, q_c, sgs_quad, T′T′, q′q′, corr_Tq, α)

Single quadrature pass returning `(sigma_S, λ_lagrange, CF_d)`:

  - `sigma_S = sqrt(Σᵢ wᵢ·S′ᵢ²)`: SGS standard deviation of the sampled
    centred excess, clipped at `ϵ_numerics(FT)`.
  - `λ_lagrange`: Lagrange multiplier satisfying the mass-conservation
    constraint `Σᵢ wᵢ·max(0, λ + α·S′ᵢ) = q_c` exactly under the discrete
    quadrature measure — the same points and weights the microphysics
    evaluator integrates over (see `_fit_discrete_lagrange`; the analytic
    truncated-Gaussian inverse `_compute_z` provides the seed).
  - `CF_d = Σᵢ wᵢ·sᵢ`: the discrete cloudy mass of the same measure, with `sᵢ`
    a smoothed cloudy indicator (see `_discrete_cloud_fraction`). Accumulated
    over points the pass already visits.

The SGS mean `μ_S = q_tot − q_sat(T, ρ)` is analytic under the closure's
linearization (see `_sgs_saturation_moments`) and is recomputed on demand
wherever it is needed downstream.

Without SGS sampling (`nothing`, `GridMeanSGS`), all mass sits at the mean:
`sigma_S = ϵ_numerics(FT)` and the constraint gives `λ_lagrange = q_c`
directly, matching the σ_S → 0 limit of the sampled branch. `CF_d` is then the
all-mass-at-the-mean limit `q_c > 0 ? 1 : 0`.
"""
@inline function _compute_sgs_moments(
    thp, ρ, T, q_tot, q_c,
    sgs_quad, T′T′, q′q′, corr_Tq, α,
)
    FT = typeof(ρ)
    not_quadrature(sgs_quad) && return (;
        sigma_S = ϵ_numerics(FT),
        λ_lagrange = q_c,
        CF_d = ifelse(q_c > zero(FT), one(FT), zero(FT)),
    )

    mu_S = q_tot - TD.q_vap_saturation(thp, T, ρ)
    transform =
        build_physical_transform(sgs_quad, q_tot, T, q′q′, T′T′, corr_Tq)
    S′s = quadrature_point_values(
        SGSExcessEvaluator(thp, ρ, mu_S), transform, sgs_quad,
    )
    ws = quadrature_prob_weights(sgs_quad)
    sigma_S = max(sqrt(sum(ws .* S′s .* S′s)), ϵ_numerics(FT))

    # Analytic truncated-Gaussian seed, then the exact discrete fit.
    σ_S_eff = α * sigma_S
    λ0 = _compute_z(q_c / σ_S_eff) * σ_S_eff
    λ_lagrange = _fit_discrete_lagrange(λ0, q_c, α, S′s, ws)
    CF_d = _discrete_cloud_fraction(q_c, λ_lagrange, α, sigma_S, S′s, ws)
    return (; sigma_S, λ_lagrange, CF_d)
end

"""
    set_sgs_moments_and_cloud_fraction!(Y, p)

Final post-Aitken update. No-op when `ᶜsgs_moments` is not allocated (dry / 0M).

Uses ONE quadrature pass via `_compute_sgs_moments` to fill
`ᶜsgs_moments = (sigma_S, λ_lagrange, CF_d)`, then computes
`ᶜcloud_fraction` consistently with the augmented `σ_aug` closure (see
`_compute_cloud_fraction`) and applies EDMF updraft weighting.

Finally runs `set_precip_fraction!`, the column overlap sweep that turns the
freshly computed `CF_d` into the precipitation fraction `ᶜprecip_frac`.
"""
NVTX.@annotate function set_sgs_moments_and_cloud_fraction!(Y, p)
    hasproperty(p.precomputed, :ᶜsgs_moments) || return nothing

    thermo_params = CAP.thermodynamics_params(p.params)
    turbconv_model = p.atmos.turbconv_model
    microphysics_model = p.atmos.microphysics_model

    ᶜρ_env, ᶜT_mean, ᶜq_mean = _get_env_ρ_T_q(Y, p, thermo_params, turbconv_model)
    ᶜq_lcl, ᶜq_icl = _get_condensate_means(Y, p, turbconv_model, microphysics_model)
    sgs_quad = p.atmos.sgs_quadrature
    corr_Tq = correlation_Tq(p.params)
    FT = eltype(p.params)
    α = sgs_variance_fidelity(CAP.cloud_fraction_steepness_scale(p.params))
    floor = cloud_fraction_floor_params(p.params)
    (; ᶜT′T′, ᶜq′q′) = p.precomputed

    # ONE quadrature pass → (sigma_S, λ_lagrange, CF_d).
    @. p.precomputed.ᶜsgs_moments = _compute_sgs_moments(
        thermo_params, ᶜρ_env, ᶜT_mean, ᶜq_mean, ᶜq_lcl + ᶜq_icl,
        $(sgs_quad), ᶜT′T′, ᶜq′q′, corr_Tq, FT(α),
    )
    # Recompute CF from q_c and σ_S using the augmented-σ closure. We cannot
    # use `Φ(λ/σ_aug)` because λ was computed with the equilibrium σ_S_eff,
    # not σ_aug — `Φ(λ/σ_aug)` would not match the truncated-Gaussian
    # closure for the augmented variance. This overwrites the Picard iterate
    # with a value consistent with the final SGS moments, so EDMF weighting
    # must be re-applied here even though `set_cloud_fraction!` already
    # applied it during Picard.
    @. p.precomputed.ᶜcloud_fraction = _compute_cloud_fraction(
        ᶜq_lcl + ᶜq_icl,
        # μ_S recomputed analytically, matching `_sgs_saturation_moments`
        # (condensate-free q_sat, consistent with the linear excess S).
        ᶜq_mean - TD.q_vap_saturation(thermo_params, ᶜT_mean, ᶜρ_env),
        p.precomputed.ᶜsgs_moments.sigma_S,
        TD.q_vap_saturation(thermo_params, ᶜT_mean, ᶜρ_env, ᶜq_lcl, ᶜq_icl),
        FT(α),
        $(floor),
    )
    _apply_edmf_cloud_weighting!(Y, p, turbconv_model, thermo_params)
    # Set precipitation fraction, which depends on CF_d.
    set_precip_fraction!(Y, p)
end

"""
    set_precip_fraction!(Y, p)

Fill `p.precomputed.ᶜprecip_frac` with the precipitation fraction `a_p`, the
area fraction of the cell covered by the precipitation shaft, by maximum-random
overlap swept from the model top down:

    a_p(k) = max(CF_d(k), f_decay · a_p(k+1))   where q_rai + q_sno > q_min
    a_p(k) = 0                                   otherwise

with `f_decay = precip_overlap_decay` (1 is pure maximum overlap; the shrink is
applied per level, not per metre). A precipitation-free level resets the
recursion, so a shaft that evaporates completely does not seed the layers below
it.

The recursion needs a binary "is there a shaft here" test, and the mask has to
close at the bottom of the shaft — otherwise `a_p` would be inherited all the
way to the surface through air that carries no precipitation. The presence
threshold is Thermodynamics' `q_min`, the same one `TD.has_condensate` uses to
decide that a water species is present (and that this file already uses for the
binary updraft cloud check), so where the mask closes is set by real
precipitation rather than by numerical residue.

The recursion is driven by the discrete cloudy mass `CF_d`, **not** by
`ᶜcloud_fraction`, so that `a_p` and the conditioning weights that consume it
share one measure: the discrete quadrature measure (see
`_discrete_cloud_fraction`). Under `PrognosticEDMFX` this makes `a_p`
environment-relative, matching the environment quantities the
quadrature integrates, so the precipitation mask uses `q_rai⁰ + q_sno⁰` too.

Mutates `p.precomputed.ᶜprecip_frac`; the return value is unused.
"""
NVTX.@annotate function set_precip_fraction!(Y, p)
    FT = eltype(p.params)
    thermo_params = CAP.thermodynamics_params(p.params)
    _precip_fraction_sweep!(
        p.precomputed.ᶜprecip_frac,
        p.precomputed.ᶜsgs_moments.CF_d,
        _get_precip_mean(Y, p, p.atmos.turbconv_model),
        FT(CAP.precip_overlap_decay(p.params)),
        FT(TD.Parameters.q_min(thermo_params)),
    )
    return nothing
end

"""
    _precip_fraction_sweep!(ᶜprecip_frac, ᶜCF_d, ᶜq_precip, f_decay, q_precip_min)

Run the maximum-random overlap recursion of `set_precip_fraction!` on plain
fields, writing `a_p` into `ᶜprecip_frac`. Levels with
`ᶜq_precip ≤ q_precip_min` carry no shaft.

Implemented as a loop of level broadcasts rather than with
`Operators.column_accumulate!`, which is the natural spelling but is not
affordable here. Its CPU path walks columns one at a time and materializes a
level `Field` per level per column, costing ~240 B of garbage per
column-level: ~8.6 MB per call on a 1536-column sphere and ~83 MB on the
`bm_default_1m` benchmark (13824 columns × 25 levels), against a 663 kB
allocation budget for the entire timestep — and this runs every explicit
stage. The loop below allocates ~32 B per level *independent of the number of
columns* (~1 kB per call at 25 levels) and gives bitwise-identical results.
Its cost on GPU is one kernel launch per level instead of one per sweep.

The precipitation mask is folded into `ᶜprecip_frac` first, as a negative
sentinel, so the recursion reads and writes a single field instead of carrying
a tuple down the column. The sentinel is needed because a level holding no
precipitation must both report `a_p = 0` *and* stop the shaft above it from
being inherited further down, so "no precipitation" has to stay
distinguishable from "precipitating, but with no cloud of its own"
(`CF_d = 0`, which does inherit).
"""
function _precip_fraction_sweep!(
    ᶜprecip_frac,
    ᶜCF_d,
    ᶜq_precip,
    f_decay,
    q_precip_min,
)
    FT = eltype(ᶜprecip_frac)

    # The recursion as ClimaCore would spell it. Kept for reference, and to
    # switch back to if `column_accumulate!` ever stops materializing a level
    # `Field` per column-level on CPU (see the docstring); the loop below is
    # bitwise-identical to it.
    #
    #     input = @. lazy(tuple(ᶜCF_d, ᶜq_precip))
    #     Operators.column_accumulate!(
    #         ᶜprecip_frac,
    #         input;
    #         init = zero(FT),
    #         reverse = true,
    #     ) do a_p_above, (CF_d_level, q_precip_level)
    #         ifelse(
    #             q_precip_level > q_precip_min,
    #             max(CF_d_level, f_decay * a_p_above),
    #             zero(FT),
    #         )
    #     end

    @. ᶜprecip_frac = ifelse(ᶜq_precip > q_precip_min, ᶜCF_d, -one(FT))

    # Level 1 is the bottom model level, so the recursion runs from the last
    # level down. The top level has no shaft above it, so its sentinel just
    # resolves to zero.
    nz = Spaces.nlevels(axes(ᶜprecip_frac))
    ᶜa_p_above = Fields.level(ᶜprecip_frac, nz)
    @. ᶜa_p_above = max(ᶜa_p_above, zero(FT))
    for level in (nz - 1):-1:1
        ᶜa_p = Fields.level(ᶜprecip_frac, level)
        # `ᶜa_p_above` has already been resolved, so it is non-negative here.
        @. ᶜa_p = ifelse(
            ᶜa_p < zero(FT),
            zero(FT),
            max(ᶜa_p, f_decay * ᶜa_p_above),
        )
        ᶜa_p_above = ᶜa_p
    end
    return nothing
end

"""
    _get_precip_mean(Y, p, turbconv_model)

Mean precipitation specific humidity `q_rai + q_sno` of the domain that carries
the SGS closure: the environment for `PrognosticEDMFX`, the grid mean
otherwise. Matches the domain of `_get_condensate_means`, so the precipitation
mask and `CF_d` in `set_precip_fraction!` describe the same air.
"""
function _get_precip_mean(Y, p, turbconv_model)
    if turbconv_model isa PrognosticEDMFX
        ᶜq_rai⁰ = ᶜspecific_env_value(@name(q_rai), Y, p)
        ᶜq_sno⁰ = ᶜspecific_env_value(@name(q_sno), Y, p)
        return @. lazy(max(0, ᶜq_rai⁰) + max(0, ᶜq_sno⁰))
    else
        return @. lazy(
            max(0, specific(Y.c.ρq_rai, Y.c.ρ)) +
            max(0, specific(Y.c.ρq_sno, Y.c.ρ)),
        )
    end
end


# ============================================================================
# Cloud Fraction Dispatch Methods
# ============================================================================

"""
    set_cloud_fraction!(Y, p, microphysics_model, cloud_model)

Compute the grid-scale cloud fraction from subgrid-scale properties and store it in
`p.precomputed.ᶜcloud_fraction`.

Dispatches on `microphysics_model` and `cloud_model`:

  - `DryModel`: cloud fraction is zero everywhere.
  - `GridScaleCloud`: 1 where grid-scale condensate is present, 0 otherwise.
  - `QuadratureCloud`: the truncated-Gaussian closure with the fused `σ_S²`
    quadrature pass (see `_compute_cloud_fraction`).
  - `MLCloud`: the neural-network prediction (see
    `set_ml_cloud_fraction!`).

With `PrognosticEDMFX`, the environment value is weighted by the environment area
fraction and binary updraft contributions are added
(`_apply_edmf_cloud_weighting!`).

Mutates `p.precomputed.ᶜcloud_fraction`; the return value is unused.
"""
NVTX.@annotate function set_cloud_fraction!(Y, p, ::DryModel, _)
    FT = eltype(p.params)
    p.precomputed.ᶜcloud_fraction .= FT(0)
end
NVTX.@annotate function set_cloud_fraction!(
    Y,
    p,
    microphysics_model::MoistMicrophysics,
    ::GridScaleCloud,
)
    ᶜq_lcl, ᶜq_icl = _grid_mean_cloud_condensate(Y, p, microphysics_model)
    thermo_params = CAP.thermodynamics_params(p.params)
    FT = eltype(p.params)
    @. p.precomputed.ᶜcloud_fraction =
        ifelse(
            TD.has_condensate(thermo_params, ᶜq_lcl + ᶜq_icl),
            FT(1),
            FT(0),
        )
end

"""
    _grid_mean_cloud_condensate(Y, p, microphysics_model)

Grid-mean cloud condensate `(ᶜq_lcl, ᶜq_icl)`, used by `GridScaleCloud` and,
without EDMF, by `_get_condensate_means`. With non-equilibrium
microphysics, uses the prognostic cloud condensate only; the precomputed
`ᶜq_liq` / `ᶜq_ice` include precipitation (`q_rai` / `q_sno`), which should
not count as cloud.
"""
_grid_mean_cloud_condensate(Y, p, ::NonEquilibriumMicrophysics) = (
    (@. lazy(max(0, specific(Y.c.ρq_lcl, Y.c.ρ)))),
    (@. lazy(max(0, specific(Y.c.ρq_icl, Y.c.ρ)))),
)
_grid_mean_cloud_condensate(Y, p, microphysics_model) =
    (p.precomputed.ᶜq_liq, p.precomputed.ᶜq_ice)
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
    corr_Tq = correlation_Tq(p.params)
    FT = eltype(p.params)
    α = sgs_variance_fidelity(CAP.cloud_fraction_steepness_scale(p.params))
    floor = cloud_fraction_floor_params(p.params)

    (; ᶜT′T′, ᶜq′q′) = p.precomputed

    # Hybrid cloud fraction: the σ_S² quadrature pass is fused into this
    # broadcast kernel, so the moments stay in registers and are never written
    # to a Field.
    @. p.precomputed.ᶜcloud_fraction = _compute_cloud_fraction(
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
        FT(α),
        $(floor),
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

Mean cloud condensate `(ᶜq_lcl, ᶜq_icl)` of the domain that carries the SGS
cloud closure: the environment for PrognosticEDMFX
(`_env_cloud_condensate`), the grid mean otherwise
(`_grid_mean_cloud_condensate`). Updraft contributions are added
separately by `_apply_edmf_cloud_weighting!`.
"""
_get_condensate_means(Y, p, turbconv_model, microphysics_model) =
    turbconv_model isa PrognosticEDMFX ?
    _env_cloud_condensate(Y, p, microphysics_model) :
    _grid_mean_cloud_condensate(Y, p, microphysics_model)

"""
    _env_cloud_condensate(Y, p, microphysics_model)

Environment cloud condensate `(ᶜq_lcl⁰, ᶜq_icl⁰)` for PrognosticEDMFX. With
non-equilibrium microphysics, uses the environment cloud condensate only; the
precomputed `ᶜq_liq⁰` / `ᶜq_ice⁰` include precipitation (`q_rai⁰` / `q_sno⁰`),
which should not count as cloud.
"""
_env_cloud_condensate(Y, p, ::NonEquilibriumMicrophysics) = (
    (@. lazy(max(0, $(ᶜspecific_env_value(@name(q_lcl), Y, p))))),
    (@. lazy(max(0, $(ᶜspecific_env_value(@name(q_icl), Y, p))))),
)
_env_cloud_condensate(Y, p, microphysics_model) =
    (p.precomputed.ᶜq_liq⁰, p.precomputed.ᶜq_ice⁰)

"""
    _apply_edmf_cloud_weighting!(Y, p, turbconv_model, thermo_params)

Apply EDMF-specific adjustments to cloud diagnostics.

For PrognosticEDMFX:

 1. Weights environment cloud diagnostics by environment area fraction
 2. Adds updraft contributions weighted by their respective area fractions

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
    if turbconv_model isa PrognosticEDMFX
        n = n_mass_flux_subdomains(turbconv_model)
        (; ᶜρʲs) = p.precomputed
        microphysics_model = p.atmos.microphysics_model
        for j in 1:n
            ᶜρaʲ = Y.c.sgsʲs.:($j).ρa
            ᶜq_liqʲ, ᶜq_iceʲ =
                _updraft_cloud_condensate(Y, p, j, microphysics_model)

            @. p.precomputed.ᶜcloud_fraction +=
                ifelse(
                    TD.has_condensate(
                        thermo_params,
                        max(0, ᶜq_liqʲ + ᶜq_iceʲ),
                    ),
                    draft_area(ᶜρaʲ, ᶜρʲs.:($$j)),
                    0,
                )
        end
    end
end

"""
    _updraft_cloud_condensate(Y, p, j, microphysics_model)

Cloud condensate of updraft `j`, used for the binary updraft cloud check.
With non-equilibrium microphysics, uses the prognostic cloud condensate
(`q_lclʲ`, `q_iclʲ`) only; the precomputed `ᶜq_liqʲs` / `ᶜq_iceʲs` include
precipitation (`q_raiʲ` / `q_snoʲ`), which should not count as cloud.
"""
_updraft_cloud_condensate(Y, p, j, ::NonEquilibriumMicrophysics) =
    (Y.c.sgsʲs.:($j).q_lcl, Y.c.sgsʲs.:($j).q_icl)
_updraft_cloud_condensate(Y, p, j, microphysics_model) =
    (p.precomputed.ᶜq_liqʲs.:($j), p.precomputed.ᶜq_iceʲs.:($j))

# ============================================================================
# Machine Learning Cloud Fraction
# ============================================================================

"""
    set_ml_cloud_fraction!(Y, p, ml_cloud, thermo_params, turbconv_model,
                          ᶜρ_env, ᶜT_mean, ᶜq_mean, ᶜθ_mean)

Overwrite `p.precomputed.ᶜcloud_fraction` with the ML prediction.

The network is evaluated point-wise on non-dimensional π groups built from the
thermodynamic state, the mixing length, and the vertical gradients of `q_tot` and
`θ_liq_ice` (see `compute_ml_cloud_fraction`). Only cloud fraction is
predicted; the condensate is left as diagnosed elsewhere. EDMF area weighting is
applied by the caller.

# Arguments

  - `Y`: State vector.
  - `p`: Cache; reads `p.precomputed.ᶜgradᵥ_q_tot` and `ᶜgradᵥ_θ_liq_ice`, and uses
    `p.scratch.ᶜtemp_scalar_2` and `ᶜtemp_scalar_3` for the projected gradients.
  - `ml_cloud`: `MLCloud` configuration holding the trained network.
  - `thermo_params`: Thermodynamics parameters.
  - `turbconv_model`: Turbulence-convection model.
  - `ᶜρ_env`: Environment air density [kg/m³].
  - `ᶜT_mean`: Mean temperature [K].
  - `ᶜq_mean`: Mean total specific humidity [kg/kg].
  - `ᶜθ_mean`: Mean liquid-ice potential temperature [K].

Mutates `p.precomputed.ᶜcloud_fraction`; the return value is unused.
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
    ᶜmixing_length_field = materialized_mixing_length!(Y, p)

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
