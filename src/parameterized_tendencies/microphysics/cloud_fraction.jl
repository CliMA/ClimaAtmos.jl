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
    # The hybrid cloud fraction fuses the σ_S² quadrature pass inline inside
    # the CF broadcast (see `_compute_cloud_fraction`) — no separate moments
    # materialization during Picard. CF, μ_S, and λ are all written once after
    # Picard converges, in the final `set_sgs_moments_and_cloud_fraction!` call.
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

    # Final post-Aitken update: one quadrature pass refreshes both CF and the
    # microphysics SGS moments (μ_S, λ) using the final covariance.
    set_sgs_moments_and_cloud_fraction!(Y, p)

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
    SGSMomentsEvaluator(tps, ρ)

GPU-safe functor returning `(mu_S = S(ξ), s_sq = S(ξ)²)` at each quadrature
point, where S = q_tot_hat − q_sat_hat is the linear saturation excess.
Used by `_sgs_saturation_moments` to compute μ_S = E[S] and σ_S² = Var[S].
"""
struct SGSMomentsEvaluator{TPS, FT}
    tps::TPS
    ρ::FT
end

@inline function (eval::SGSMomentsEvaluator)(T_hat, q_tot_hat)
    q_sat_hat = TD.q_vap_saturation(eval.tps, T_hat, eval.ρ)
    s = q_tot_hat - q_sat_hat
    return (mu_S = s, s_sq = s * s)
end


"""
    _sgs_saturation_moments(thp, ρ, T_mean, q_tot_mean,
                            sgs_quad, T′T′, q′q′, corr_Tq)

Compute μ_S = E[S] and σ_S = sqrt(Var[S]) of the linear saturation excess
S = q_tot − q_sat via a single Gauss-Hermite quadrature pass over the SGS PDF.

The variance is guarded against negative values from quadrature cancellation
(clipped at zero) and the standard deviation is floored at `ϵ_numerics(FT)`
so the normalised closure (`C = q_c / (α·σ_S)`) is well-conditioned.

Returns `(; mu_S, sigma_S)`.
"""
@inline function _sgs_saturation_moments(
    thp, ρ, T_mean, q_tot_mean,
    sgs_quad, T′T′, q′q′, corr_Tq,
)
    FT = typeof(ρ)
    sgs_quad_eff = isnothing(sgs_quad) ? GridMeanSGS() : sgs_quad
    evaluator = SGSMomentsEvaluator(thp, ρ)
    raw = integrate_over_sgs(
        evaluator, sgs_quad_eff, q_tot_mean, T_mean, q′q′, T′T′, corr_Tq,
    )
    sigma_S_sq = max(raw.s_sq - raw.mu_S * raw.mu_S, zero(FT))
    return (;
        mu_S = raw.mu_S,
        sigma_S = max(sqrt(sigma_S_sq), ϵ_numerics(FT)),
    )
end

# ============================================================================
# Cloud Fraction: truncated-Gaussian closure with Lagrange-multiplier inversion
# ============================================================================
#
# **Physical model**: the subgrid saturation excess S = q_tot − q_sat is
# assumed Gaussian: S ~ N(μ_S, σ_S²), with σ_S computed by the pre-pass
# quadrature.  Working with the centred excess S′ = S − μ_S ~ N(0, σ_S²),
# we seek a Lagrange multiplier λ that enforces mass conservation:
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
# For the *cloud fraction* we use an augmented variance with a fixed
# non-equilibrium floor `σ_S_fix` (hardcoded inside
# `_compute_cloud_fraction`):
#
#     σ_aug = α · sqrt(σ_S² + σ_S_fix²),
#
# which keeps CF well-behaved in the singular limit (q_c, σ_S²) → 0.  CF is
# always computed by solving the truncated-Gaussian closure with `σ_aug`,
# i.e. `C_aug = q_c / σ_aug`, `z_aug = _compute_z(C_aug)`, `CF = Φ(z_aug)`.
# `λ_lagrange` is *not* used to recover CF (the natural shortcut
# `Φ(λ/σ_aug)` would be inconsistent because `λ` is computed with the
# equilibrium `σ_S_eff`, not `σ_aug`).  `λ_lagrange` is computed from the
# raw `σ_S_eff` so the mass-conservation constraint (*) is preserved for
# the microphysics tendencies.
#
# Inverting C = z·Φ(z)+φ(z) for z uses Newton iteration on
# F(z) = z·Φ(z)+φ(z)−C with F′(z) = Φ(z):
#
#     z_{n+1} = (C − φ(z_n)) / Φ(z_n).
#
# **Algorithm** (one Newton step with a fitted initial guess):
# 1. Initial guess:  Φ(z₀) = tanh(1.35·C)  [least-squares fit to exact solution],
#                    z₀    = Φ⁻¹(Φ(z₀))  via `normal_cdf_inv` (A&S 26.2.22).
#    The A&S inverse correctly scales as −√(−2 ln Φ(z₀)) in the tail, avoiding
#    the extreme underestimate of the tanh-based inverse that causes divergence.
# 2. One Newton step:  z₁ = (C − φ(z₀)) / Φ(z₀).
# 3. CF = Φ(z₁) via `normal_cdf` (A&S 26.2.17, max error ≈ 7.5×10⁻⁸).

"""
    sgs_variance_fidelity(cf_steepness_coeff)

Return the variance fidelity parameter `α = 1 / cf_steepness_coeff`, where
`cf_steepness_coeff = CAP.cloud_fraction_steepness_scale(params)`.

`α` controls how much of the SGS saturation variance enters the local
condensate computation:

    E[max(0, λ + α·S′)] = q_c,

where `S′ = S − μ_S ~ N(0, σ_S²)`.  The effective standard deviation used by
the Lagrange-multiplier closure is `σ_S_eff = α · σ_S` (with `σ_S` clipped at
`ϵ_numerics(FT)` inside [`_sgs_saturation_moments`](@ref)), so `C = q_c / σ_S_eff`
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
condensate relation `C = z·Φ(z) + φ(z)` via one Newton step seeded with an
analytic initial guess.

`C = q_c / σ_eff` is the normalised condensate; the caller is responsible
for computing it (typically with `σ_eff = α · σ_S` or `σ_eff = σ_aug` for
the smooth-floored CF formula) so this helper stays free of parameter-
dependent logic.
"""
@inline function _compute_z(C)
    FT = typeof(C)

    # 1. Initial guess: Φ(z₀) = tanh(1.35 · C).
    Φz0 = tanh(FT(1.35) * C)
    # Upper bound must be representably less than 1: 1 - eps(FT) is the
    # largest Float strictly below 1, avoiding normal_cdf_inv(1) → log(0) → NaN.
    Φz0_safe = clamp(Φz0, ϵ_numerics(FT), one(FT) - eps(FT))

    # z₀ = Φ⁻¹(Φz0) via A&S 26.2.22
    z0 = normal_cdf_inv(Φz0_safe)

    # 2. One Newton step: z₁ = (C − φ(z₀)) / Φ(z₀)
    φz0 = exp(-z0 * z0 / 2) / sqrt(FT(2) * FT(π))
    return (C - φz0) / Φz0_safe
end

"""
    _compute_cloud_fraction(q_c, sigma_S, α)

Cloud fraction `CF = Φ(z)` where `z` solves the truncated-Gaussian condensate
relation `q_c/σ_aug = z·Φ(z) + φ(z)` (see [`_compute_z`](@ref)) with the
augmented standard deviation `σ_aug = α · sqrt(σ_S² + σ_S_fix²)`.

Physical motivation: we assume the local condensate `q_c` fluctuates partly
through the equilibrium variations of (T, q_tot) captured by the quadrature
(`σ_S` from `_sgs_saturation_moments`), and partly through additional
non-equilibrium variations not captured by the equilibrium SGS PDF.
`σ_S_fix` models that always-present non-equilibrium contribution. We
include it *only* in the CF computation so that CF stays small when both
`q_c` and the quadrature `σ_S` are small (the singular limit where the
unmodified truncated-Gaussian gives the mathematically correct but
physically unhelpful `CF → 1`). The Lagrange multiplier `λ` and the
`_compute_z` call inside `_compute_sgs_moments` use only the equilibrium
`σ_S` so that mass conservation `E[max(0, λ + α·S′)] = q_c` is exactly
preserved for the microphysics tendencies.
"""
@inline function _compute_cloud_fraction(q_c, sigma_S, α)
    FT = typeof(sigma_S)
    # TODO: promote `σ_S_fix` to a calibrated parameter once values are clear.
    σ_S_fix = FT(1e-6)
    σ_aug = α * sqrt(sigma_S * sigma_S + σ_S_fix * σ_S_fix)
    C = q_c / σ_aug
    z = _compute_z(C)
    return normal_cdf(z)
end

"""
    _compute_cloud_fraction(
        thermo_params, T, ρ, q_tot, q_liq, q_ice,
        sgs_quad, T′T′, q′q′, corr_Tq, α,
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
)
    moments = _sgs_saturation_moments(
        thermo_params, ρ, T, q_tot, sgs_quad, T′T′, q′q′, corr_Tq,
    )
    return _compute_cloud_fraction(q_liq + q_ice, moments.sigma_S, α)
end

"""
    _compute_sgs_moments(thp, ρ, T, q_tot, q_c, sgs_quad, T′T′, q′q′, corr_Tq, α)

Single quadrature pass returning `(mu_S, sigma_S, λ_lagrange)`:

  - `mu_S       = E[S]`: SGS mean saturation variable.
  - `sigma_S    = sqrt(Var[S])`: SGS standard deviation, clipped at
    `ϵ_numerics(FT)` (see [`_sgs_saturation_moments`](@ref)).
  - `λ_lagrange = z·α·σ_S`: Lagrange multiplier satisfying
    `E[max(0, λ + α·S′)] = q_c`.
"""
@inline function _compute_sgs_moments(
    thp, ρ, T, q_tot, q_c,
    sgs_quad, T′T′, q′q′, corr_Tq, α,
)
    moments =
        _sgs_saturation_moments(thp, ρ, T, q_tot, sgs_quad, T′T′, q′q′, corr_Tq)
    σ_S_eff = α * moments.sigma_S
    C = q_c / σ_S_eff
    z = _compute_z(C)
    λ_lagrange = z * σ_S_eff
    return (; moments.mu_S, moments.sigma_S, λ_lagrange)
end

"""
    set_sgs_moments_and_cloud_fraction!(Y, p)

Final post-Aitken update. No-op when `ᶜsgs_moments` is not allocated (dry / 0M).

Uses ONE quadrature pass via `_compute_sgs_moments` to fill
`ᶜsgs_moments = (mu_S, sigma_S, λ_lagrange)`, then computes
`ᶜcloud_fraction` consistently with the augmented `σ_aug` closure (see
[`_compute_cloud_fraction`](@ref)) and applies EDMF updraft weighting.
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
    (; ᶜT′T′, ᶜq′q′) = p.precomputed

    # ONE quadrature pass → (mu_S, sigma_S, λ_lagrange).
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
        p.precomputed.ᶜsgs_moments.sigma_S,
        FT(α),
    )
    _apply_edmf_cloud_weighting!(Y, p, turbconv_model, thermo_params)
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
    (`_compute_cloud_fraction`).
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
    corr_Tq = correlation_Tq(p.params)
    FT = eltype(p.params)
    α = sgs_variance_fidelity(CAP.cloud_fraction_steepness_scale(p.params))

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
