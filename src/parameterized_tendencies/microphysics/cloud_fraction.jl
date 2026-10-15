import NVTX
import StaticArrays as SA

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
    (; ᶜbuoygrad) = p.precomputed
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
    if p.atmos.numerics.reproducible_restart
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
        # Cache SGS covariances (no-op for dry/0M/GridScaleCloud configs). The vertical
        # gradients and N² coefficients were materialized above by
        # set_buoyancy_gradient_inputs! for every turbconv model.
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
    # (`set_covariance_cache!` writes _2, _3, _5, _6; the QuadratureCloud
    # `set_cloud_fraction!` writes `ᶜtemp_scalar`, `_2`, `_3`).
    # Conversely, those callees and `set_sgs_moments_and_cloud_fraction!` must never write
    # ᶜtemp_scalar_4 or ᶜtemp_scalar_7. c2 shares ᶜtemp_scalar, which is safe only
    # because it is written after the last picard_step!() and consumed by the Aitken
    # update before `set_sgs_moments_and_cloud_fraction!` overwrites it.
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
    materialized_variance_fidelity!(Y, p)

Refresh `p.precomputed.ᶜα` from the current `ᶜl_mix`, TKE, and condensate
state (see `sgs_variance_fidelity`). Called at the end of
`set_covariance_cache!` so `ᶜα` tracks each Picard iterate.

`σ_S` is estimated with the Gaussian-linear analytic formula
`σ_S² = σ_q² − 2γ Cov(T,q) + γ² σ_T²`, with `γ = ∂q_vap_sat_∂T`. This is
the fast estimate; the downstream quadrature closures still compute their
own `σ_S`.

Non-EDMF configurations without `ᶜl_mix` or TKE substitute `TKE = 0`,
`l_mix = 1 m`, collapsing the Damköhler factor to 1.
"""
function materialized_variance_fidelity!(Y, p)
    hasproperty(p.precomputed, :ᶜα) || return nothing
    (; ᶜT, ᶜq_tot_nonneg, ᶜT′T′, ᶜq′q′, ᶜcorr_Tq, ᶜα) = p.precomputed
    FT = eltype(p.params)
    A = FT(CAP.cloud_fraction_steepness_scale(p.params))

    if p.atmos.microphysics_model isa EquilibriumMicrophysics0M
        @. ᶜα = one(FT) / A
        return nothing
    end

    τ_cond = FT(CAP.condensation_evaporation_timescale(p.params))
    τ_sub = FT(CAP.sublimation_deposition_timescale(p.params))
    w_eq = FT(CAP.sgs_fidelity_equilibrium_weight(p.params))
    thp = CAP.thermodynamics_params(p.params)
    q_min = FT(TD.Parameters.q_min(thp))
    turbconv_params = CAP.turbconv_params(p.params)
    c_d = tke_dissipation_coefficient(turbconv_params)
    # Non-EDMF fallbacks: TKE → 0 drives α_Da → 1.
    ᶜtke_or_zero =
        hasproperty(Y.c, :ρtke) ?
        (@. lazy(specific(Y.c.ρtke, Y.c.ρ))) : zero(FT)
    ᶜl_mix_or_one =
        hasproperty(p.precomputed, :ᶜl_mix) ? p.precomputed.ᶜl_mix : one(FT)
    ᶜq_lcl, ᶜq_icl = _grid_mean_cloud_condensate(Y, p, p.atmos.microphysics_model)
    # Linear σ_S² = σ_q² − 2γ Cov(T,q) + γ² σ_T², with Clausius-Clapeyron γ.
    ᶜγ = @. lazy(TD.∂q_vap_sat_∂T(thp, ᶜT, Y.c.ρ))
    ᶜsigma_S = @. lazy(
        sqrt(
            max(
                ᶜq′q′ - 2 * ᶜγ * ᶜcorr_Tq * sqrt(max(ᶜT′T′ * ᶜq′q′, zero(FT))) +
                ᶜγ^2 * ᶜT′T′,
                zero(FT),
            ),
        ),
    )
    ᶜmu_S = @. lazy(ᶜq_tot_nonneg - TD.q_vap_saturation(thp, ᶜT, Y.c.ρ))
    @. ᶜα = sgs_variance_fidelity(
        A,
        sgs_microphysics_timescale(thp, ᶜT, τ_cond, τ_sub),
        c_d,
        ᶜtke_or_zero,
        ᶜl_mix_or_one,
        w_eq,
        ᶜq_lcl + ᶜq_icl,
        ᶜmu_S,
        ᶜsigma_S,
        q_min,
    )
    return nothing
end

"""
    along_surface_q_variance(inv_q, inv_θq, inv_θ, ∂q∂z, ∂θ∂z, ℓ_h, ℓ_v)

Variance of `q_tot` from subgrid parcel displacements along the local `θ_li`
surface, without the prefactor `c_g`.

A parcel moves by a Gaussian displacement `δ` with covariance
`D = diag(ℓ_h², ℓ_h², ℓ_v²)` and stays on the surface, `δ · ∇θ_li = 0`. With
`g = ∇q_tot` and `n = ∇θ_li` (3D gradients), the variance of `δ · g` under
this condition is

    gᵀ D g − (gᵀ D n)² / (nᵀ D n).

The inputs are the horizontal invariants `inv_q = |∇_h q|²`,
`inv_θq = ∇_h q · ∇_h θ`, `inv_θ = |∇_h θ|²` and the vertical gradients. The
result lies in `[0, gᵀ D g]` (Cauchy–Schwarz). For a flat surface it is
`ℓ_h² |∇_h q|²`. For a vertical surface it is the horizontal gradient along
the surface plus `ℓ_v² (∂q/∂z)²`. Where `∇θ_li = 0` there is no surface, and
the result is the unconstrained `gᵀ D g`.
"""
@inline function along_surface_q_variance(
    inv_q,
    inv_θq,
    inv_θ,
    ∂q∂z,
    ∂θ∂z,
    ℓ_h,
    ℓ_v,
)
    G = ℓ_h^2 * inv_q + ℓ_v^2 * ∂q∂z^2
    M = ℓ_h^2 * inv_θq + ℓ_v^2 * ∂q∂z * ∂θ∂z
    N = ℓ_h^2 * inv_θ + ℓ_v^2 * ∂θ∂z^2
    return ifelse(N > zero(N), max(G - M^2 / N, zero(G)), G)
end

"""
    across_surface_q_gradient(inv_θq, inv_θ, ∂q∂z, ∂θ∂z)

Change of `q_tot` per unit vertical displacement across the local `θ_li`
surface, `∂_z θ (∇q · ∇θ) / |∇θ|²` with 3D gradients. A vertical step has the
component `∂_z θ / |∇θ|` along the surface normal, and `q_tot` changes along
the normal at the rate `∇q · ∇θ / |∇θ|`. The result equals `∂q/∂z` for a flat
surface and is zero for a vertical surface or where `∇θ_li = 0`. The vertical
uniform-box block uses it in place of `∂q/∂z`, so that block samples the
displacement across the surface and `along_surface_q_variance` the
displacement along it.
"""
@inline function across_surface_q_gradient(inv_θq, inv_θ, ∂q∂z, ∂θ∂z)
    gn = inv_θq + ∂q∂z * ∂θ∂z
    nn = inv_θ + ∂θ∂z^2
    return ifelse(nn > zero(nn), ∂θ∂z * gn / nn, zero(gn))
end

"""
    geometric_vertical_q_variance(inv_q, inv_θq, inv_θ, ∂q∂z, ∂θ∂z, ℓ_h, ℓ_a, ℓ_v)

The part of the geometric (resolved-gradient) q variance that needs a
vertical displacement, without the prefactor `c_g`: the along-surface
variance beyond its along-contour part (the displacement with `ℓ_a = 0`,
which stays on the surface without moving vertically) plus the across-surface
uniform-box term,

    V_v = max(V_along(ℓ_a) − V_along(0), 0) + ℓ_v² q_n².

On a flat surface the along part vanishes and `V_v = ℓ_v² (∂_z q)²`; on a
vertical surface, or where `∇θ_li = 0`, `q_n = 0` and `V_v = ℓ_a² (∂_z q)²`;
on a tilted surface the two parts are orthogonal and add.
"""
@inline function geometric_vertical_q_variance(
    inv_q, inv_θq, inv_θ, ∂q∂z, ∂θ∂z, ℓ_h, ℓ_a, ℓ_v,
)
    V_all = along_surface_q_variance(inv_q, inv_θq, inv_θ, ∂q∂z, ∂θ∂z, ℓ_h, ℓ_a)
    V_h = along_surface_q_variance(
        inv_q, inv_θq, inv_θ, ∂q∂z, ∂θ∂z, ℓ_h, zero(ℓ_a),
    )
    q_n = across_surface_q_gradient(inv_θq, inv_θ, ∂q∂z, ∂θ∂z)
    return max(V_all - V_h, zero(V_all)) + ℓ_v^2 * q_n^2
end

"""
    hgrad_invariant!(ᶜinv, ᶜψ, p)

Write the element-scale horizontal-gradient invariant `|∇_h ψ|²` of the center field `ᶜψ`
into `ᶜinv`: the invariant of the nodal spectral-element gradient, replaced within each
element by its lumped GLL{2} restriction (`lumpedₕ`, ClimaCore's `LumpedRestriction`), in
one fused broadcast. The nodal invariant is systematically too large at the element-boundary
nodes and too small inside once `ψ` has structure near the element scale; the restriction removes
that node-position bias while conserving each element's WJ-weighted integral of the invariant.
One weighted DSS then makes the result continuous across elements. On a space that
needs no DSS (single column) the restriction is a no-op and the nodal invariant is used.
"""
function hgrad_invariant!(ᶜinv, ᶜψ, p)
    @. ᶜinv = lumpedₕ(norm_sqr(gradₕ(ᶜψ)))
    buf = p.scratch.ᶜscalar_dss_buffer
    buf === nothing || Spaces.weighted_dss!(ᶜinv, buf)
    return nothing
end

"""
    hgrad_cross_invariant!(ᶜinv, ᶜψ1, ᶜψ2, p)

Write the horizontal-gradient cross invariant `∇_h ψ1 · ∇_h ψ2` into `ᶜinv`, with the
same gradient estimate and element filter as `hgrad_invariant!`: the nodal
cross invariant, replaced within each element by its lumped GLL{2} restriction, then one
weighted DSS (skipped on a space that needs no DSS).
"""
function hgrad_cross_invariant!(ᶜinv, ᶜψ1, ᶜψ2, p)
    @. ᶜinv = lumpedₕ(dot(Geometry.Contravariant12Vector(gradₕ(ᶜψ1)), gradₕ(ᶜψ2)))
    buf = p.scratch.ᶜscalar_dss_buffer
    buf === nothing || Spaces.weighted_dss!(ᶜinv, buf)
    return nothing
end

"""
    sgs_correlation_Tq(T′T′, q′q′, T′q′, fallback)

Diagnosed SGS T–q correlation `clamp(T′q′ / √(T′T′ q′q′), -1, 1)`, or
`fallback` where the product of the variances is at the round-off floor.

In exact arithmetic each variance block satisfies its own Cauchy-Schwarz
inequality and the Cauchy-Schwarz bound on the sum of the two (turbulent,
across-surface) covariances also holds — the along-surface block
contributes variance to q′q′ only and no covariance — so
`|ρ| ≤ 1` by construction. The explicit clamp is a cheap safeguard against
(a) Float32 roundoff at the Cauchy-Schwarz boundary (which can produce
`|ρ| = 1 + O(ε)` and feed NaN into downstream `√(1 − ρ²)` in the bivariate
quadrature) and (b) a user-supplied `Tq_correlation_coefficient` outside
`[-1, 1]` (which violates the turbulent-block CS before we even sum).
"""
@inline function sgs_correlation_Tq(T′T′, q′q′, T′q′, fallback)
    FT = typeof(T′T′)
    v = max(T′T′, zero(FT)) * max(q′q′, zero(FT))
    return ifelse(
        v > eps(FT)^2,
        clamp(T′q′ / sqrt(max(v, eps(FT)^2)), -one(FT), one(FT)),
        fallback,
    )
end

"""
    set_tq_correlation!(p, correlation_model)

Fill `p.precomputed.ᶜcorr_Tq` with the SGS T–q correlation the quadrature samples.
`ConstantTqCorrelation`: the prescribed `Tq_correlation_coefficient`.
`DiagnosedTqCorrelation`: `T′q′ / √(T′T′ q′q′)` from the gradient covariance
accumulated in `p.precomputed.ᶜT′q′` by `set_covariance_cache!` (see
`sgs_correlation_Tq`); the fallback where the variances vanish is the
prescribed value. `ᶜT′T′` and `ᶜq′q′` must already be in the T basis and
bounded. The covariance diagnostic is formed lazily from `ᶜcorr_Tq` and the
variances (`compute_covariance_diagnostics`).
"""
function set_tq_correlation!(p, ::ConstantTqCorrelation)
    (; ᶜcorr_Tq) = p.precomputed
    corr = correlation_Tq(p.params)
    @. ᶜcorr_Tq = corr
    return nothing
end
function set_tq_correlation!(p, ::DiagnosedTqCorrelation)
    (; ᶜT′T′, ᶜq′q′, ᶜT′q′, ᶜcorr_Tq) = p.precomputed
    fallback = correlation_Tq(p.params)
    @. ᶜcorr_Tq = sgs_correlation_Tq(ᶜT′T′, ᶜq′q′, ᶜT′q′, fallback)
    return nothing
end

"""
    set_covariance_cache!(Y, p, thermo_params)

Materializes T-based SGS covariances into cached fields for use by downstream
computations (SGS quadrature, cloud fraction). Populates
`p.precomputed.(ᶜT′T′, ᶜq′q′, ᶜcorr_Tq)` (and `ᶜT′q′` for the diagnosed correlation).

Pipeline:

 1. Compute mixing length via `materialized_mixing_length!`
 2. Materialize θ-based turbulent covariances from gradients (`2 c l_mix² |∇_z ψ|²`).
    For the diagnosed T–q correlation the turbulent cross is set from the prescribed
    `ρ_turb` scalar as `ρ_turb · σ_T,turb · σ_q,turb` (not the gradient-direction
    product), so the turbulent block carries a physical correlation independent of
    the sign of `∂_z θ · ∂_z q`.
 3. Add the resolved-gradient (geometric) block, the variance from a subgrid
    parcel displacement with covariance `c_g diag(ℓ_h², ℓ_h², ℓ_v²)`,
    `ℓ_h = c_Δx Δx_h`, `ℓ_v = H`. The displacement along the `θ_li`
    surface adds `along_surface_q_variance` to q′q′ only, since θ_li is
    constant along the surface; its vertical scale is the depth `H` of the
    unresolved slantwise motion (`sgs_variance_isentropic_depth`), bounded
    by the height above the surface, `min(H, z − z_sfc)`.
    The displacement across the surface adds
    the vertical uniform-box terms `c_g ℓ_v² (∂_z θ)²` to θ′θ′,
    `c_g ℓ_v² q_n²` to q′q′ and `c_g ℓ_v² ∂_z θ q_n` to θ′q′, with `q_n` the
    across-surface q gradient (`across_surface_q_gradient`). The part of the
    geometric variance that needs a vertical displacement
    (`geometric_vertical_q_variance`, and `c_g ℓ_v² (∂_z θ)²` for θ) and the
    turbulent block's vertical-displacement variance are two estimates of
    the same quantity, so the block takes the larger of the two instead of
    adding them; the along-contour part of the along-surface variance adds.
    The block is skipped when `c_Δx` and `c_Δz` are both 0,
    and the horizontal invariants are computed only when `c_Δx` is nonzero.
 4. Transform θ→T using `compute_∂T_∂θ!`
 5. Apply the closure-validity bound `σ_q ≤ sgs_variance_max_rel_std * q_tot`
 6. Set the sampled T–q correlation (`set_tq_correlation!`; for the diagnosed
    model the covariance is first rescaled with the bound so the correlation is
    invariant under it)

The turbulent block uses the prescribed `correlation_Tq` for its cross-covariance;
the across-surface part of the resolved-gradient block uses the correlation set
by the gradient directions.
"""
function set_covariance_cache!(Y, p, thermo_params)
    # Covariance fields are only allocated when the configuration needs them.
    # No-op otherwise (e.g. EquilMoist + 0M + GridScaleCloud).
    uses_covariances(p.atmos) || return nothing

    FT = eltype(p.params)
    diag_corr = p.atmos.tq_correlation_model isa DiagnosedTqCorrelation
    (;
        ᶜT′T′,
        ᶜq′q′,
        ᶜgradᵥ_q_tot,
        ᶜgradᵥ_θ_liq_ice,
        ᶜq_tot_nonneg,
    ) = p.precomputed

    # Resolved-gradient (geometric) displacement scales. `c_Δx` and `c_Δz`
    # scale the grid lengths; the block is skipped when both are 0.
    c_g = CAP.sgs_variance_geometric_coeff(p.params)
    c_Δx = CAP.sgs_variance_horizontal_scale_factor(p.params)
    c_Δz = CAP.sgs_variance_vertical_scale_factor(p.params)
    use_horizontal = !iszero(c_Δx)
    use_geometric = use_horizontal || !iszero(c_Δz)
    ℓ_h = if use_horizontal
        c_Δx * FT(
            Spaces.node_horizontal_length_scale(
                Spaces.horizontal_space(axes(Y.c)),
            ),
        )
    else
        zero(FT)
    end

    ᶜl_mix = materialized_mixing_length!(Y, p)
    coeff = CAP.diagnostic_covariance_coeff(p.params)
    cov_from_grad(C, L, ∇Φ, ∇Ψ) = 2 * C * L^2 * dot(∇Φ, ∇Ψ)

    # ----------------------------------------------------------------------
    # 1. Turbulent block (θ basis).
    #    The cross uses the *prescribed* `correlation_Tq` scalar `ρ_turb`,
    #    not the gradient-direction product `cov_from_grad(c, L, ∇θ, ∇q)`,
    #    which would pin |ρ_turb,eff| = 1 with sign tied to
    #    `sign(∂_z θ · ∂_z q)` and flip across every gradient sign change
    #    (giving ρ = −1 in the stable BL where ∂_z θ > 0, ∂_z q < 0, though
    #    observed T–q correlation there is positive). The σ_T, σ_q amplitudes
    #    still carry the l_mix² · |∇_z ψ|² scaling via the turbulent variances.
    # ----------------------------------------------------------------------
    @. ᶜq′q′ = cov_from_grad(
        coeff, ᶜl_mix,
        Geometry.WVector(ᶜgradᵥ_q_tot),
        Geometry.WVector(ᶜgradᵥ_q_tot),
    )
    @. ᶜT′T′ = cov_from_grad(  # holds θ′θ′ until the ∂T/∂θ transform below.
        coeff, ᶜl_mix,
        Geometry.WVector(ᶜgradᵥ_θ_liq_ice),
        Geometry.WVector(ᶜgradᵥ_θ_liq_ice),
    )
    ρ_turb = FT(correlation_Tq(p.params))
    if diag_corr
        @. p.precomputed.ᶜT′q′ = ρ_turb * sqrt(ᶜT′T′ * ᶜq′q′)
    end

    # ----------------------------------------------------------------------
    # 2. Resolved-gradient (geometric) block: a subgrid parcel displacement
    #    with covariance c_g diag(ℓ_h², ℓ_h², ℓ_v²). Parcels keep θ_li and
    #    q_tot, so the displacement along the θ_li surface changes q_tot
    #    only (`along_surface_q_variance`). The displacement across the
    #    surface is the vertical uniform-box block, with the q gradient
    #    taken across the surface (`across_surface_q_gradient`). The two
    #    parts are orthogonal. q′q′ is in the q basis, so adding it before
    #    the θ→T transform is the same as adding it after. Scratch:
    #    `ᶜtemp_scalar_2`, `_3`, `_5`, `_6` are free until the θ→T transform.
    # ----------------------------------------------------------------------
    if use_geometric
        ᶜlg = Fields.local_geometry_field(Y.c)
        ᶜΔz = Fields.Δz_field(axes(Y.c))
        ᶜℓ_v = @. lazy(c_Δz * ᶜΔz)
        ᶜ∂q∂z = @. lazy(projected_vector_data(C3, ᶜgradᵥ_q_tot, ᶜlg))
        ᶜ∂θ∂z = @. lazy(projected_vector_data(C3, ᶜgradᵥ_θ_liq_ice, ᶜlg))
        if use_horizontal
            (; ᶜT, ᶜq_liq, ᶜq_ice) = p.precomputed
            ᶜθ_li = p.scratch.ᶜtemp_scalar_3
            @. ᶜθ_li = TD.liquid_ice_pottemp(
                thermo_params, ᶜT, Y.c.ρ, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice,
            )
            ᶜinv_q = p.scratch.ᶜtemp_scalar_5
            ᶜinv_θ = p.scratch.ᶜtemp_scalar_2
            ᶜinv_θq = p.scratch.ᶜtemp_scalar_6
            hgrad_invariant!(ᶜinv_q, ᶜq_tot_nonneg, p)
            hgrad_invariant!(ᶜinv_θ, ᶜθ_li, p)
            hgrad_cross_invariant!(ᶜinv_θq, ᶜθ_li, ᶜq_tot_nonneg, p)
            # Vertical scale of the along-surface displacement: the depth `H`
            # of the unresolved slantwise motion, bounded by the height above
            # the surface. In a neutral layer the θ_li surface is vertical and
            # the along-surface displacement is the vertical eddy displacement,
            # which cannot reach below the ground.
            H = CAP.sgs_variance_isentropic_depth(p.params)
            ᶜz = Fields.coordinate_field(Y.c).z
            z_sfc = Fields.level(Fields.coordinate_field(Y.f).z, Fields.half)
            ᶜℓ_a = @. lazy(min(H, ᶜz - z_sfc))
            add_resolved_gradient_block!(
                p, ᶜinv_q, ᶜinv_θq, ᶜinv_θ, ᶜ∂q∂z, ᶜ∂θ∂z, ℓ_h, ᶜℓ_a, ᶜℓ_v, c_g,
                diag_corr, ρ_turb,
            )
        else
            add_resolved_gradient_block!(
                p, zero(FT), zero(FT), zero(FT), ᶜ∂q∂z, ᶜ∂θ∂z, ℓ_h, ᶜℓ_v, ᶜℓ_v, c_g,
                diag_corr, ρ_turb,
            )
        end
    end

    # ----------------------------------------------------------------------
    # 3. θ → T basis via the thermodynamic Jacobian ∂T/∂θ. q is unaffected.
    # ----------------------------------------------------------------------
    ᶜ∂T_∂θ = p.scratch.ᶜtemp_scalar_2
    compute_∂T_∂θ!(ᶜ∂T_∂θ, Y, p, thermo_params)
    @. ᶜT′T′ = ᶜ∂T_∂θ^2 * ᶜT′T′
    if diag_corr
        @. p.precomputed.ᶜT′q′ = ᶜ∂T_∂θ * p.precomputed.ᶜT′q′
    end

    # ----------------------------------------------------------------------
    # 4. Closure-validity bound `σ_q ≤ sgs_variance_max_rel_std · q_tot`
    #    (default 0.5). A wider Gaussian puts a quadrature node at negative
    #    total water; the clamped quadrature then carries more total water
    #    than the grid mean holds, and condensing it can drive the grid-mean
    #    vapour negative. For the diagnosed path we rescale T′q′ by the same
    #    factor as σ_q so the diagnosed correlation is invariant under the
    #    bound.
    # ----------------------------------------------------------------------
    r_max = CAP.sgs_variance_max_rel_std(p.params)
    if diag_corr
        @. p.precomputed.ᶜT′q′ *= sqrt(min(
            one(FT),
            (r_max * ᶜq_tot_nonneg)^2 / max(ᶜq′q′, eps(FT)^2),
        ))
    end
    @. ᶜq′q′ = min(ᶜq′q′, (r_max * ᶜq_tot_nonneg)^2)

    # ----------------------------------------------------------------------
    # 5. Fill `ᶜcorr_Tq` with the correlation the quadrature samples.
    # ----------------------------------------------------------------------
    set_tq_correlation!(p, p.atmos.tq_correlation_model)

    # ----------------------------------------------------------------------
    # 7. Refresh `ᶜα` with the current `ᶜl_mix` and state.
    # ----------------------------------------------------------------------
    materialized_variance_fidelity!(Y, p)
    return nothing
end


"""
    add_resolved_gradient_block!(p, ᶜinv_q, ᶜinv_θq, ᶜinv_θ, ᶜ∂q∂z, ᶜ∂θ∂z,
                                 ℓ_h, ᶜℓ_a, ᶜℓ_v, c_g, diag_corr, ρ_turb)

Combine the resolved-gradient (geometric) block with the turbulent block
already held in `ᶜq′q′` and `ᶜT′T′` (θ basis). The along-contour part of the
along-surface q variance (the displacement with no vertical excursion) adds;
the part that needs a vertical displacement (`geometric_vertical_q_variance`,
with the vertical scale `ᶜℓ_a` along the surface and `ᶜℓ_v` across it) and
the turbulent block's vertical-displacement variance describe the same
displacement with two closures, so the larger of the two is kept:

    q′q′ = c_g V_h + max(c_g V_v, q′q′_turb),
    θ′θ′ = max(c_g ℓ_v² (∂_z θ)², θ′θ′_turb).

When `diag_corr`, the cross covariance of the vertical displacement is the
prescribed `ρ_turb` on the combined vertical amplitudes; the along-surface
variance adds no cross.
"""
function add_resolved_gradient_block!(
    p,
    ᶜinv_q,
    ᶜinv_θq,
    ᶜinv_θ,
    ᶜ∂q∂z,
    ᶜ∂θ∂z,
    ℓ_h,
    ᶜℓ_a,
    ᶜℓ_v,
    c_g,
    diag_corr,
    ρ_turb,
)
    (; ᶜT′T′, ᶜq′q′) = p.precomputed
    ᶜq_n = @. lazy(across_surface_q_gradient(ᶜinv_θq, ᶜinv_θ, ᶜ∂q∂z, ᶜ∂θ∂z))
    ᶜV_h = @. lazy(
        along_surface_q_variance(
            ᶜinv_q, ᶜinv_θq, ᶜinv_θ, ᶜ∂q∂z, ᶜ∂θ∂z, ℓ_h, zero(ℓ_h),
        ),
    )
    ᶜV_v = @. lazy(
        geometric_vertical_q_variance(
            ᶜinv_q, ᶜinv_θq, ᶜinv_θ, ᶜ∂q∂z, ᶜ∂θ∂z, ℓ_h, ᶜℓ_a, ᶜℓ_v,
        ),
    )
    ᶜθθ_v = @. lazy(ᶜℓ_v^2 * ᶜ∂θ∂z^2)
    if diag_corr
        # Uses the turbulent values still in `ᶜT′T′`, `ᶜq′q′`.
        @. p.precomputed.ᶜT′q′ =
            ρ_turb * sqrt(
                max(c_g * ᶜθθ_v, ᶜT′T′) * max(c_g * ᶜℓ_v^2 * ᶜq_n^2, ᶜq′q′),
            )
    end
    @. ᶜq′q′ = c_g * ᶜV_h + max(c_g * ᶜV_v, ᶜq′q′)
    @. ᶜT′T′ = max(c_g * ᶜθθ_v, ᶜT′T′)
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
#     σ_aug = sqrt((α · σ_S)² + σ_S_floor²),
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
    sgs_microphysics_timescale(thp, T, τ_cond, τ_sub)

Phase-blended microphysics relaxation timescale
`τ_mp = λ_eq(T)·τ_cond + (1 − λ_eq(T))·τ_sub`, with `λ_eq` the equilibrium
liquid-fraction ramp. Used by `sgs_variance_fidelity`.
"""
@inline function sgs_microphysics_timescale(thp, T, τ_cond, τ_sub)
    FT = typeof(T)
    λ_eq = FT(TD.liquid_fraction_ramp(thp, T))
    return λ_eq * FT(τ_cond) + (one(FT) - λ_eq) * FT(τ_sub)
end

"""
    sgs_variance_fidelity(
        A, τ_mp, c_d, tke, l_mix, w_eq, q_c_actual, mu_S, sigma_S, q_min,
    )

Local SGS variance fidelity `α = α_Da · α_eq` for the condensate
reconstruction `E[max(0, λ + α·S′)] = q_c`.

  - Damköhler factor
    `α_Da = 1 / (1 + A · τ_mp · c_d · √TKE / max(l_mix, 1 m))`:
    quiescent (TKE → 0) → 1, well-mixed → 0.
  - Equilibrium-ratio modulator `α_eq = r / (w_eq + (1 − w_eq) r)` with
    `r = (q_c_actual + q_min) / (q_c_eq + q_min)` and `q_c_eq` the
    truncated-Gaussian expectation `μ_S Φ(z) + σ_S φ(z)`, `z = μ_S/σ_S` —
    the expected saturation excess under the SGS PDF (always ≥ 0 even for
    subsaturated means).
"""
@inline function sgs_variance_fidelity(
    A, τ_mp, c_d, tke, l_mix, w_eq, q_c_actual, mu_S, sigma_S, q_min,
)
    FT = typeof(tke)
    inv_τ_turb = FT(c_d) * sqrt(max(tke, zero(FT))) / max(l_mix, one(FT))
    α_Da = one(FT) / (one(FT) + FT(A) * FT(τ_mp) * inv_τ_turb)
    inv_sqrt2π = one(FT) / sqrt(FT(2) * FT(π))
    z = FT(mu_S) / max(FT(sigma_S), ϵ_numerics(FT))
    q_c_eq = FT(mu_S) * normal_cdf(z) + FT(sigma_S) * exp(-z * z / 2) * inv_sqrt2π
    r = (FT(q_c_actual) + FT(q_min)) / (q_c_eq + FT(q_min))
    w = FT(w_eq)
    α_eq = r / (w + (one(FT) - w) * r)
    return α_Da * α_eq
end


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
relation `q_c/σ_aug = z·Φ(z) + φ(z)` (see `_compute_z`) with
`σ_aug = sqrt((α · σ_S)² + σ_S_floor²)`. `α` scales only the equilibrium
part; the non-equilibrium floor adds in quadrature. `floor` is a
`CloudFractionFloorParams` bundle (see `cloud_fraction_floor_params`).

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

    w  = sqrt((c_w)²·((α·σ_S)² + σ_abs²) + (c_a·ε_rel·q_sat)²),
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
        margin^2 * ((α * sigma_S)^2 + σ_abs^2) +
        (abs_margin * ε_rel * q_sat)^2,
    )
    x = max(mu_S, zero(FT)) / max(w, ϵ_numerics(FT))
    # `sharpness == 1` is the default release profile; the fast path keeps
    # it identical to the plain saturation-margin release.
    D_shape =
        sharpness == one(FT) ? 1 / sqrt(1 + x^2) : (1 + x^2)^(-sharpness / 2)
    D = residual + (1 - residual) * D_shape
    σ_S_floor_sq = (D * ε_rel * q_sat)^2 + σ_abs^2
    # α scales only the equilibrium part; the non-eq floor adds in quadrature.
    σ_aug = sqrt((α * sigma_S)^2 + σ_S_floor_sq)
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
    _compute_sgs_moments(thp, ρ, T, q_tot, q_c, sgs_quad, T′T′, q′q′, corr_Tq, α)

Single quadrature pass returning `(sigma_S, λ_lagrange)`:

  - `sigma_S = sqrt(Σᵢ wᵢ·S′ᵢ²)`: SGS standard deviation of the sampled
    centred excess, clipped at `ϵ_numerics(FT)`.
  - `λ_lagrange`: Lagrange multiplier satisfying the mass-conservation
    constraint `Σᵢ wᵢ·max(0, λ + α·S′ᵢ) = q_c` exactly under the discrete
    quadrature measure — the same points and weights the microphysics
    evaluator integrates over (see `_fit_discrete_lagrange`; the analytic
    truncated-Gaussian inverse `_compute_z` provides the seed).

The SGS mean `μ_S = q_tot − q_sat(T, ρ)` is analytic under the closure's
linearization (see `_sgs_saturation_moments`) and is recomputed on demand
wherever it is needed downstream.

Without SGS sampling (`nothing`, `GridMeanSGS`), all mass sits at the mean:
`sigma_S = ϵ_numerics(FT)` and the constraint gives `λ_lagrange = q_c`
directly, matching the σ_S → 0 limit of the sampled branch.
"""
@inline function _compute_sgs_moments(
    thp, ρ, T, q_tot, q_c,
    sgs_quad, T′T′, q′q′, corr_Tq, α,
)
    FT = typeof(ρ)
    not_quadrature(sgs_quad) &&
        return (; sigma_S = ϵ_numerics(FT), λ_lagrange = q_c)

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
    return (; sigma_S, λ_lagrange)
end

"""
    set_sgs_moments_and_cloud_fraction!(Y, p)

Final post-Aitken update. No-op when `ᶜsgs_moments` is not allocated (dry / 0M).

Uses ONE quadrature pass via `_compute_sgs_moments` to fill
`ᶜsgs_moments = (sigma_S, λ_lagrange)`, then computes
`ᶜcloud_fraction` consistently with the augmented `σ_aug` closure (see
`_compute_cloud_fraction`) from the environment cloud condensate
(`_get_condensate_means`) and applies EDMF updraft weighting. Finally
runs `set_precip_fraction!`, the column overlap sweep that turns the fresh
cover into the precipitation fraction `ᶜprecip_frac`.

Overwrites `p.scratch.ᶜtemp_scalar`, `ᶜtemp_scalar_2`, and `ᶜtemp_scalar_3`.
It must not touch `ᶜtemp_scalar_4` or `ᶜtemp_scalar_7`, which
`set_covariance_cache_and_cloud_fraction!` reserves for the Picard iterates.
"""
NVTX.@annotate function set_sgs_moments_and_cloud_fraction!(Y, p)
    hasproperty(p.precomputed, :ᶜsgs_moments) || return nothing

    thermo_params = CAP.thermodynamics_params(p.params)
    turbconv_model = p.atmos.turbconv_model
    microphysics_model = p.atmos.microphysics_model

    ᶜρ_env_lazy, ᶜT_mean, ᶜq_mean = _get_env_ρ_T_q(Y, p, thermo_params, turbconv_model)
    ᶜq_lcl_lazy, ᶜq_icl_lazy =
        _get_condensate_means(Y, p, turbconv_model, microphysics_model)
    sgs_quad = p.atmos.sgs_quadrature
    floor = cloud_fraction_floor_params(p.params)
    (; ᶜT′T′, ᶜq′q′, ᶜcorr_Tq, ᶜα, ᶜsgs_moments, ᶜcloud_fraction) =
        p.precomputed

    # Materialize lazy fields to pass to foreach_point. Only the scratch fields
    # that the Picard step already clobbers are used, so the iterates that
    # `set_covariance_cache_and_cloud_fraction!` keeps in ᶜtemp_scalar_4 and
    # ᶜtemp_scalar_7 are never overwritten.
    ᶜρ_env = (p.scratch.ᶜtemp_scalar .= ᶜρ_env_lazy)
    ᶜq_lcl = (p.scratch.ᶜtemp_scalar_2 .= ᶜq_lcl_lazy)
    ᶜq_icl = (p.scratch.ᶜtemp_scalar_3 .= ᶜq_icl_lazy)

    DataLayouts.foreach_point(
        ᶜsgs_moments, ᶜcloud_fraction, ᶜρ_env, ᶜT_mean, ᶜq_mean,
        ᶜq_lcl, ᶜq_icl, ᶜT′T′, ᶜq′q′, ᶜcorr_Tq, ᶜα,
    ) do ᶜsgs_moments, ᶜcloud_fraction, ᶜρ_env, ᶜT_mean, ᶜq_mean,
    ᶜq_lcl, ᶜq_icl, ᶜT′T′, ᶜq′q′, ᶜcorr_Tq, ᶜα

        @. ᶜsgs_moments = _compute_sgs_moments(
            thermo_params, ᶜρ_env, ᶜT_mean, ᶜq_mean, ᶜq_lcl + ᶜq_icl,
            $(sgs_quad), ᶜT′T′, ᶜq′q′, ᶜcorr_Tq, ᶜα,
        )
        # Recompute CF from q_c and σ_S using the augmented-σ closure. We cannot
        # use `Φ(λ/σ_aug)` because λ was computed with the equilibrium σ_S_eff,
        # not σ_aug — `Φ(λ/σ_aug)` would not match the truncated-Gaussian
        # closure for the augmented variance. This overwrites the Picard iterate
        # with a value consistent with the final SGS moments, so EDMF weighting
        # must be re-applied below even though `set_cloud_fraction!` already
        # applied it during Picard.
        @. ᶜcloud_fraction = _compute_cloud_fraction(
            ᶜq_lcl + ᶜq_icl,
            ᶜq_mean - TD.q_vap_saturation(thermo_params, ᶜT_mean, ᶜρ_env),
            ᶜsgs_moments.sigma_S,
            TD.q_vap_saturation(thermo_params, ᶜT_mean, ᶜρ_env, ᶜq_lcl, ᶜq_icl),
            ᶜα,
            $(floor),
        )
    end

    _apply_edmf_cloud_weighting!(Y, p, turbconv_model, thermo_params)
    set_precip_fraction!(Y, p)
    return nothing
end

"""
    set_precip_fraction!(Y, p)

Fill `p.precomputed.ᶜprecip_frac` with the precipitation fraction `a_p`, the
area fraction of the cell that precipitation falls through, by maximum-random
overlap of the cloud cover swept from the model top down:

    a_p(k) = max(CF(k), f_decay · a_p(k+1))   where q_rai + q_sno > q_min
    a_p(k) = 0                                 otherwise

with `f_decay = sgs_precip_overlap_decay` (`1` is pure maximum overlap; the
shrink is applied per level). A precipitation-free level resets the recursion,
so a shaft that evaporates completely does not seed the layers below it.

The recursion is seeded with `ᶜcloud_fraction`, the single-domain cover of the
cell; the precipitation mask uses the precipitation of the
domain the quadrature integrates (the environment under `PrognosticEDMFX`).
`a_p` enters the quadrature only as the fraction of the PDF that carries the
shaft (`sgs_precip_shaft_threshold`); the per-point assignment is normalized
by the discrete weight of the selected points, so no measure consistency
between the cover and the quadrature is required for conservation.

Mutates `p.precomputed.ᶜprecip_frac`.
"""
NVTX.@annotate function set_precip_fraction!(Y, p)
    hasproperty(p.precomputed, :ᶜprecip_frac) || return nothing
    FT = eltype(p.params)
    f_decay = clamp(FT(CAP.sgs_precip_overlap_decay(p.params)), zero(FT), one(FT))
    thermo_params = CAP.thermodynamics_params(p.params)
    _precip_fraction_sweep!(
        p.precomputed.ᶜprecip_frac,
        p.precomputed.ᶜcloud_fraction,
        _get_precip_mean(Y, p, p.atmos.turbconv_model),
        f_decay,
        FT(TD.Parameters.q_min(thermo_params)),
    )
    return nothing
end

"""
    _precip_fraction_sweep!(ᶜprecip_frac, ᶜcf, ᶜq_precip, f_decay, q_precip_min)

Run the maximum-random overlap recursion of `set_precip_fraction!` on plain
fields, writing `a_p` into `ᶜprecip_frac`. Levels with
`ᶜq_precip ≤ q_precip_min` carry no shaft; the cover is taken non-negative.

Implemented as a loop of level broadcasts rather than with
`Operators.column_accumulate!`: the two give the same result to the bit, but
the CPU path of `column_accumulate!` materializes a level `Field` per level
per column. The loop allocates per level independent of the number
of columns; its cost on GPU is one kernel launch per level.
"""
function _precip_fraction_sweep!(ᶜprecip_frac, ᶜcf, ᶜq_precip, f_decay, q_precip_min)
    FT = eltype(ᶜprecip_frac)
    # A negative value is the sentinel for a precipitation-free level, so the
    # seed takes the cover clamped non-negative.
    @. ᶜprecip_frac =
        ifelse(ᶜq_precip > q_precip_min, max(ᶜcf, zero(FT)), -one(FT))
    # Level 1 is the bottom model level, so the recursion runs from the last
    # level down. The top level has no shaft above it, so its sentinel just
    # resolves to zero.
    nz = Spaces.nlevels(axes(ᶜprecip_frac))
    ᶜa_p_above = Fields.level(ᶜprecip_frac, nz)
    @. ᶜa_p_above = max(ᶜa_p_above, zero(FT))
    for level in (nz - 1):-1:1
        ᶜa_p = Fields.level(ᶜprecip_frac, level)
        @. ᶜa_p = ifelse(
            ᶜa_p < zero(FT),
            zero(FT),
            min(one(FT), max(ᶜa_p, f_decay * ᶜa_p_above)),
        )
        ᶜa_p_above = ᶜa_p
    end
    return nothing
end
# `column_accumulate!` form of the same recursion (allocates on the CPU path,
# see the docstring above):
#     input = Base.broadcasted(tuple, ᶜcf, ᶜq_precip)
#     Operators.column_accumulate!(
#         ᶜprecip_frac, input; init = zero(FT), reverse = true,
#     ) do a_p_above, (cf, q_precip)
#         ifelse(
#             q_precip > q_precip_min,
#             min(one(FT), max(cf, f_decay * a_p_above)),
#             zero(FT),
#         )
#     end

"""
    _get_precip_mean(Y, p, turbconv_model)

Mean precipitation specific humidity `q_rai + q_sno` of the domain that carries
the SGS closure: the environment for `PrognosticEDMFX`, the grid mean
otherwise.
"""
function _get_precip_mean(Y, p, turbconv_model)
    if turbconv_model isa PrognosticEDMFX
        return @. lazy(
            max(0, $(ᶜspecific_env_value(@name(q_rai), Y, p))) +
            max(0, $(ᶜspecific_env_value(@name(q_sno), Y, p))),
        )
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
    ᶜρ_env_lazy, ᶜT_mean, ᶜq_mean = _get_env_ρ_T_q(Y, p, thermo_params, turbconv_model)

    # Materialize lazy fields to pass to foreach_point. This method runs inside
    # the Picard step of `set_covariance_cache_and_cloud_fraction!`, so it may
    # never touch ᶜtemp_scalar_4 or ᶜtemp_scalar_7 (which hold the Picard
    # iterates).
    ᶜρ_env = (p.scratch.ᶜtemp_scalar .= ᶜρ_env_lazy)

    # Condensate of the domain that carries the SGS closure (env under
    # PrognosticEDMFX, grid-mean otherwise). Updraft contributions are added
    # separately by `_apply_edmf_cloud_weighting!`.
    ᶜq_lcl_lazy, ᶜq_icl_lazy =
        _get_condensate_means(Y, p, turbconv_model, microphysics_model)
    ᶜq_lcl = (p.scratch.ᶜtemp_scalar_2 .= ᶜq_lcl_lazy)
    ᶜq_icl = (p.scratch.ᶜtemp_scalar_3 .= ᶜq_icl_lazy)

    sgs_quad = p.atmos.sgs_quadrature
    floor = cloud_fraction_floor_params(p.params)

    (; ᶜT′T′, ᶜq′q′, ᶜcorr_Tq, ᶜα) = p.precomputed
    ᶜcloud_fraction = p.precomputed.ᶜcloud_fraction

    DataLayouts.foreach_point(
        ᶜcloud_fraction, ᶜT_mean, ᶜρ_env, ᶜq_mean, ᶜq_lcl, ᶜq_icl,
        ᶜT′T′, ᶜq′q′, ᶜcorr_Tq, ᶜα,
    ) do ᶜcloud_fraction, ᶜT_mean, ᶜρ_env, ᶜq_mean, ᶜq_lcl, ᶜq_icl,
    ᶜT′T′, ᶜq′q′, ᶜcorr_Tq, ᶜα

        @. ᶜcloud_fraction = _compute_cloud_fraction(
            thermo_params, ᶜT_mean, ᶜρ_env, ᶜq_mean, ᶜq_lcl, ᶜq_icl,
            $(sgs_quad), ᶜT′T′, ᶜq′q′, ᶜcorr_Tq, ᶜα, $(floor),
        )
    end

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
    ᶜρ_env, ᶜT_mean, ᶜq_mean, ᶜθ_mean, _, _, _, _ =
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

        # Add contributions from the updrafts if using EDMF
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
not applied in machine learning cloud fracrtion.

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

    ᶜlg = Fields.local_geometry_field(Y.c)

    # Vertical gradients of q_tot and θ_liq_ice
    ᶜ∇q = p.scratch.ᶜtemp_scalar_2
    @. ᶜ∇q = projected_vector_data(C3, p.precomputed.ᶜgradᵥ_q_tot, ᶜlg)
    ᶜ∇θ = p.scratch.ᶜtemp_scalar_3
    @. ᶜ∇θ = projected_vector_data(C3, p.precomputed.ᶜgradᵥ_θ_liq_ice, ᶜlg)

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
