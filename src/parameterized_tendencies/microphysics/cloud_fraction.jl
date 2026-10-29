import NVTX
import StaticArrays as SA
import ClimaComms

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
    (; cloud_model, microphysics_model, turbconv_model) = p.atmos
    (; ᶜgradᵥ_q_tot, ᶜgradᵥ_θ_liq_ice, ᶜcloud_fraction) = p.precomputed
    (; ᶜbuoygrad, ᶜbg_coeffs, ᶜq_tot_nonneg) = p.precomputed
    thermo_params = CAP.thermodynamics_params(p.params)

    # Materialize the pointwise buoyancy-gradient chain-rule coefficients,
    # exact face gradients, and centered vertical gradients of (θ_li, q_tot)
    # once (`ᶜθ_li` remains in `p.scratch.ᶜtemp_scalar`).
    set_buoyancy_gradient_inputs!(Y, p, thermo_params)

    # Precompute horizontal gradient invariants once before the Picard loop:
    # they depend only on `(ᶜq_tot_nonneg, ᶜθ_li)`, which are fixed during
    # Picard.
    c_Δx = CAP.sgs_variance_horizontal_scale_factor(p.params)
    if uses_covariances(p.atmos) && !iszero(c_Δx)
        ᶜθ_li = p.scratch.ᶜtemp_scalar
        ᶜinv_q = p.scratch.ᶜtemp_scalar_5
        ᶜinv_θ = Fields.Field(p.scratch.temp_data, axes(Y.c))
        ᶜinv_θq = p.scratch.ᶜtemp_scalar_6
        hgrad_invariant!(ᶜinv_q, ᶜq_tot_nonneg, p)
        hgrad_invariant!(ᶜinv_θ, ᶜθ_li, p)
        hgrad_cross_invariant!(ᶜinv_θq, ᶜθ_li, ᶜq_tot_nonneg, p)
    end

    if p.atmos.numerics.reproducible_restart
        set_cloud_fraction!(Y, p, microphysics_model, GridScaleCloud())
    end

    # A moist `QuadratureCloud` updates the cloud fraction in fused column loops,
    # with the state of the domain that carries the SGS closure fixed during
    # Picard; the other cloud models (and dry runs) call `set_cloud_fraction!`
    # and let `set_covariance_cache!` compute that state itself.
    fused = cloud_model isa QuadratureCloud && microphysics_model isa MoistMicrophysics
    is_edmf = turbconv_model isa PrognosticEDMFX
    env_state =
        fused ? cloud_env_state(Y, p, thermo_params; materialize = is_edmf) :
        nothing

    # Picard iterates: c0 = initial cloud fraction, c1 = first Picard iterate.
    c0 = p.scratch.ᶜtemp_scalar_4
    c1 = p.scratch.ᶜtemp_scalar_7
    @. c0 = ᶜcloud_fraction

    # One Picard step: fused buoygrad + covariance + l_mix + α, then the cloud
    # fraction. The first step stores its iterate in `c1`; the second applies
    # the Aitken acceleration.
    function picard_step!(::Val{aitken}) where {aitken}
        set_covariance_cache!(
            Y, p, thermo_params; recompute_buoygrad = true, env_state,
        )
        if fused
            set_quadrature_cloud_fraction!(Y, p, env_state, c0, c1, Val(aitken))
        else
            set_cloud_fraction!(Y, p, microphysics_model, cloud_model)
            if aitken
                @. ᶜcloud_fraction = _aitken_picard_helper(c0, c1, ᶜcloud_fraction)
            else
                @. c1 = ᶜcloud_fraction
            end
        end
        return nothing
    end
    picard_step!(Val(false))
    picard_step!(Val(true))

    # Final post-Aitken covariance, then, for the 1M and 2M schemes, the SGS
    # moments, the cloud fraction consistent with them, and the precipitation
    # fraction.
    set_covariance_cache!(
        Y, p, thermo_params; recompute_buoygrad = true, env_state,
    )
    hasproperty(p.precomputed, :ᶜsgs_moments) || return nothing
    final_env_state =
        fused ? env_state :
        cloud_env_state(Y, p, thermo_params; materialize = is_edmf)
    set_sgs_moments_and_cloud_fraction!(Y, p, final_env_state)
    set_precip_fraction!(Y, p, final_env_state.ᶜq_precip)
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
reused by TKE dissipation, the covariance closure, variance fidelity, and
diagnostics); every `AbstractEDMF` carries `Y.c.ρtke`, so `ᶜmixing_length` is
well defined. Otherwise it falls back to the grid-mean Smagorinsky-Lilly
length, materialized in `p.scratch.ᶜtemp_scalar` by `compute_gm_mixing_length`.
"""
function materialized_mixing_length!(Y, p)
    if p.atmos.turbconv_model isa AbstractEDMF
        ᶜl_mix = p.precomputed.ᶜl_mix
        ᶜl_mix .= ᶜmixing_length(Y, p)
        return ᶜl_mix
    else
        return compute_gm_mixing_length(Y, p)
    end
end

"""
    floor_velocity(Δx_h, w_ref, Δx_ref, p)

Velocity scale `w_min = w_ref (Δx_h / Δx_ref)^p` of the unresolved vertical
motions at the horizontal grid scale `Δx_h`. A single column has no
horizontal grid scale (`Δx_h = Inf`) and stands for a cell of the reference
size, so it gets `w_ref`.

The exponent follows from the spectral slope. For a field with spectrum
`E(k) ∝ k^(−s)`, the variance below the grid scale is
`∫_{k > 2π/Δ} E dk ∝ Δ^(s−1)`. The horizontal term of the covariance,
`ℓ_h² |∇_h ψ|²` with `ℓ_h ∝ Δ`, scales the same way by itself, because the
variance of the coarse-grained gradient is
`∫_{k < 2π/Δ} k² E dk ∝ Δ^(s−3)`; so its coefficient is independent of
resolution. The vertical term `ℓ_v² (∂_z ψ)²` does not: the vertical
gradient is a vertical coarse-graining and does not change with `Δx_h`, so
the growth `Δ^(s−1)` of the sub-cell variance must be carried by the
amplitude, `ℓ_v ∝ w_min ∝ Δx_h^((s−1)/2)`. A `−5/3` spectrum gives
`p = 1/3`; the default `p = 2/3`, from the within-cell variances of
kilometre-scale global simulations coarse-grained to 60, 120 and 240 km,
corresponds to an effective slope near `−7/3`, between the mesoscale `−5/3`
and the synoptic `−3`, which is where the spectra of temperature and
humidity lie at those scales.
"""
@inline function floor_velocity(Δx_h, w_ref, Δx_ref, p)
    Δx = ifelse(isfinite(Δx_h), Δx_h, Δx_ref)
    return w_ref * (Δx / Δx_ref)^p
end

"""
    floor_displacement_length(κ_b, N², τ_turb, τ_0, l_z, Δz)

Vertical displacement `√κ_b · min(1/N, τ_turb, τ_0)` of the unresolved
vertical motions with buoyancy-weighted kinetic energy `κ_b = c_b κ_min`,
capped by the wall distance `l_z` and by the cell depth `Δz`, over which the
gradients the displacement acts on are defined. The displacement lasts until
the parcel's vertical velocity decorrelates: in stably stratified air
(`N² > 0`) after a time `1/N`, when buoyancy stops it, which gives the
buoyancy scale `√κ_b / N` of the mixing length; after the Lagrangian time
scale of the turbulence `τ_turb = l_disp / √(c_b κ_iso / (2 c_σ))`
(`turbulent_lagrangian_time`, with `l_disp` the buoyancy-bounded mixing
length); or after the quiescent limit `τ_0`. Where the air is neutral or
unstable (`N² ≤ 0`) buoyancy imposes no limit and the displacement is
`√κ_b · min(τ_turb, τ_0)`, the limit the stable branch approaches as
`N² → 0⁺`, so the closure is continuous in `N²`. This is the vertical
dispersion of a stochastic Lagrangian model in stratified flow.

`l_b` is a displacement scale in the normalization of the mixing length,
`ℓ_v² = 2 c_σ l²`: with `κ_b = w_min² / (2 c_σ)` the physical standard
deviation of the displacement is `√(2 c_σ) l_b = w_min · min(1/N, τ_turb, τ_0)`.
The caps bound `l_b` in that normalization, so that `max(l_mix, l_b)` in
`displacement_length` compares like with like (the wall limit `κ z` of the
mixing length is in it too), and the physical displacement is bounded by
`√(2 c_σ)` times the cell depth and the wall distance. A cap on the standard
deviation is loose in any normalization, since the Gaussian tails extend
beyond it, and the calibrated `w_ref` and `τ_0` assume the caps as written.
"""
@inline function floor_displacement_length(κ_b, N², τ_turb, τ_0, l_z, Δz)
    FT = typeof(N²)
    l_cap = min(max(l_z, zero(FT)), Δz)
    τ_N = ifelse(N² > zero(FT), inv(sqrt(max(N², floatmin(FT)))), FT(Inf))
    τ = min(τ_0, τ_turb, τ_N)
    # Without a floor velocity (`κ_b = 0`) the displacement vanishes even where
    # every decorrelation time is infinite, where `0 * Inf` would be NaN.
    l_b = ifelse(κ_b > zero(FT), sqrt(max(κ_b, zero(FT))) * τ, zero(FT))
    return min(l_b, l_cap)
end

"""
    turbulent_lagrangian_time(l_mix, κ_b)

Lagrangian time scale `l_mix / √κ_b` of the turbulence, with buoyancy-weighted
vertical velocity variance `κ_b = c_b κ_iso / (2 c_σ)`; infinite where there is
no TKE (`κ_b ≤ 0`), so that it imposes no limit there. Where `l_mix` exceeds
the buoyancy scale `√κ_b / N`, `turbulent_lagrangian_time(l_disp, κ_b)`
approaches `1 / N`, matching the buoyancy period of `floor_displacement_length`.
"""
@inline function turbulent_lagrangian_time(l_mix, κ_b)
    FT = typeof(κ_b)
    return ifelse(
        κ_b > floatmin(FT),
        l_mix / sqrt(max(κ_b, floatmin(FT))),
        FT(Inf),
    )
end

"""
    buoyancy_bounded_mixing_length(l_mix, κ_b, N²)

Turbulent mixing length `l_mix` bounded smoothly by the buoyancy scale
`√(κ_b / N²)`, as the harmonic combination
`l_disp = l_mix / √(1 + N² l_mix² / κ_b)`, so that
`1/l_disp² = 1/l_mix² + N²/κ_b`, where `N² > 0` and `κ_b > 0`; elsewhere
`l_disp = l_mix`. With the vertical velocity variance
`κ_b = c_b κ_iso / (2 c_σ)` of the displacing eddies, the vertical
displacement variance `2 c_σ l_disp²` stays within the adiabatic buoyancy
excursion `c_b κ_iso / N²`. The same face-maximum dry `N²` as in
`floor_displacement_length` keeps that bound active in a cell adjacent to a
capping inversion, where the Van Leer harmonic-mean center gradient vanishes
on the neutral side of the cell.
"""
@inline function buoyancy_bounded_mixing_length(l_mix, κ_b, N²)
    FT = typeof(N²)
    return ifelse(
        (N² > zero(FT)) & (κ_b > floatmin(FT)),
        l_mix /
        sqrt(
            one(FT) +
            max(N², zero(FT)) * l_mix^2 / max(κ_b, floatmin(FT)),
        ),
        l_mix,
    )
end

"""
    displacement_length(l_mix, l_b, l_z)

Vertical displacement length `min(max(l_mix, l_b), l_z)`: the larger of the
buoyancy-bounded mixing length `l_mix` and the floor displacement scale `l_b`,
capped by the wall distance `l_z` since air cannot be displaced below the
ground. The cap, like `l_mix` and `l_b`, is in the normalization of the
mixing length, so the physical standard deviation `√(2 c_σ) l_d` of the
displacement is bounded by `√(2 c_σ) l_z` (see `floor_displacement_length`).
Using `max(l_mix, l_b)` avoids double-counting turbulent and floor
displacements in quadrature when both represent the vertical parcel excursion.

In the variance closure (`set_covariance_cache!`), `l_d` takes the maximum of
`l_disp` (the mixing length `l_mix` bounded by
`buoyancy_bounded_mixing_length` with `κ_b = c_b κ_iso / (2 c_σ)`) and the floor
displacement scale `floor_displacement_length`, evaluated with the face-maximum
dry buoyancy gradient `N²_disp` (`set_displacement_buoyancy_gradient!`), with
`κ_min = w_min² / (2 c_σ c_b)`, the velocity scale `w_min` of the unresolved
vertical motions at the horizontal grid scale (`floor_velocity`;
`sgs_variance_vertical_velocity_floor`,
`sgs_variance_floor_reference_resolution`,
`sgs_variance_floor_resolution_exponent`) and the time over which their
displacement persists, the smallest of the buoyancy period `1/N_disp`, the
Lagrangian time scale of the turbulence `l_disp / √(c_b κ_iso / (2 c_σ))`
(`turbulent_lagrangian_time`, with `l_disp` the buoyancy-bounded mixing
length) and the quiescent limit `τ_0` (`sgs_variance_floor_time_scale`). In
neutral turbulent layers, `l_disp = l_mix` dominates and the vertical
displacement variance `2 c_σ l_d²` is `2 c_σ l_mix²` (the production–dissipation
balance), while in stably stratified layers `2 c_σ l_disp²` is smoothly bounded
by the adiabatic buoyancy excursion `c_b κ_iso / N²_disp`
(`buoyancy_bounded_mixing_length`). In stably stratified air without TKE,
`l_d = w_min / (√(2 c_σ) N_disp)`, so the vertical displacement variance
`2 c_σ l_d²` is `(w_min / N_disp)²`: the displacement of unresolved vertical
motions (gravity waves, mesoscale circulations) with velocity scale `w_min`,
which displace air without mixing it. Where the air is neutral or unstable
(`N²_disp ≤ 0`) buoyancy imposes no limit and the floor displacement is
`w_min min(τ_turb, τ_0) / √(2 c_σ)`, the limit of the stable branch as
`N²_disp → 0⁺`, so `l_d` is continuous in `N²_disp`. The cell-depth cap applies
to the floor scale only; `l_disp` keeps its own limits, so `l_d` exceeds the
cell depth wherever `l_disp` does. The two scales are combined via `max` and
bounded by the wall distance.
The floor is kept out of the mixing length itself, because the
surface-layer wall scale `l_W` of the mixing length vanishes with the
friction velocity and in a stable surface layer, which would collapse `l_d`
in the free troposphere. The floor enters the variance only, not `ᶜl_mix` or
the diffusivity.
"""
@inline displacement_length(l_mix, l_b, l_z) =
    min(max(l_mix, l_b), max(l_z, zero(l_z)))

"""
    displacement_covariance(ℓ_h², ℓ_v², inv_h, ∂ψ∂z, ∂φ∂z)

Covariance `ℓ_h² ∇_h ψ · ∇_h φ + ℓ_v² ∂_z ψ ∂_z φ` of two scalars under a
subgrid displacement with covariance `diag(ℓ_h², ℓ_h², ℓ_v²)`, from the
horizontal invariant `inv_h = ∇_h ψ · ∇_h φ` and the vertical gradients.
"""
@inline displacement_covariance(ℓ_h², ℓ_v², inv_h, ∂ψ∂z, ∂φ∂z) =
    ℓ_h² * inv_h + ℓ_v² * ∂ψ∂z * ∂φ∂z

"""
    displacement_curvature_coefficients(ℓ_v, χ, ∂q∂z, ∂θ∂z, ∂²q∂z², ∂²θ∂z²)

Coefficients `(lin, curv)` of the saturation excess of a parcel displaced
vertically by `δ = ℓ_v x`, `x` standard normal, to second order in `δ`,

    S = μ_S + lin x + ½ curv (x² − 1) + h,
    lin  = ℓ_v  (∂_z q_t − χ ∂_z θ_li),
    curv = ℓ_v² (∂_z² q_t − χ ∂_z² θ_li),

with `χ = r_{v,θ} γ ∂T/∂θ_li`, `γ = ∂q_sat/∂T`, the chain rule that maps the
potential-temperature profile to the saturation humidity the parcel meets at
the cell's pressure, with the `θ_li` displacement scaled by `r_{v,θ}` as in
`set_covariance_cache!`, and `h` the horizontal displacement part. The
quadratic term is centred so that the mean of `S` is `μ_S`; its variance
`lin²` is the vertical part of `σ_S²` of the linear closure.
"""
@inline function displacement_curvature_coefficients(
    ℓ_v, χ, ∂q∂z, ∂θ∂z, ∂²q∂z², ∂²θ∂z²,
)
    lin = ℓ_v * (∂q∂z - χ * ∂θ∂z)
    curv = ℓ_v^2 * (∂²q∂z² - χ * ∂²θ∂z²)
    return (; lin, curv)
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
    sgs_correlation_Tq(T′T′, q′q′, T′q′, T, q_tot, fallback)

Diagnosed SGS T–q correlation `clamp(T′q′ / (√T′T′ √q′q′), -1, 1)`, or
`fallback` where either variance vanishes relative to its own mean, i.e. is at
the round-off floor of the grid-mean value: `q′q′ ≤ (ε q_tot)²` or
`T′T′ ≤ (ε T)²` with `ε = eps(FT)`. The test is relative so that it does not
depend on the units or magnitude of the variables: an absolute floor on
`T′T′ q′q′` (such as `eps(FT)²`) is crossed in the dry upper troposphere, where
`q_tot` and hence `q′q′` are small but well resolved, and would replace a
diagnosed (typically negative) correlation with the prescribed value there.

In exact arithmetic the covariance comes from correlated displacements of
`θ_li` and `q_tot` whose cross-correlation lies in `[-1, 1]`
(`set_covariance_cache!`), so it satisfies the Cauchy-Schwarz inequality and
`|ρ| ≤ 1` by construction. The explicit clamp is a cheap safeguard against
Float32 roundoff at the Cauchy-Schwarz boundary (which can produce
`|ρ| = 1 + O(ε)` and feed NaN into downstream `√(1 − ρ²)` in the bivariate
quadrature) and a user-supplied `Tq_correlation_coefficient` outside `[-1, 1]`.
"""
@inline function sgs_correlation_Tq(T′T′, q′q′, T′q′, T, q_tot, fallback)
    FT = typeof(T′T′)
    vT = max(T′T′, zero(FT))
    vq = max(q′q′, zero(FT))
    resolved = (vT > (eps(FT) * T)^2) & (vq > (eps(FT) * q_tot)^2)
    # √ of each variance separately, so the denominator does not underflow
    # where q′q′ is tiny but well above its round-off floor.
    σ = sqrt(vT) * sqrt(vq)
    return ifelse(
        resolved & (σ > zero(FT)),
        clamp(T′q′ / max(σ, floatmin(FT)), -one(FT), one(FT)),
        fallback,
    )
end

@inline _disp_lin(d::NamedTuple) = d.lin
@inline _disp_lin(d::Real) = zero(d)
@inline _disp_curv(d::NamedTuple) = d.curv
@inline _disp_curv(d::Real) = zero(d)
@inline _sgs_sigma_S(m) = m.sigma_S
@inline _1(t) = t[1]
@inline _2(t) = t[2]
@inline _3(t) = t[3]
@inline _4(t) = t[4]
@inline _5(t) = t[5]
@inline _6(t) = t[6]

"""
    set_covariance_cache!(Y, p, thermo_params)

Materializes T-based SGS covariances into cached fields for use by downstream
computations (SGS quadrature, cloud fraction). Populates
`p.precomputed.(ᶜT′T′, ᶜq′q′, ᶜcorr_Tq)` (and `ᶜT′q′` for the diagnosed correlation).

The covariance of `(θ_li, q_tot)` is that of a Gaussian subgrid displacement
acting on the resolved gradients (`displacement_covariance`), with moisture
horizontal scale `ℓ_{h,0}² = c_g (c_Δx Δx_h)²` from the horizontal grid scale,
moisture vertical displacement scale `ℓ_{v,0}² = 2 c_σ l_d²` from the
displacement length `l_d` (`displacement_length`),
potential-temperature scale ratios `r_{Δx,θ} = ℓ_{h,θ} / ℓ_{h,0}`
(`sgs_variance_theta_horizontal_scale_ratio`) and
`r_{v,θ} = ℓ_{v,θ} / ℓ_{v,0}` (`sgs_variance_theta_vertical_scale_ratio`), and
horizontal displacement cross-correlation `r_{h,θq}`
(`sgs_variance_horizontal_cross_correlation`); the vertical displacements of
the two scalars are perfectly correlated. The resulting `2 × 2` covariance is
positive semi-definite for `|r_{h,θq}| ≤ 1`, which `SGSQuadratureParameters`
enforces.

Each column is evaluated in one fused loop (`_set_covariance_cache_loop!`):

 1. The mixing length: under `AbstractEDMF`, `ᶜl_mix` is refreshed in the loop
    (with `ᶜbuoygrad` first when `recompute_buoygrad`); otherwise the grid-mean
    mixing length is passed in.
 2. The displacement length `l_d` (`displacement_length`) of the buoyancy-bounded
    mixing length (`buoyancy_bounded_mixing_length`; `l_mix` without TKE) and the
    floor displacement scale (`floor_displacement_length`), both with the
    face-maximum dry buoyancy gradient `N²_disp` of the two bounding faces.
 3. The θ→T Jacobian `∂T/∂θ_li` (`∂T_∂θ_li`), from the grid-mean variables,
    consistent with the gradients.
 4. θ′θ′, q′q′ and, for the diagnosed correlation, θ′q′ from the displacement,
    transformed to T. The caller fills the horizontal gradient invariants of
    `q_tot`, `θ_li` and their cross term (`hgrad_invariant!`,
    `hgrad_cross_invariant!`), which are zero when
    `sgs_variance_horizontal_scale_factor` is 0.
 5. With `sgs_variance_curvature`, the displacement coefficients `ᶜdisp_S`
    (`displacement_curvature_coefficients`) from `ℓ_v = √(2 c_σ) l_d`, the Van Leer
    centre gradients, and the second derivatives `(∂ψ∂z⁺ − ∂ψ∂z⁻) / Δz` of the two
    bounding face gradients `ᶠ∂qt∂z` and `ᶠ∂θli∂z`. At the domain boundaries the
    adjacent interior face gradient stands in for the boundary face, so the
    curvature is zero in the bottom and top cells. `lin` uses the
    harmonic-mean-limited centre gradient that the covariance uses, so `lin²`
    stays the vertical part of `σ_S²`; the limiter zeros it in a cell whose two
    face gradients differ in sign, where `curv` is at its largest and the whole
    vertical effect is the bound of the quadratic.
 6. The closure-validity bound `σ_q ≤ sgs_variance_max_rel_std * q_tot`. For the
    diagnosed correlation, the covariance is rescaled with the bound first, so the
    correlation is invariant under it.
 7. The T–q correlation `ᶜcorr_Tq` that the quadrature samples: the prescribed
    `Tq_correlation_coefficient` for `ConstantTqCorrelation`, and
    `T′q′ / √(T′T′ q′q′)` (`sgs_correlation_Tq`) for `DiagnosedTqCorrelation`,
    with the prescribed value where a variance vanishes relative to its grid mean.
 8. The variance fidelity `ᶜα` (`sgs_variance_fidelity`), at the state of the
    domain that carries the SGS closure, as in
    `set_sgs_moments_and_cloud_fraction!`: the environment for PrognosticEDMFX,
    the grid mean otherwise. `σ_S` is the Gaussian-linear estimate
    `σ_S² = σ_q² − 2γ Cov(T,q) + γ² σ_T²` with `γ = ∂q_vap_sat_∂T` at that state
    and the grid-mean variances and correlation; the downstream quadrature
    closures compute their own `σ_S`. The condensate relaxes toward the excess
    `q_tot − q_rai − q_sno − q_sat`, since rain and snow are not in equilibrium
    with vapor. The relaxation time of the Damköhler factor is
    `sgs_microphysics_relaxation_time`, from the 1M process rates under the 1M
    scheme. Without TKE, `TKE = 0` and `l_mix = 1 m`, so the Damköhler factor is
    1; under 0M microphysics, `ᶜα = 1 / A`.

Scratch: the caller fills `ᶜtemp_scalar_5`, `temp_data`, and `ᶜtemp_scalar_6`
with the moisture, potential-temperature, and cross invariants. For `ᶜα`,
`ᶜtemp_scalar`, `ᶜtemp_scalar_2`, and `ᶜtemp_scalar_3` hold the environment
density and cloud condensates, and `ᶜtemp_C3` the precipitation, when they are
not fields already and `env_state` does not provide them.
"""
function set_covariance_cache!(
    Y,
    p,
    thermo_params;
    recompute_buoygrad::Bool = false,
    env_state = nothing,
)
    uses_covariances(p.atmos) || return nothing

    FT = eltype(p.params)
    diag_corr = p.atmos.tq_correlation_model isa DiagnosedTqCorrelation
    has_curvature = p.atmos.sgs_variance_curvature
    (;
        ᶜT′T′,
        ᶜq′q′,
        ᶜcorr_Tq,
        ᶜgradᵥ_q_tot,
        ᶜgradᵥ_θ_liq_ice,
        ᶠ∂θli∂z,
        ᶠ∂qt∂z,
        ᶜq_tot_nonneg,
        ᶜT,
        ᶜq_liq,
        ᶜq_ice,
        ᶜbuoygrad,
        ᶜbg_coeffs,
        ᶜcloud_fraction,
    ) = p.precomputed

    c_g = FT(CAP.sgs_variance_geometric_coeff(p.params))
    c_Δx = FT(CAP.sgs_variance_horizontal_scale_factor(p.params))
    r_Δx_θ = FT(CAP.sgs_variance_theta_horizontal_scale_ratio(p.params))
    r_v_θ = FT(CAP.sgs_variance_theta_vertical_scale_ratio(p.params))
    r_h_θq = FT(CAP.sgs_variance_horizontal_cross_correlation(p.params))
    Δx_h = FT(
        Spaces.node_horizontal_length_scale(
            Spaces.horizontal_space(axes(Y.c)),
        ),
    )
    ℓ_h_q² = c_g * (c_Δx * Δx_h)^2
    ℓ_h_θ² = r_Δx_θ^2 * ℓ_h_q²
    ℓ_h_θq = r_h_θq * r_Δx_θ * ℓ_h_q²
    c_σ = FT(CAP.diagnostic_covariance_coeff(p.params))

    if iszero(c_Δx)
        ᶜinv_q = ᶜinv_θ = ᶜinv_θq = zero(FT)
    else
        ᶜinv_q = p.scratch.ᶜtemp_scalar_5
        ᶜinv_θ = Fields.Field(p.scratch.temp_data, axes(Y.c))
        # Filled by the caller (see set_covariance_cache_and_cloud_fraction!)
        ᶜinv_θq = p.scratch.ᶜtemp_scalar_6
    end

    turbconv_model = p.atmos.turbconv_model
    is_edmf = turbconv_model isa AbstractEDMF
    if !is_edmf
        # Refreshes `ᶜbuoygrad` (`blended_N²`) from the current cloud fraction.
        ᶜl_mix = compute_gm_mixing_length(Y, p)
    else
        ᶜl_mix = p.precomputed.ᶜl_mix
    end

    turbconv_params = CAP.turbconv_params(p.params)
    c_b = FT(CAP.static_stab_coeff(turbconv_params))
    κ_b_per_tke = c_b / (2 * c_σ)
    w_min = floor_velocity(
        horizontal_filter_scale(axes(Y.c)),
        FT(CAP.sgs_variance_vertical_velocity_floor(p.params)),
        FT(CAP.sgs_variance_floor_reference_resolution(p.params)),
        FT(CAP.sgs_variance_floor_resolution_exponent(p.params)),
    )
    τ_0 = FT(CAP.sgs_variance_floor_time_scale(p.params))
    κ_min = w_min^2 / (2 * c_σ * c_b)
    c_b_κ_min = c_b * κ_min

    r_max = FT(CAP.sgs_variance_max_rel_std(p.params))
    corr_fallback = FT(correlation_Tq(p.params))
    ᶜlg = Fields.local_geometry_field(Y.c)
    ᶜΔz = Fields.Δz_field(axes(Y.c))
    z_sfc = Fields.level(Fields.coordinate_field(Y.f).z, Fields.half)

    has_tke = hasproperty(Y.c, :ρtke)
    is_dry_or_0m = p.atmos.microphysics_model isa Union{DryModel, EquilibriumMicrophysics0M}
    is_0m = p.atmos.microphysics_model isa EquilibriumMicrophysics0M
    has_α = hasproperty(p.precomputed, :ᶜα)
    A = FT(CAP.cloud_fraction_steepness_scale(p.params))
    τ_mp_params =
        p.atmos.microphysics_model isa NonEquilibriumMicrophysics1M ?
        CAP.cloud_liquid_formation_1m_params(p.params) :
        ConstantMicrophysicsTimescales(
            FT(CAP.condensation_evaporation_timescale(p.params)),
            FT(CAP.sublimation_deposition_timescale(p.params)),
        )
    w_eq = FT(CAP.sgs_fidelity_equilibrium_weight(p.params))
    q_min = FT(TD.Parameters.q_min(thermo_params))
    c_d = tke_dissipation_coefficient(turbconv_params)
    c_trunc = equilibrium_condensate_truncation(p)
    sf_params = CAP.surface_fluxes_params(p.params)
    vkc = CAP.von_karman_const(p.params)
    ᶜΔ_f = resolvability_filter_scale(axes(Y.c))
    scale_blending_method = is_edmf ? p.atmos.edmfx_model.scale_blending_method : nothing
    (; sfc_conditions, ᶜstrain_rate_norm) = p.precomputed
    ᶜα = has_α ? p.precomputed.ᶜα : ᶜcorr_Tq
    ᶜT′q′ = diag_corr ? p.precomputed.ᶜT′q′ : ᶜcorr_Tq
    ᶜdisp_S = has_curvature ? p.precomputed.ᶜdisp_S : ᶜcorr_Tq
    (zero_FT, one_FT, ϵ_FT, inf_FT) = (zero(FT), one(FT), eps(FT), FT(Inf))

    if has_α && !is_0m
        (; ᶜρ_env, ᶜT_mean, ᶜq_mean, ᶜq_lcl, ᶜq_icl, ᶜq_precip) =
            isnothing(env_state) ?
            cloud_env_state(Y, p, thermo_params; materialize = true) : env_state
        ᶜenv_pack = @. lazy(
            tuple(ᶜρ_env, ᶜT_mean, ᶜq_mean - ᶜq_precip, ᶜq_lcl + ᶜq_icl, ᶜq_lcl, ᶜq_icl),
        )
    else
        ᶜenv_pack = @. lazy(
            tuple(ᶜcorr_Tq, ᶜcorr_Tq, ᶜcorr_Tq, ᶜcorr_Tq, ᶜcorr_Tq, ᶜcorr_Tq),
        )
    end
    ᶜinv_pack = @. lazy(tuple(ᶜinv_q, ᶜinv_θq, ᶜinv_θ))

    _set_covariance_cache_loop!(
        Y, thermo_params,
        ᶜbuoygrad, ᶜl_mix, ᶜq′q′, ᶜT′T′, ᶜT′q′, ᶜcorr_Tq, ᶜα, ᶜdisp_S,
        ᶜbg_coeffs, ᶜcloud_fraction, ᶜgradᵥ_θ_liq_ice, ᶜgradᵥ_q_tot,
        ᶠ∂θli∂z, ᶠ∂qt∂z,
        ᶜT, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice,
        ᶜenv_pack, ᶜinv_pack,
        ᶜlg, ᶜΔz, z_sfc, ᶜΔ_f,
        sfc_conditions, ᶜstrain_rate_norm,
        c_σ, r_v_θ, ℓ_h_q², ℓ_h_θ², ℓ_h_θq,
        κ_b_per_tke, c_b_κ_min, τ_0, r_max, corr_fallback,
        A, τ_mp_params, w_eq, q_min, c_d, c_trunc, vkc,
        turbconv_params, sf_params, scale_blending_method,
        zero_FT, one_FT, ϵ_FT, inf_FT,
        Val(recompute_buoygrad),
        Val(is_edmf),
        Val(has_tke),
        Val(diag_corr),
        Val(has_curvature),
        Val(is_dry_or_0m),
        Val(has_α),
        Val(is_0m),
    )
    return nothing
end

function _set_covariance_cache_loop!(
    Y, thermo_params,
    ᶜbuoygrad, ᶜl_mix, ᶜq′q′, ᶜT′T′, ᶜT′q′, ᶜcorr_Tq, ᶜα, ᶜdisp_S,
    ᶜbg_coeffs, ᶜcloud_fraction, ᶜgradᵥ_θ_liq_ice, ᶜgradᵥ_q_tot,
    ᶠ∂θli∂z, ᶠ∂qt∂z,
    ᶜT, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice,
    ᶜenv_pack, ᶜinv_pack,
    ᶜlg, ᶜΔz, z_sfc, ᶜΔ_f,
    sfc_conditions, ᶜstrain_rate_norm,
    c_σ, r_v_θ, ℓ_h_q², ℓ_h_θ², ℓ_h_θq,
    κ_b_per_tke, c_b_κ_min, τ_0, r_max, corr_fallback,
    A, τ_mp_params, w_eq, q_min, c_d, c_trunc, vkc,
    turbconv_params, sf_params, scale_blending_method,
    zero_FT, one_FT, ϵ_FT, inf_FT,
    ::Val{recompute_buoygrad},
    ::Val{is_edmf},
    ::Val{has_tke},
    ::Val{diag_corr},
    ::Val{has_curvature},
    ::Val{is_dry_or_0m},
    ::Val{has_α},
    ::Val{is_0m},
) where {
    recompute_buoygrad,
    is_edmf,
    has_tke,
    diag_corr,
    has_curvature,
    is_dry_or_0m,
    has_α,
    is_0m,
}
    val_master = Val{:master}()
    DataLayouts.foreach_column(
        ᶜbuoygrad, ᶜl_mix, ᶜq′q′, ᶜT′T′, ᶜT′q′, ᶜcorr_Tq, ᶜα, ᶜdisp_S,
        Y.c, ᶜbg_coeffs, ᶜcloud_fraction, ᶜgradᵥ_θ_liq_ice, ᶜgradᵥ_q_tot,
        ᶠ∂θli∂z, ᶠ∂qt∂z,
        ᶜT, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice,
        ᶜenv_pack, ᶜinv_pack,
        ᶜlg, ᶜΔz, z_sfc, ᶜΔ_f,
        sfc_conditions, ᶜstrain_rate_norm,
    ) do ᶜbuoygrad, ᶜl_mix, ᶜq′q′, ᶜT′T′, ᶜT′q′, ᶜcorr_Tq, ᶜα, ᶜdisp_S,
    Yc, ᶜbg_coeffs, ᶜcloud_fraction, ᶜgradᵥ_θ_liq_ice, ᶜgradᵥ_q_tot,
    ᶠ∂θli∂z, ᶠ∂qt∂z,
    ᶜT, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice,
    ᶜenv_pack, ᶜinv_pack,
    ᶜlg, ᶜΔz, z_sfc, ᶜΔ_f,
    sfc_conditions, ᶜstrain_rate_norm
        ᶜz = ᶜlg.coordinates.z
        ᶜtke = has_tke ? (@. lazy(specific(Yc.ρtke, Yc.ρ))) : zero_FT
        sfc_tke =
            has_tke ?
            (@. lazy(specific(Fields.level(Yc.ρtke, 1), Fields.level(Yc.ρ, 1)))) :
            zero_FT
        if recompute_buoygrad && is_edmf
            @. ᶜbuoygrad = blended_N²(
                ᶜbg_coeffs,
                ᶜcloud_fraction,
                projected_vector_data(C3, ᶜgradᵥ_θ_liq_ice, ᶜlg),
                projected_vector_data(C3, ᶜgradᵥ_q_tot, ᶜlg),
            )
        end
        if is_edmf
            ᶜprandtl_nvec = @. lazy(
                turbulent_prandtl_number(turbconv_params, ᶜbuoygrad, ᶜstrain_rate_norm),
            )
            ᶜml = @. lazy(
                mixing_length_lopez_gomez_2020(
                    turbconv_params,
                    sf_params,
                    vkc,
                    sfc_conditions.ustar,
                    ᶜz,
                    z_sfc,
                    ᶜΔ_f,
                    sfc_tke,
                    ᶜbuoygrad,
                    ᶜtke,
                    sfc_conditions.obukhov_length,
                    ᶜstrain_rate_norm,
                    ᶜprandtl_nvec,
                    scale_blending_method,
                ),
            )
            @. ᶜl_mix = get_mixing_length_field(ᶜml, val_master)
        end

        nc = Spaces.nlevels(axes(ᶜbuoygrad))
        ᶜθ_below = Operators.BottomBiasedF2C(
            bottom = Operators.SetValue(Fields.level(ᶠ∂θli∂z, 1 + Fields.half)),
        )
        ᶜθ_above = Operators.TopBiasedF2C(
            top = Operators.SetValue(Fields.level(ᶠ∂θli∂z, nc - Fields.half)),
        )
        ᶜq_below = Operators.BottomBiasedF2C(
            bottom = Operators.SetValue(Fields.level(ᶠ∂qt∂z, 1 + Fields.half)),
        )
        ᶜq_above = Operators.TopBiasedF2C(
            top = Operators.SetValue(Fields.level(ᶠ∂qt∂z, nc - Fields.half)),
        )
        ᶜN²_disp = @. max(
            dry_N²(ᶜbg_coeffs, ᶜθ_below(ᶠ∂θli∂z), ᶜq_below(ᶠ∂qt∂z)),
            dry_N²(ᶜbg_coeffs, ᶜθ_above(ᶠ∂θli∂z), ᶜq_above(ᶠ∂qt∂z)),
        )

        if has_tke
            ᶜtke_pos = @. lazy(max(ᶜtke, zero_FT))
            ᶜl_mix_disp = @. buoyancy_bounded_mixing_length(
                ᶜl_mix,
                κ_b_per_tke * ᶜtke_pos,
                ᶜN²_disp,
            )
            ᶜτ_turb = @. lazy(
                turbulent_lagrangian_time(ᶜl_mix_disp, κ_b_per_tke * ᶜtke_pos),
            )
            ᶜl_d = @. displacement_length(
                ᶜl_mix_disp,
                floor_displacement_length(
                    c_b_κ_min, ᶜN²_disp, ᶜτ_turb, τ_0, ᶜz - z_sfc, ᶜΔz,
                ),
                ᶜz - z_sfc,
            )
        else
            ᶜl_mix_disp = one_FT
            ᶜl_d = @. displacement_length(
                ᶜl_mix,
                floor_displacement_length(
                    c_b_κ_min, ᶜN²_disp, inf_FT, τ_0, ᶜz - z_sfc, ᶜΔz,
                ),
                ᶜz - z_sfc,
            )
        end

        ᶜℓ_v_q² = @. lazy(2 * c_σ * ᶜl_d^2)
        ᶜℓ_v_θ² = @. lazy(r_v_θ^2 * ᶜℓ_v_q²)
        ᶜℓ_v_θq = @. lazy(r_v_θ * ᶜℓ_v_q²)
        ᶜ∂q∂z = @. lazy(projected_vector_data(C3, ᶜgradᵥ_q_tot, ᶜlg))
        ᶜ∂θ∂z = @. lazy(projected_vector_data(C3, ᶜgradᵥ_θ_liq_ice, ᶜlg))

        if is_dry_or_0m
            ᶜq_l_jac = ᶜq_liq
            ᶜq_i_jac = ᶜq_ice
            ᶜq_t_jac = ᶜq_tot_nonneg
        else
            ᶜq_l_jac = @. lazy(specific(Yc.ρq_lcl, Yc.ρ))
            ᶜq_i_jac = @. lazy(specific(Yc.ρq_icl, Yc.ρ))
            ᶜq_t_jac = @. lazy(specific(Yc.ρq_tot, Yc.ρ))
        end
        ᶜθ_li_jac = @. lazy(
            TD.liquid_ice_pottemp(thermo_params, ᶜT, Yc.ρ, ᶜq_t_jac, ᶜq_l_jac, ᶜq_i_jac),
        )
        ᶜ∂T_∂θ = @. ∂T_∂θ_li(
            thermo_params, ᶜT, ᶜθ_li_jac, ᶜq_l_jac, ᶜq_i_jac, ᶜq_t_jac, Yc.ρ,
        )

        @. ᶜq′q′ = displacement_covariance(ℓ_h_q², ᶜℓ_v_q², _1(ᶜinv_pack), ᶜ∂q∂z, ᶜ∂q∂z)
        @. ᶜT′T′ =
            ᶜ∂T_∂θ^2 *
            displacement_covariance(ℓ_h_θ², ᶜℓ_v_θ², _3(ᶜinv_pack), ᶜ∂θ∂z, ᶜ∂θ∂z)
        if diag_corr
            @. ᶜT′q′ =
                ᶜ∂T_∂θ *
                displacement_covariance(ℓ_h_θq, ᶜℓ_v_θq, _2(ᶜinv_pack), ᶜ∂θ∂z, ᶜ∂q∂z)
            @. ᶜT′q′ *= sqrt(min(
                one_FT,
                (r_max * ᶜq_tot_nonneg)^2 / max(ᶜq′q′, ϵ_FT^2),
            ))
        end
        if has_curvature
            ᶜχ = @. lazy(r_v_θ * TD.∂q_vap_sat_∂T(thermo_params, ᶜT, Yc.ρ) * ᶜ∂T_∂θ)
            @. ᶜdisp_S = displacement_curvature_coefficients(
                sqrt(2 * c_σ) * ᶜl_d,
                ᶜχ,
                ᶜ∂q∂z,
                ᶜ∂θ∂z,
                (ᶜq_above(ᶠ∂qt∂z) - ᶜq_below(ᶠ∂qt∂z)) / ᶜΔz,
                (ᶜθ_above(ᶠ∂θli∂z) - ᶜθ_below(ᶠ∂θli∂z)) / ᶜΔz,
            )
        end
        @. ᶜq′q′ = min(ᶜq′q′, (r_max * ᶜq_tot_nonneg)^2)

        if diag_corr
            @. ᶜcorr_Tq = sgs_correlation_Tq(
                ᶜT′T′, ᶜq′q′, ᶜT′q′, ᶜT, ᶜq_tot_nonneg, corr_fallback,
            )
        else
            @. ᶜcorr_Tq = corr_fallback
        end

        if has_α
            if is_0m
                @. ᶜα = one_FT / A
            else
                ᶜγ = @. lazy(TD.∂q_vap_sat_∂T(thermo_params, _2(ᶜenv_pack), _1(ᶜenv_pack)))
                ᶜsigma_S = @. lazy(
                    sqrt(
                        max(
                            ᶜq′q′ - 2 * ᶜγ * ᶜcorr_Tq * sqrt(max(ᶜT′T′ * ᶜq′q′, zero_FT)) +
                            ᶜγ^2 * ᶜT′T′,
                            zero_FT,
                        ),
                    ),
                )
                ᶜmu_S = @. lazy(
                    _3(ᶜenv_pack) -
                    TD.q_vap_saturation(thermo_params, _2(ᶜenv_pack), _1(ᶜenv_pack)),
                )
                @. ᶜα = sgs_variance_fidelity(
                    A,
                    sgs_microphysics_relaxation_time(
                        τ_mp_params,
                        thermo_params,
                        _2(ᶜenv_pack),
                        _1(ᶜenv_pack),
                        _3(ᶜenv_pack),
                        _5(ᶜenv_pack),
                        _6(ᶜenv_pack),
                    ),
                    c_d,
                    has_tke ? max(ᶜtke, zero_FT) : zero_FT,
                    has_tke && is_edmf ? ᶜl_mix_disp : one_FT,
                    w_eq,
                    _4(ᶜenv_pack),
                    ᶜmu_S,
                    ᶜsigma_S,
                    q_min,
                    has_curvature ? _disp_lin(ᶜdisp_S) : zero_FT,
                    has_curvature ? _disp_curv(ᶜdisp_S) : zero_FT,
                    c_trunc,
                )
            end
        end
        return nothing
    end
    return nothing
end

# ============================================================================
# SGS Moments — linearized saturation-excess moments
# ============================================================================
#
# The saturation variable is the linear excess S = q_tot − q_sat, linearized
# about the subdomain mean (T_mean, ρ) with the Clausius–Clapeyron slope
# γ = ∂q_sat/∂T:
#
#     μ_S = q_tot_mean − q_sat(T_mean, ρ),
#     σ_S = sqrt(σ_q² − 2 γ ρ_{Tq} σ_T σ_q + γ² σ_T²).
#
# They drive the truncated-Gaussian cloud fraction, the Lagrange multipliers of
# the 1M condensate closures and the conditional-moment node rule.

"""
    _sgs_saturation_moments(thp, ρ, T_mean, q_tot_mean,
                            sgs_quad, T′T′, q′q′, corr_Tq)

μ_S and σ_S of the linear saturation excess S = q_tot − q_sat, from the
subdomain mean state and the SGS (co)variances (`_sgs_linear_sigma_S`).
Without SGS sampling (`nothing`, `GridMeanSGS`) all mass sits at the mean and
`σ_S = ϵ_numerics(FT)`; σ_S is floored there so the normalised closure
`C = q_c / (α·σ_S)` stays well-conditioned.

Returns `(; mu_S, sigma_S)`.
"""
@inline function _sgs_saturation_moments(
    thp, ρ, T_mean, q_tot_mean,
    sgs_quad, T′T′, q′q′, corr_Tq,
)
    FT = typeof(ρ)
    mu_S = q_tot_mean - TD.q_vap_saturation(_unwrapped(thp), T_mean, ρ)
    sigma_S =
        not_quadrature(sgs_quad) ? ϵ_numerics(FT) :
        _sgs_linear_sigma_S(thp, ρ, T_mean, T′T′, q′q′, corr_Tq)
    return (; mu_S, sigma_S)
end

"""
    _sgs_linear_sigma_S(thp, ρ, T, T′T′, q′q′, corr_Tq)

Linearized Clausius–Clapeyron saturation-excess standard deviation
`σ_S = sqrt(max(σ_q² − 2 γ corr_Tq σ_T σ_q + γ² σ_T², 0))` with
`γ = ∂q_sat/∂T(T, ρ)`, floored at `ϵ_numerics(FT)`.
"""
@inline function _sgs_linear_sigma_S(thp, ρ, T, T′T′, q′q′, corr_Tq)
    FT = typeof(ρ)
    (σ_q, σ_T, corr) = sgs_stddevs_and_correlation(q′q′, T′T′, corr_Tq)
    γ = FT(TD.∂q_vap_sat_∂T(_unwrapped(thp), T, ρ))
    var_S = σ_q^2 - 2 * γ * corr * σ_T * σ_q + γ^2 * σ_T^2
    return max(sqrt(max(var_S, zero(FT))), ϵ_numerics(FT))
end

# ============================================================================
# Cloud Fraction: truncated-Gaussian closure with Lagrange-multiplier inversion
# ============================================================================
#
# **Physical model**: the subgrid saturation excess S = q_tot − q_sat is
# assumed Gaussian: S ~ N(μ_S, σ_S²), with μ_S set analytically to the
# linearized mean μ_S = q_tot_mean − q_sat(T_mean, ρ) (exact under the
# same linearization that justifies Gaussianity of S) and σ_S the linearized
# width, σ_S² = σ_q² − 2 γ corr_Tq σ_T σ_q + γ² σ_T² with γ = ∂q_sat/∂T
# (`_sgs_linear_sigma_S`).
# Working with the centred excess S′ = S − μ_S ~ N(0, σ_S²), we seek a
# Lagrange multiplier λ that enforces mass conservation:
#
#     E[max(0, λ + α·S′)] = q_c,                                     (*)
#
# where α = `sgs_variance_fidelity` controls how much of the SGS variance is
# propagated into the local condensate (α = α_Da·α_eq ∈ (0, 1]; the
# equilibrium multiplier λ_eq has α = 1 and is the mean cloud excess). The effective
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
# `λ_lagrange` is the solution of this continuous constraint
# (`_continuous_lagrange`): the conditional-moment node rule of the 1M
# microphysics integrates the kinked integrand max(0, ·) exactly, so the
# continuous multiplier conserves the condensate at its nodes.
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
# inconsistent because `λ` is solved with `σ_S_eff`, not `σ_aug`).
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
liquid-fraction ramp. A phase of weight zero drops out even where its time is
infinite (a disabled process), so the blend stays finite at either end of the
ramp. Used by `sgs_variance_fidelity`.
"""
@inline function sgs_microphysics_timescale(thp, T, τ_cond, τ_sub)
    FT = typeof(T)
    λ_eq = FT(TD.liquid_fraction_ramp(thp, T))
    τ_l, τ_i = FT(τ_cond), FT(τ_sub)
    return ifelse(
        λ_eq >= one(FT),
        τ_l,
        ifelse(λ_eq <= zero(FT), τ_i, λ_eq * τ_l + (one(FT) - λ_eq) * τ_i),
    )
end

"""
    ConstantMicrophysicsTimescales(τ_cond, τ_sub)

The constant liquid and ice relaxation times (`condensation_evaporation_timescale`,
`sublimation_deposition_timescale`) of `sgs_microphysics_relaxation_time` for the
schemes other than 1M. It broadcasts as a scalar, so it can be passed inside `@.`.
"""
struct ConstantMicrophysicsTimescales{FT}
    τ_cond::FT
    τ_sub::FT
end
Base.broadcastable(τ::ConstantMicrophysicsTimescales) = tuple(τ)

"""
    sgs_microphysics_relaxation_time(τ_params, thp, T, ρ, q_cloud, q_lcl, q_icl)

The phase-blended relaxation time `sgs_microphysics_timescale` of the Damköhler
factor of `sgs_variance_fidelity`, at a point. With the 1M scheme (`τ_params` its
`CAP.cloud_liquid_formation_1m_params`), the liquid and ice times are those of its
condensation–evaporation and deposition–sublimation processes at the state
`(T, ρ, q_cloud, q_lcl, q_icl)` (`cloud_species_relaxation_time`, with
`q_cloud = q_tot − q_rai − q_sno` the cloud-available water), so the factor
follows the configured ice-formation variant; with the other schemes
(`τ_params` a `ConstantMicrophysicsTimescales`), they are the constant parameters
`condensation_evaporation_timescale` and `sublimation_deposition_timescale`,
which are the relaxation times of the 2M scheme. Evaluated inside the fused
column loop of `set_covariance_cache!`.
"""
@inline sgs_microphysics_relaxation_time(
    τ::ConstantMicrophysicsTimescales,
    thp,
    T,
    ρ,
    q_cloud,
    q_lcl,
    q_icl,
) = sgs_microphysics_timescale(thp, T, τ.τ_cond, τ.τ_sub)
@inline sgs_microphysics_relaxation_time(cmp, thp, T, ρ, q_cloud, q_lcl, q_icl) =
    sgs_microphysics_timescale(
        thp,
        T,
        cloud_species_relaxation_time(
            cmp, thp, T, ρ, q_cloud, q_lcl, q_icl, TD.Liquid(),
        ),
        cloud_species_relaxation_time(cmp, thp, T, ρ, q_cloud, q_lcl, q_icl, TD.Ice()),
    )

"""
    truncated_gaussian_condensate(mu_S, sigma_S)

Expected saturation excess `E[max(0, S)] = μ_S Φ(z) + σ_S φ(z)` with
`z = μ_S / σ_S` for a Gaussian saturation excess `S ~ N(μ_S, σ_S²)`.
"""
@inline function truncated_gaussian_condensate(mu_S, sigma_S)
    FT = typeof(mu_S)
    inv_sqrt2π = one(FT) / sqrt(FT(2) * FT(π))
    σ = max(FT(sigma_S), ϵ_numerics(FT))
    z = mu_S / σ
    return max(zero(FT), mu_S * normal_cdf(z) + σ * exp(-z^2 / 2) * inv_sqrt2π)
end

"""
    truncated_displacement_scale(c)

Factor `s = (1 − 2c φ(c) / (2Φ(c) − 1))^(−1/2) ≥ 1` by which a standard normal
truncated to `[−c, c]` is stretched to keep unit variance; `1` for `c = Inf`.
"""
@inline function truncated_displacement_scale(c)
    FT = typeof(c)
    inv_sqrt2π = one(FT) / sqrt(FT(2) * FT(π))
    # Both arms of the `ifelse` are evaluated, so `c = Inf` is replaced before
    # the finite-`c` arm, where `c φ(c)` would be `Inf * 0`.
    c_finite = ifelse(isfinite(c), c, one(FT))
    φc = exp(-c_finite^2 / 2) * inv_sqrt2π
    Z = one(FT) - 2 * normal_cdf(-c_finite)
    return ifelse(isfinite(c), sqrt(Z / (Z - 2 * c_finite * φc)), one(FT))
end

"""
    truncated_displacement_condensate(mu_S, sigma_S, c)

Expected saturation excess `E[max(0, S)]` for `S = μ_S + s σ_S x`, with the
displacement `x` standard normal truncated to `[−c, c]` and `s`
(`truncated_displacement_scale`) keeping the variance `σ_S²` of the covariance
closure. With `Z = Φ(c) − Φ(−c)` and `a = max(−c, −μ_S / (s σ_S))`,

    E[max(0, S)] = (μ_S (Φ(c) − Φ(a)) + s σ_S (φ(a) − φ(c))) / Z,

which is zero for `μ_S ≤ −s c σ_S`, where no displaced parcel reaches
saturation, and `μ_S` for `μ_S ≥ s c σ_S`. The displacement acts on the
linearized profiles, so the truncation bounds how far it extrapolates the
gradients. `c = Inf` returns `truncated_gaussian_condensate(μ_S, σ_S)` exactly.
The factors that depend on `c` alone keep the parameter's type.
"""
@inline function truncated_displacement_condensate(mu_S, sigma_S, c)
    FT = typeof(mu_S)
    gaussian = truncated_gaussian_condensate(mu_S, sigma_S)
    inv_sqrt2π = one(FT) / sqrt(FT(2) * FT(π))
    c_finite = ifelse(isfinite(c), c, one(c))
    Z = one(c) - 2 * normal_cdf(-c_finite)
    σ = max(sigma_S, ϵ_numerics(FT)) * truncated_displacement_scale(c)
    a = clamp(-mu_S / σ, -c_finite, c_finite)
    # Upper-tail differences, accurate where `a` approaches `c`. Each difference
    # takes the same expression at both ends, so it vanishes exactly where `a`
    # is clamped to `c`.
    I₀ = normal_cdf(-a) - normal_cdf(-c_finite)
    I₁ = (exp(-a^2 / 2) - exp(-c_finite^2 / 2)) * inv_sqrt2π
    truncated = max(zero(FT), (mu_S * I₀ + σ * I₁) / Z)
    return ifelse(isfinite(c), truncated, gaussian)
end

"""
    equilibrium_condensate(mu_S, sigma_S, lin, curv, c)

Equilibrium condensate the SGS closures relax toward: the truncated-displacement
expectation `truncated_displacement_condensate(μ_S, σ_S, c)`, or, with nonzero
curvature `curv` (`sgs_variance_curvature`), the curvature-aware
`displacement_condensate(μ_S, σ_S, lin, curv)`, whose tail is bounded by the
quadratic where the curvature is negative. With `sgs_variance_curvature`, the
callers pass `c = Inf` (`equilibrium_condensate_truncation`), so a cell with
`curv = 0`, such as the bottom and top cells, has the Gaussian expectation, the
limit of the curvature-aware one as `curv → 0`.
"""
@inline equilibrium_condensate(mu_S, sigma_S, lin, curv, c) = ifelse(
    iszero(curv),
    truncated_displacement_condensate(mu_S, sigma_S, c),
    displacement_condensate(mu_S, sigma_S, lin, curv),
)

"""
    equilibrium_condensate_truncation(p)

Displacement truncation `c` passed to `equilibrium_condensate`: the parameter
`sgs_displacement_truncation`, or `Inf` with `sgs_variance_curvature`, whose
curvature-aware closure bounds the moist tail itself.
"""
function equilibrium_condensate_truncation(p)
    c = CAP.sgs_displacement_truncation(p.params)
    return p.atmos.sgs_variance_curvature ? oftype(c, Inf) : c
end

"""
    _gaussian_interval_moments(a, b)

Moments `(I₀, I₁, I₂) = ∫ₐᵇ (1, x, x²) φ(x) dx` of the standard normal
density over `[a, b]`, `a ≤ b`, with `I₁ = φ(a) − φ(b)` and
`I₂ = I₀ + a φ(a) − b φ(b)`; the products `x φ(x)` are taken as zero at
infinite limits.
"""
@inline function _gaussian_interval_moments(a, b)
    FT = typeof(a)
    inv_sqrt2π = one(FT) / sqrt(FT(2) * FT(π))
    φa = ifelse(isfinite(a), exp(-a^2 / 2) * inv_sqrt2π, zero(FT))
    φb = ifelse(isfinite(b), exp(-b^2 / 2) * inv_sqrt2π, zero(FT))
    I₀ = normal_cdf(b) - normal_cdf(a)
    I₁ = φa - φb
    I₂ = I₀ + ifelse(isfinite(a), a * φa, zero(FT)) -
         ifelse(isfinite(b), b * φb, zero(FT))
    return (I₀, I₁, I₂)
end

"""
    quadratic_condensate(mu_S, lin, curv)

Expected positive part `E[max(0, S)]` of the quadratic displacement model
without horizontal part, `S = μ_S + lin x + ½ curv (x² − 1)` with `x`
standard normal, in closed form. `S` is positive between the roots of the
quadratic for `curv < 0`, which bounds `S` by
`μ_S + (lin² + curv²)/(2|curv|)`, and outside them for `curv > 0`; a
curvature below `√ε (|lin| + |μ_S|)` is taken as zero, where the expectation
is `truncated_gaussian_condensate(μ_S, |lin|)`.
"""
@inline function quadratic_condensate(mu_S, lin, curv)
    FT = typeof(mu_S)
    l = FT(lin)
    c = FT(curv)
    small = abs(c) <= sqrt(eps(FT)) * (abs(l) + abs(mu_S))
    linear = truncated_gaussian_condensate(mu_S, abs(l))
    # Roots of ½ c x² + l x + m₀, with the stable form that keeps the root
    # that stays finite as c → 0 free of cancellation.
    m₀ = mu_S - c / 2
    D = l^2 - 2 * c * m₀
    s = sqrt(max(D, zero(FT)))
    r = -(l + copysign(s, l)) / 2
    r_safe = ifelse(r == zero(FT), one(FT), r)
    c_safe = ifelse(c == zero(FT), one(FT), c)
    x_lo = min(2 * r_safe / c_safe, m₀ / r_safe)
    x_hi = max(2 * r_safe / c_safe, m₀ / r_safe)
    (I₀, I₁, I₂) = _gaussian_interval_moments(x_lo, x_hi)
    inner = c / 2 * I₂ + l * I₁ + m₀ * I₀
    # Without real roots, the quadratic keeps one sign: nonnegative
    # everywhere for c > 0, so the expectation is the mean, and nonpositive
    # for c < 0.
    no_roots = ifelse(c > zero(FT), max(zero(FT), mu_S), zero(FT))
    quadratic = ifelse(
        D <= zero(FT),
        no_roots,
        max(zero(FT), ifelse(c < zero(FT), inner, mu_S - inner)),
    )
    return ifelse(small, linear, quadratic)
end

"""
    _gauss_hermite_10(FT)

Positive nodes `x_k` and weights `w_k`, `k = 1:5`, of the probabilists'
Gauss–Hermite rule of order 10, `∫ g(x) φ(x) dx ≈ Σ_k w_k (g(x_k) + g(−x_k))`.
"""
@inline function _gauss_hermite_10(::Type{FT}) where {FT}
    nodes = (
        FT(0.48493570751549964),
        FT(1.4659890943911615),
        FT(2.484325841638953),
        FT(3.5818234835519265),
        FT(4.859462828332313),
    )
    weights = (
        FT(0.34464233493201735),
        FT(0.13548370298026768),
        FT(0.01911158050077044),
        FT(0.0007580709343122166),
        FT(4.3106526307182725e-6),
    )
    return (nodes, weights)
end

"""
    _gauss_hermite_10_sum(g, FT)

`Σ_k w_k (g(x_k) + g(−x_k))` over the rule of `_gauss_hermite_10`, an
estimate of `E[g(x)]` for standard normal `x`.
"""
@inline function _gauss_hermite_10_sum(g::G, ::Type{FT}) where {G, FT}
    (nodes, weights) = _gauss_hermite_10(FT)
    return sum(ntuple(k -> weights[k] * (g(nodes[k]) + g(-nodes[k])), Val(5)))
end

"""
    _smoothing_bump(m, σ_h)

`truncated_gaussian_condensate(m, σ_h) − max(0, m)`: the change a Gaussian
part of width `σ_h` makes to the positive part of a mean excess `m`, a bump
of height `σ_h/√(2π)` around `m = 0` that decays like a Gaussian in `m/σ_h`.
"""
@inline _smoothing_bump(m, σ_h) =
    truncated_gaussian_condensate(m, σ_h) - max(zero(m), m)

"""
    VerticalCurvatureChange(mu_S, σ_h, lin, curv)

Functor `y ↦ Q(μ_S + σ_h y, lin, curv) − Q(μ_S + σ_h y, lin, 0)`, with `Q`
`quadratic_condensate`: the change the curvature makes to the closed-form
vertical model at the mean excess shifted by the horizontal part `σ_h y`.
"""
struct VerticalCurvatureChange{FT}
    mu_S::FT
    σ_h::FT
    lin::FT
    curv::FT
end
@inline function (f::VerticalCurvatureChange)(y)
    m = f.mu_S + f.σ_h * y
    return quadratic_condensate(m, f.lin, f.curv) -
           quadratic_condensate(m, f.lin, zero(m))
end

"""
    SmoothingBumpChange(mu_S, σ_h, lin, curv)

Functor `x ↦ b(μ_S + lin x + ½ curv (x² − 1)) − b(μ_S + lin x)`, with `b`
`_smoothing_bump` at width `σ_h`: the change the curvature makes to the
horizontal smoothing of the vertical model at the displacement `x`.
"""
struct SmoothingBumpChange{FT}
    mu_S::FT
    σ_h::FT
    lin::FT
    curv::FT
end
@inline function (f::SmoothingBumpChange)(x)
    m_lin = f.mu_S + f.lin * x
    m_curv = m_lin + f.curv * (x^2 - 1) / 2
    return _smoothing_bump(m_curv, f.σ_h) - _smoothing_bump(m_lin, f.σ_h)
end

"""
    _curvature_correction(mu_S, lin, curv, σ_h)

Change `E[max(0, S_curv)] − E[max(0, S_lin)]` of the expected positive part
between the quadratic `S_curv = μ_S + lin x + ½ curv (x² − 1) + h` and the
linear `S_lin = μ_S + lin x + h`, `x ~ N(0, 1)`, `h ~ N(0, σ_h²)`, by a
10-point Gauss–Hermite rule in whichever of the two variables has the
narrower spread, so that the integrand it samples is smooth on the scale of
the rule:

  - vertical part dominant (`σ_h ≤ |lin|`): the rule runs over `h`, and the
    `x` integral is the closed form `quadratic_condensate(μ_S + h, lin, curv)`
    (`VerticalCurvatureChange`);
  - horizontal part dominant (`σ_h ≥ 3 |lin|`): the rule runs over `x`, on the
    change of the smoothing `_smoothing_bump` that `h` adds to the closed-form
    vertical model (`SmoothingBumpChange`);

with a linear blend of the two between. Both estimates vanish identically
for `curv = 0`, and the first is exact for `σ_h = 0`. Against brute-force
integration over means from `−2.5 σ_S` to `σ_S` and `|curv|` up to `3 σ_S`,
the error of the expectation is below 0.1 % in the median and below 3 % for
nine states in ten; the largest, about 25 %, lie in the subsaturated tail
where the linear vertical part vanishes and the curvature term holds half of
the variance.
"""
@inline function _curvature_correction(mu_S, lin, curv, σ_h)
    FT = typeof(mu_S)
    ratio = σ_h / max(abs(lin), ϵ_numerics(FT))
    w_x = clamp((ratio - 1) / 2, zero(FT), one(FT))
    # Each branch skips the ten closed-form evaluations of a rule whose
    # weight is zero.
    over_h = if w_x < one(FT)
        _gauss_hermite_10_sum(VerticalCurvatureChange(mu_S, σ_h, lin, curv), FT)
    else
        zero(FT)
    end
    over_x = if w_x > zero(FT)
        quadratic_condensate(mu_S, lin, curv) -
        quadratic_condensate(mu_S, lin, zero(FT)) +
        _gauss_hermite_10_sum(SmoothingBumpChange(mu_S, σ_h, lin, curv), FT)
    else
        zero(FT)
    end
    return w_x * over_x + (one(FT) - w_x) * over_h
end

"""
    displacement_condensate(mu_S, sigma_S, lin, curv)

Expected saturation excess `E[max(0, S)]` under the vertical displacement
acting on the curved profile of the saturation excess,

    S = μ_S + lin x + ½ curv (x² − 1) + h,   x ~ N(0, 1),  h ~ N(0, σ_h²),

with `(lin, curv)` from `displacement_curvature_coefficients` and the
horizontal part `σ_h² = σ_S² − lin² − curv²/2`, so that the variance of `S`
is the `σ_S²` of the covariance closure, which the cloud fraction and the
Lagrange multiplier of the actual condensate use. Where the vertical part
alone, `lin² + curv²/2`, exceeds `σ_S²` (the σ_q bound of
`set_covariance_cache!`), the displacement is shortened by the factor `s`
with `s² lin² + s⁴ curv²/2 = σ_S²`, which scales `lin` by `s` and `curv` by
`s²` and leaves no horizontal part.

The result is the truncated-Gaussian expectation of the linear model,
`truncated_gaussian_condensate(μ_S, √(lin² + σ_h²))`, plus the change the
curvature makes to it (`_curvature_correction`, with the vertical model in
closed form, `quadratic_condensate`), so `curv = 0` returns
`truncated_gaussian_condensate(μ_S, σ_S)` exactly, and for `curv < 0` with
`σ_h = 0` the moist tail is bounded by the maximum
`μ_S + (lin² + curv²)/(2|curv|)` of the quadratic, the largest saturation
excess the displaced profile reaches.
"""
@inline function displacement_condensate(mu_S, sigma_S, lin, curv)
    FT = typeof(mu_S)
    σ = max(FT(sigma_S), ϵ_numerics(FT))
    gaussian = truncated_gaussian_condensate(mu_S, σ)
    # The Gaussian expectation is the exact result without curvature, and the
    # early return skips the two Gauss–Hermite rules there.
    curv == zero(FT) && return gaussian
    v_lin = FT(lin)^2
    v_curv = FT(curv)^2 / 2
    # `s² = u` solves `v_curv u² + v_lin u = σ²`, in the form that stays
    # finite as either coefficient vanishes.
    u = 2 * σ^2 / (v_lin + sqrt(v_lin^2 + 4 * v_curv * σ^2))
    s² = min(one(FT), u)
    l = sqrt(s²) * FT(lin)
    c = s² * FT(curv)
    σ_h = sqrt(max(σ^2 - l^2 - c^2 / 2, zero(FT)))
    linear = truncated_gaussian_condensate(mu_S, sqrt(l^2 + σ_h^2))
    return max(zero(FT), linear + _curvature_correction(mu_S, l, c, σ_h))
end

"""
    sgs_variance_fidelity(
        A, τ_mp, c_d, tke, l_mix, w_eq, q_c_actual, mu_S, sigma_S, q_min,
        lin = 0, curv = 0, c = Inf32,
    )

Local SGS variance fidelity `α = α_Da · α_eq` for the condensate
reconstruction `E[max(0, λ + α·S′)] = q_c`.

  - Damköhler factor
    `α_Da = 1 / (1 + A · τ_mp · c_d · √TKE / max(l_mix, 1 m))`:
    quiescent (TKE → 0) → 1, also for an infinite `τ_mp` (a disabled
    process); well-mixed → 0.
  - Equilibrium-ratio modulator `α_eq = r / (w_eq + (1 − w_eq) r)` with
    `r = (q_c_actual + q_min) / (q_c_eq + q_min)` and `q_c_eq` the expected
    saturation excess under the SGS PDF (always ≥ 0 even for subsaturated
    means): `equilibrium_condensate(μ_S, σ_S, lin, curv, c)`, the expectation
    under the displacement truncated at `c` standard deviations, or, with
    nonzero displacement coefficients `lin`, `curv`, the curvature-aware
    `displacement_condensate(μ_S, σ_S, lin, curv)`, the same target the
    condensate relaxes toward.
"""
@inline function sgs_variance_fidelity(
    A, τ_mp, c_d, tke, l_mix, w_eq, q_c_actual, mu_S, sigma_S, q_min,
    lin = zero(mu_S), curv = zero(mu_S), c = Inf32,
)
    FT = typeof(tke)
    inv_τ_turb = FT(c_d) * sqrt(max(tke, zero(FT))) / max(l_mix, one(FT))
    # The Damköhler number is zero in quiescent air (`inv_τ_turb = 0`) even
    # where `τ_mp` is infinite (a disabled process), so `α_Da → 1` there
    # with no `Inf · 0`.
    Da = ifelse(
        inv_τ_turb > zero(FT),
        FT(A) * FT(τ_mp) * inv_τ_turb,
        zero(FT),
    )
    α_Da = one(FT) / (one(FT) + Da)
    q_c_eq =
        equilibrium_condensate(FT(mu_S), FT(sigma_S), FT(lin), FT(curv), c)
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
    φz0 = exp(-z0^2 / 2) * inv_sqrt2π
    z1 = (C - φz0) / Φz0_safe

    # 3. Second Newton step. Applied only where Φ(z₁) is above
    # the ϵ_numerics clamp; below that (z₁ ≲ −7 at Float32) the clamped
    # denominator corrupts the ratio φ(z₁)/Φ(z₁), and the result is
    # condensate- and cloud-free to machine precision either way.
    φz1 = exp(-z1^2 / 2) * inv_sqrt2π
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
    q_c <= zero(FT) && return normal_cdf(_compute_z(zero(FT)))
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
inlined call that obtains `σ_S` (`_sgs_saturation_moments`, the linearized
width) and evaluates the truncated-Gaussian approximate closure in one
broadcast kernel. Used by
`set_quadrature_cloud_fraction!` so the moments are never
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
    q_c = q_liq + q_ice
    FT = typeof(q_c)
    q_c <= zero(FT) && return normal_cdf(_compute_z(zero(FT)))
    moments = _sgs_saturation_moments(
        thermo_params, ρ, T, q_tot, sgs_quad, T′T′, q′q′, corr_Tq,
    )
    q_sat = TD.q_vap_saturation(_unwrapped(thermo_params), T, ρ, q_liq, q_ice)
    return _compute_cloud_fraction(
        q_c, moments.mu_S, moments.sigma_S, q_sat, α, floor,
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
    _compute_sgs_moments(
        thp, ρ, T, q_tot, q_c, sgs_quad, T′T′, q′q′, corr_Tq, α,
        q_tot_cloud = q_tot, lin = 0, curv = 0, c = Inf32,
    )

The SGS moments `(sigma_S, λ_lagrange, λ_eq)` of the 1M condensate closures:

  - `sigma_S`: the linearized width of the centred saturation excess
    (`_sgs_linear_sigma_S`), clipped at `ϵ_numerics(FT)`.
  - `λ_lagrange`: the Lagrange multiplier of the actual condensate
    `max(0, λ + α S′)`, solving the continuous truncated-Gaussian constraint
    `E[max(0, λ + α S′)] = q_c` (`_continuous_lagrange`, with
    `σ_S_eff = α σ_S` floored so it stays finite where `q_c` and `α` both
    vanish).
  - `λ_eq`: the multiplier of the equilibrium condensate `max(0, λ_eq + S′)`,
    the mean of the Gaussian whose expectation is the equilibrium condensate
    `q_c_eq = equilibrium_condensate(μ_{S,cloud}, σ_S, lin, curv, c)` of the
    mean cloud excess `μ_{S,cloud} = q_tot_cloud − q_sat(T, ρ)`
    (`q_tot_cloud = q_tot − q_rai − q_sno`): the expectation under the
    displacement truncated at `c` standard deviations
    (`truncated_displacement_condensate`), or, with nonzero displacement
    coefficients `(lin, curv)` (`displacement_curvature_coefficients`), the
    curvature-aware `displacement_condensate(μ_{S,cloud}, σ_S, lin, curv)`.
    For the untruncated Gaussian (`c = Inf`, `curv = 0`), `λ_eq` is
    `μ_{S,cloud}` itself; where the truncated `q_c_eq` vanishes,
    `λ_eq = −Inf`, so that no node holds equilibrium condensate. The nodes
    keep the Gaussian `S′`, so with truncation or curvature only the mean
    equilibrium condensate is matched; its edge `−λ_eq` differs from the
    support of the truncated distribution.

The conditional-moment node rule integrates both piecewise-linear condensates
exactly, so these continuous multipliers conserve the condensate at its nodes.

Without SGS sampling (`nothing`, `GridMeanSGS`), all mass sits at the mean:
`sigma_S = ϵ_numerics(FT)`, `λ_lagrange = q_c`, and `λ_eq = max(0, μ_{S,cloud})`
directly, matching the σ_S → 0 limit.
"""
@inline function _compute_sgs_moments(
    thp, ρ, T, q_tot, q_c,
    sgs_quad, T′T′, q′q′, corr_Tq, α,
    q_tot_cloud = q_tot,
    lin = zero(ρ),
    curv = zero(ρ),
    c = Inf32,
)
    FT = typeof(ρ)
    q_sat = TD.q_vap_saturation(_unwrapped(thp), T, ρ)
    mu_S_cloud = q_tot_cloud - q_sat
    not_quadrature(sgs_quad) &&
        return (;
            sigma_S = ϵ_numerics(FT),
            λ_lagrange = q_c,
            λ_eq = max(zero(FT), mu_S_cloud),
        )

    sigma_S = _sgs_linear_sigma_S(thp, ρ, T, T′T′, q′q′, corr_Tq)
    λ_lagrange = _continuous_lagrange(q_c, max(α * sigma_S, ϵ_numerics(FT)))
    # Equilibrium condensate `q_c_eq = E[max(0, S_cloud)]` under the
    # displacement truncated at `c` standard deviations, or the curvature-aware
    # displacement expectation where `curv ≠ 0`, and the mean of the Gaussian
    # with that expectation. The untruncated Gaussian's multiplier is the mean
    # cloud excess itself. Where the truncated `q_c_eq` vanishes,
    # `λ_eq = −Inf` puts the equilibrium cloud edge at infinity, which the
    # node rule gives zero weight. All are evaluated and `ifelse` selects one,
    # so the kernel has no branch on the curvature.
    q_c_eq = equilibrium_condensate(mu_S_cloud, sigma_S, lin, curv, c)
    λ_eq_fit = _continuous_lagrange(q_c_eq, sigma_S)
    λ_eq_trunc = ifelse(
        isinf(c),
        mu_S_cloud,
        ifelse(q_c_eq > zero(FT), λ_eq_fit, FT(-Inf)),
    )
    λ_eq = ifelse(iszero(curv), λ_eq_trunc, λ_eq_fit)
    return (; sigma_S, λ_lagrange, λ_eq)
end

"""
    cloud_env_state(Y, p, thermo_params; materialize)

State of the domain that carries the SGS cloud closure — the environment under
`PrognosticEDMFX`, the grid mean otherwise — as
`(; ᶜρ_env, ᶜT_mean, ᶜq_mean, ᶜq_lcl, ᶜq_icl, ᶜq_precip)` (see `_get_env_ρ_T_q`,
`_get_condensate_means`, and `_get_precip_mean`).

With `materialize`, the lazy fields are written to `ᶜtemp_scalar`,
`ᶜtemp_scalar_2`, `ᶜtemp_scalar_3`, and `ᶜtemp_C3`; otherwise they stay lazy.
Without EDMF, the grid-mean mixing length overwrites `ᶜtemp_scalar` and
`ᶜtemp_scalar_2` in every covariance update, so a state that must outlive one
update stays lazy.
"""
function cloud_env_state(Y, p, thermo_params; materialize)
    (; turbconv_model, microphysics_model) = p.atmos
    ᶜρ_env, ᶜT_mean, ᶜq_mean = _get_env_ρ_T_q(Y, p, thermo_params, turbconv_model)
    ᶜq_lcl, ᶜq_icl = _get_condensate_means(Y, p, turbconv_model, microphysics_model)
    ᶜq_precip = _get_precip_mean(Y, p, turbconv_model, microphysics_model)
    if materialize
        ᶜρ_env = _materialized(p.scratch.ᶜtemp_scalar, ᶜρ_env)
        ᶜq_lcl = _materialized(p.scratch.ᶜtemp_scalar_2, ᶜq_lcl)
        ᶜq_icl = _materialized(p.scratch.ᶜtemp_scalar_3, ᶜq_icl)
        ᶜq_precip = _materialized(
            Fields.Field(
                Fields.field_values(p.scratch.ᶜtemp_C3.components.data.:1),
                axes(Y.c),
            ),
            ᶜq_precip,
        )
    end
    return (; ᶜρ_env, ᶜT_mean, ᶜq_mean, ᶜq_lcl, ᶜq_icl, ᶜq_precip)
end
_materialized(dest, src::Union{Fields.Field, Number}) = src
_materialized(dest, src) = (dest .= src)

"""
    set_quadrature_cloud_fraction!(Y, p, env_state, c0, c1, ::Val{aitken})

Picard update of the `QuadratureCloud` cloud fraction in one column loop: the
truncated-Gaussian closure with the linearized `σ_S` (see
`_compute_cloud_fraction`) at the state `env_state` (`cloud_env_state`), with
EDMF updraft weighting (`_edmf_weight_cloud_fraction!`). Without `aitken`, the
result is also stored in the first Picard iterate `c1`; with it, the result is
the guarded Aitken acceleration of `c0`, `c1`, and the new iterate.
"""
function set_quadrature_cloud_fraction!(
    Y, p, env_state, c0, c1, ::Val{aitken},
) where {aitken}
    (; ᶜρ_env, ᶜT_mean, ᶜq_mean, ᶜq_lcl, ᶜq_icl) = env_state
    (; ᶜcloud_fraction, ᶜT′T′, ᶜq′q′, ᶜcorr_Tq, ᶜα) = p.precomputed
    (; turbconv_model) = p.atmos
    thermo_params = CAP.thermodynamics_params(p.params)
    sgs_quad = p.atmos.sgs_quadrature
    floor = cloud_fraction_floor_params(p.params)
    # Carried in by type, so the parameters fold to literals in the kernel.
    vthermo = _fold(ClimaComms.device(Y.c), thermo_params)
    nonequil_val = Val(p.atmos.microphysics_model isa NonEquilibriumMicrophysics)
    DataLayouts.foreach_column(
        ᶜcloud_fraction, c0, c1, Y.c, ᶜT_mean, ᶜρ_env, ᶜq_mean, ᶜq_lcl, ᶜq_icl,
        ᶜT′T′, ᶜq′q′, ᶜcorr_Tq, ᶜα, _edmf_weighting_fields(Y, p)...,
    ) do ᶜcloud_fraction, c0, c1, Yc, ᶜT_mean, ᶜρ_env, ᶜq_mean, ᶜq_lcl, ᶜq_icl,
    ᶜT′T′, ᶜq′q′, ᶜcorr_Tq, ᶜα, ᶜρʲs, ᶜq_liqʲs, ᶜq_iceʲs
        @. ᶜcloud_fraction = _compute_cloud_fraction(
            $(vthermo), ᶜT_mean, ᶜρ_env, ᶜq_mean, ᶜq_lcl, ᶜq_icl,
            $(sgs_quad), ᶜT′T′, ᶜq′q′, ᶜcorr_Tq, ᶜα, $(floor),
        )
        _edmf_weight_cloud_fraction!(
            ᶜcloud_fraction, Yc, ᶜρʲs, ᶜq_liqʲs, ᶜq_iceʲs, ᶜρ_env,
            turbconv_model, thermo_params, nonequil_val,
        )
        if aitken
            @. ᶜcloud_fraction = _aitken_picard_helper(c0, c1, ᶜcloud_fraction)
        else
            @. c1 = ᶜcloud_fraction
        end
        return nothing
    end
    return nothing
end

"""
    set_sgs_moments_and_cloud_fraction!(Y, p, env_state)

Final post-Aitken update for the 1M and 2M schemes, in one column loop: fill
`ᶜsgs_moments = (sigma_S, λ_lagrange, λ_eq)` (`_compute_sgs_moments`), then
compute `ᶜcloud_fraction` consistently with the augmented `σ_aug` closure (see
`_compute_cloud_fraction`) at the state `env_state` (`cloud_env_state`), with
EDMF updraft weighting (`_edmf_weight_cloud_fraction!`).
"""
NVTX.@annotate function set_sgs_moments_and_cloud_fraction!(Y, p, env_state)
    (; ᶜρ_env, ᶜT_mean, ᶜq_mean, ᶜq_lcl, ᶜq_icl, ᶜq_precip) = env_state
    (; ᶜT′T′, ᶜq′q′, ᶜcorr_Tq, ᶜα, ᶜsgs_moments, ᶜcloud_fraction) =
        p.precomputed
    (; turbconv_model) = p.atmos
    thermo_params = CAP.thermodynamics_params(p.params)
    zero_FT = zero(eltype(p.params))
    sgs_quad = p.atmos.sgs_quadrature
    floor = cloud_fraction_floor_params(p.params)
    # Carried in by type, so the parameters fold to literals in the kernel.
    vthermo = _fold(ClimaComms.device(Y.c), thermo_params)
    c = equilibrium_condensate_truncation(p)
    nonequil_val = Val(p.atmos.microphysics_model isa NonEquilibriumMicrophysics)
    # The displacement coefficients enter the loop as a field only when the
    # curvature-aware condensate is configured; otherwise `_disp_lin` and
    # `_disp_curv` of a scalar field give zeros, for which
    # `displacement_condensate` is the truncated-Gaussian expectation.
    ᶜdisp_S_or_zero =
        p.atmos.sgs_variance_curvature ? p.precomputed.ᶜdisp_S : ᶜcorr_Tq
    DataLayouts.foreach_column(
        ᶜsgs_moments, ᶜcloud_fraction, Y.c, ᶜρ_env, ᶜT_mean, ᶜq_mean,
        ᶜq_lcl, ᶜq_icl, ᶜq_precip, ᶜT′T′, ᶜq′q′, ᶜcorr_Tq, ᶜα, ᶜdisp_S_or_zero,
        _edmf_weighting_fields(Y, p)...,
    ) do ᶜsgs_moments, ᶜcloud_fraction, Yc, ᶜρ_env, ᶜT_mean, ᶜq_mean,
    ᶜq_lcl, ᶜq_icl, ᶜq_precip, ᶜT′T′, ᶜq′q′, ᶜcorr_Tq, ᶜα, ᶜdisp_S,
    ᶜρʲs, ᶜq_liqʲs, ᶜq_iceʲs
        _sgs_moments_and_cloud_fraction_point!(
            ᶜsgs_moments, ᶜcloud_fraction, ᶜρ_env, ᶜT_mean, ᶜq_mean,
            (@. lazy(max(zero_FT, ᶜq_mean - ᶜq_precip))),
            ᶜq_lcl, ᶜq_icl, ᶜT′T′, ᶜq′q′, ᶜcorr_Tq, ᶜα,
            (@. lazy(_disp_lin(ᶜdisp_S))), (@. lazy(_disp_curv(ᶜdisp_S))),
            c, vthermo, thermo_params, sgs_quad, floor,
        )
        _edmf_weight_cloud_fraction!(
            ᶜcloud_fraction, Yc, ᶜρʲs, ᶜq_liqʲs, ᶜq_iceʲs, ᶜρ_env,
            turbconv_model, thermo_params, nonequil_val,
        )
        return nothing
    end
    return nothing
end

"""
    _sgs_moments_and_cloud_fraction_point!(
        ᶜsgs_moments, ᶜcloud_fraction, ᶜρ_env, ᶜT_mean, ᶜq_mean, ᶜq_cloud,
        ᶜq_lcl, ᶜq_icl, ᶜT′T′, ᶜq′q′, ᶜcorr_Tq, ᶜα, lin, curv, c,
        vthermo, thermo_params, sgs_quad, floor,
    )

Part of the column loop of `set_sgs_moments_and_cloud_fraction!`: fill the SGS
moments (`_compute_sgs_moments`) and the cloud fraction
(`_compute_cloud_fraction`) of one column. `lin` and `curv` are the
displacement coefficients, read from `ᶜdisp_S` or zero, and `c` the
displacement truncation (`sgs_displacement_truncation`). `vthermo` carries the
thermodynamics parameters in its type, so they fold to literals in the quadrature
kernel; `thermo_params` is the same parameters as a value, for the saturation
calls beside it.
"""
@inline function _sgs_moments_and_cloud_fraction_point!(
    ᶜsgs_moments, ᶜcloud_fraction, ᶜρ_env, ᶜT_mean, ᶜq_mean, ᶜq_cloud,
    ᶜq_lcl, ᶜq_icl, ᶜT′T′, ᶜq′q′, ᶜcorr_Tq, ᶜα, lin, curv, c,
    vthermo, thermo_params, sgs_quad, floor,
)
    @. ᶜsgs_moments = _compute_sgs_moments(
        vthermo, ᶜρ_env, ᶜT_mean, ᶜq_mean, ᶜq_lcl + ᶜq_icl,
        $(sgs_quad), ᶜT′T′, ᶜq′q′, ᶜcorr_Tq, ᶜα, ᶜq_cloud, lin, curv, c,
    )
    # Recompute CF from q_c and σ_S using the augmented-σ closure. `Φ(λ/σ_aug)`
    # is not usable because λ was computed with the equilibrium σ_S_eff, not
    # σ_aug, so it would not match the truncated-Gaussian closure for the
    # augmented variance. This overwrites the Picard iterate
    # with a value consistent with the final SGS moments, so the caller
    # re-applies the EDMF weighting (`_edmf_weight_cloud_fraction!`).
    @. ᶜcloud_fraction = _compute_cloud_fraction(
        ᶜq_lcl + ᶜq_icl,
        ᶜq_mean - TD.q_vap_saturation(thermo_params, ᶜT_mean, ᶜρ_env),
        ᶜsgs_moments.sigma_S,
        TD.q_vap_saturation(thermo_params, ᶜT_mean, ᶜρ_env, ᶜq_lcl, ᶜq_icl),
        ᶜα,
        $(floor),
    )
    return nothing
end

"""
    _edmf_weight_cloud_fraction!(
        ᶜcloud_fraction, Yc, ᶜρʲs, ᶜq_liqʲs, ᶜq_iceʲs, ᶜρ_env,
        turbconv_model, thermo_params, ::Val{nonequil},
    )

Column body of the EDMF cloud weighting in fused loops: scale the environment
cloud fraction by the environment area fraction and add the cloudy area of each
updraft, which is binary: `draft_area(ρaʲ, ρʲ)` where the updraft carries cloud
condensate and `0` otherwise. With non-equilibrium microphysics, the condensate is
the prognostic `q_lclʲ + q_iclʲ` (the cached `ᶜq_liqʲs` and `ᶜq_iceʲs` include
precipitation); otherwise, it is the cached `ᶜq_liqʲs + ᶜq_iceʲs`. No-op without
`PrognosticEDMFX`. The draft fields come from `_edmf_weighting_fields`.
"""
@inline _edmf_weight_cloud_fraction!(
    ᶜcloud_fraction, Yc, ᶜρʲs, ᶜq_liqʲs, ᶜq_iceʲs, ᶜρ_env,
    turbconv_model, thermo_params, nonequil_val,
) = nothing
@inline function _edmf_weight_cloud_fraction!(
    ᶜcloud_fraction, Yc, ᶜρʲs, ᶜq_liqʲs, ᶜq_iceʲs, ᶜρ_env,
    turbconv_model::PrognosticEDMFX, thermo_params, nonequil_val,
)
    ᶜρa⁰ = @. lazy(ρa⁰(Yc.ρ, Yc.sgsʲs, turbconv_model))
    @. ᶜcloud_fraction *= draft_area(ᶜρa⁰, ᶜρ_env)
    unrolled_foreach(StaticOneTo(n_mass_flux_subdomains(turbconv_model))) do j
        @inline
        ᶜsgsʲ = Yc.sgsʲs.:($j)
        ᶜq_lclʲ, ᶜq_iclʲ = _draft_cloud_condensate(
            ᶜsgsʲ, ᶜq_liqʲs.:($j), ᶜq_iceʲs.:($j), nonequil_val,
        )
        @. ᶜcloud_fraction += updraft_cloudy_area(
            thermo_params, max(0, ᶜq_lclʲ + ᶜq_iclʲ), ᶜsgsʲ.ρa, ᶜρʲs.:($$j),
        )
        return nothing
    end
    return nothing
end
@inline _draft_cloud_condensate(ᶜsgsʲ, ᶜq_liqʲ, ᶜq_iceʲ, ::Val{true}) =
    (ᶜsgsʲ.q_lcl, ᶜsgsʲ.q_icl)
@inline _draft_cloud_condensate(ᶜsgsʲ, ᶜq_liqʲ, ᶜq_iceʲ, ::Val{false}) =
    (ᶜq_liqʲ, ᶜq_iceʲ)

# The binary cloudy area of an updraft with non-negative cloud condensate `qʲ`.
@inline updraft_cloudy_area(thermo_params, qʲ, ρaʲ, ρʲ) =
    ifelse(TD.has_condensate(thermo_params, qʲ), draft_area(ρaʲ, ρʲ), zero(qʲ))

# The draft densities and cached draft condensate the EDMF cloud weighting reads in
# fused loops, with grid-mean placeholders where the weighting reads none.
function _edmf_weighting_fields(Y, p)
    p.atmos.turbconv_model isa PrognosticEDMFX || return (Y.c.ρ, Y.c.ρ, Y.c.ρ)
    (; ᶜρʲs) = p.precomputed
    p.atmos.microphysics_model isa NonEquilibriumMicrophysics &&
        return (ᶜρʲs, ᶜρʲs, ᶜρʲs)
    return (ᶜρʲs, p.precomputed.ᶜq_liqʲs, p.precomputed.ᶜq_iceʲs)
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
`a_p` enters the node rule only as the fraction of the PDF that carries the
shaft (`sgs_precip_shaft_threshold`); the per-node assignment is normalized
by the weight of the shaft nodes, so no measure consistency between the cover
and the node rule is required for conservation.

Mutates `p.precomputed.ᶜprecip_frac`.
"""
NVTX.@annotate function set_precip_fraction!(
    Y,
    p,
    ᶜq_precip = _get_precip_mean(
        Y,
        p,
        p.atmos.turbconv_model,
        p.atmos.microphysics_model,
    ),
)
    hasproperty(p.precomputed, :ᶜprecip_frac) || return nothing
    FT = eltype(p.params)
    f_decay = clamp(FT(CAP.sgs_precip_overlap_decay(p.params)), zero(FT), one(FT))
    thermo_params = CAP.thermodynamics_params(p.params)
    _precip_fraction_sweep!(
        p.precomputed.ᶜprecip_frac,
        p.precomputed.ᶜcloud_fraction,
        ᶜq_precip,
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

On GPUs the recursion is one `Operators.column_accumulate!` from the top level
down, seeded with `-1`, which marks the top level: it takes the cover there
without the `min(1, ·)` bound of the levels below it, as the level loop does. On
CPUs it is a loop of level broadcasts, since the CPU path of
`column_accumulate!` materializes a level `Field` per level per column; the loop
allocates per level independent of the number of columns. The two give the same
result to the bit.
"""
function _precip_fraction_sweep!(ᶜprecip_frac, ᶜcf, ᶜq_precip, f_decay, q_precip_min)
    FT = eltype(ᶜprecip_frac)
    (zero_FT, one_FT) = (zero(FT), one(FT))
    if ClimaComms.device(ᶜprecip_frac) isa ClimaComms.CUDADevice
        input = Base.broadcasted(tuple, ᶜcf, ᶜq_precip)
        Operators.column_accumulate!(
            ᶜprecip_frac, input; init = -one_FT, reverse = true,
        ) do a_p_above, (cf, q_precip)
            ifelse(
                q_precip > q_precip_min,
                ifelse(
                    a_p_above < zero_FT,
                    max(cf, zero_FT),
                    min(one_FT, max(max(cf, zero_FT), f_decay * a_p_above)),
                ),
                zero_FT,
            )
        end
        return nothing
    end
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
  - `MLCloud`: the neural-network prediction (see
    `set_ml_cloud_fraction!`).

A moist `QuadratureCloud` is updated in fused column loops instead
(`set_quadrature_cloud_fraction!`).

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

"""
    _grid_mean_precipitation(Y, p, microphysics_model)

Grid-mean non-negative precipitation `q_rai + q_sno`: a lazy field for
non-equilibrium microphysics, which carries `ρq_rai` and `ρq_sno`, and zero
for models without prognostic precipitation (dry, 0M).
"""
_grid_mean_precipitation(Y, p, ::NonEquilibriumMicrophysics) = @. lazy(
    max(0, specific(Y.c.ρq_rai, Y.c.ρ)) + max(0, specific(Y.c.ρq_sno, Y.c.ρ)),
)
_grid_mean_precipitation(Y, p, microphysics_model) = zero(eltype(p.params))
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
    _env_air_density(p, thermo_params)

Lazy environment air density `ᶜρ⁰` [kg/m³] for `PrognosticEDMFX`.
"""
function _env_air_density(p, thermo_params)
    (; ᶜp, ᶜT⁰, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰) = p.precomputed
    return @. lazy(
        TD.air_density(
            thermo_params,
            ᶜT⁰,
            ᶜp,
            ᶜq_tot_nonneg⁰,
            ᶜq_liq⁰,
            ᶜq_ice⁰,
        ),
    )
end

"""
    _env_area_fraction(Y, p, thermo_params, turbconv_model)

Environment area fraction `ᶜa⁰` [-] for `PrognosticEDMFX`, or `1` otherwise.
"""
function _env_area_fraction(Y, p, thermo_params, turbconv_model::PrognosticEDMFX)
    ᶜρa⁰ = @. lazy(ρa⁰(Y.c.ρ, Y.c.sgsʲs, turbconv_model))
    ᶜρ⁰ = _env_air_density(p, thermo_params)
    return @. lazy(draft_area(ᶜρa⁰, ᶜρ⁰))
end
_env_area_fraction(Y, p, thermo_params, turbconv_model) = one(eltype(p.params))

"""
    _get_env_ρ_T_q(Y, p, thermo_params, turbconv_model)

Get environment density, temperature, and specific humidity for cloud fraction.
Lightweight alternative to `_compute_cloud_state` when only ρ, T, and q are needed.
"""
function _get_env_ρ_T_q(Y, p, thermo_params, turbconv_model)
    (; ᶜT, ᶜq_tot_nonneg) = p.precomputed
    if turbconv_model isa PrognosticEDMFX
        (; ᶜT⁰, ᶜq_tot_nonneg⁰) = p.precomputed
        ᶜρ_env = _env_air_density(p, thermo_params)
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
    (; ᶜT, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice) = p.precomputed

    if turbconv_model isa PrognosticEDMFX
        (; ᶜT⁰, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰) = p.precomputed
        ᶜρ_env = _env_air_density(p, thermo_params)
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
separately by `_edmf_weight_cloud_fraction!`.
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
    _get_precip_mean(Y, p, turbconv_model, microphysics_model)

Non-negative mean precipitation `q_rai + q_sno` of the domain that carries the
SGS cloud closure: the environment for PrognosticEDMFX
(`_env_precipitation`), the grid mean otherwise (`_grid_mean_precipitation`).
Each species is clamped as in the `q_rai_nonneg`, `q_sno_nonneg` that
`microphysics_tendencies_1m` hands to `Microphysics1MEvaluator`, so the
cloud-available excess `μ_{S,cloud}` of `_compute_sgs_moments` and the node
excess `S_raw_cloud` of the evaluator subtract the same precipitation.
"""
_get_precip_mean(Y, p, turbconv_model, microphysics_model) =
    turbconv_model isa PrognosticEDMFX ?
    _env_precipitation(Y, p, microphysics_model) :
    _grid_mean_precipitation(Y, p, microphysics_model)

"""
    _env_precipitation(Y, p, microphysics_model)

Environment non-negative precipitation `q_rai⁰ + q_sno⁰` for PrognosticEDMFX:
a lazy field for non-equilibrium microphysics, which carries `q_rai` and
`q_sno`, and zero for models without prognostic precipitation (dry, 0M).
"""
_env_precipitation(Y, p, ::NonEquilibriumMicrophysics) = @. lazy(
    max(0, $(ᶜspecific_env_value(@name(q_rai), Y, p))) +
    max(0, $(ᶜspecific_env_value(@name(q_sno), Y, p))),
)
_env_precipitation(Y, p, microphysics_model) = zero(eltype(p.params))

"""
    _updraft_cloud_condensate(Y, p, j, microphysics_model)

Cloud condensate of updraft `j`, used for the binary updraft cloud check.
With non-equilibrium microphysics, uses the prognostic cloud condensate
(`q_lclʲ`, `q_iclʲ`) only; the precomputed `ᶜq_liqʲs` / `ᶜq_iceʲs` include
precipitation (`q_raiʲ` / `q_snoʲ`), which should not count as cloud.
"""
_updraft_cloud_condensate(Y, p, j, microphysics_model) = _draft_cloud_condensate(
    Y.c.sgsʲs.:($j),
    p.precomputed.ᶜq_liqʲs.:($j),
    p.precomputed.ᶜq_iceʲs.:($j),
    Val(microphysics_model isa NonEquilibriumMicrophysics),
)

"""
    _updraft_cloud_condensate_and_area(Y, p, j, thermo_params, microphysics_model)

Non-negative cloud condensate `ᶜqʲ` [kg/kg] and binary cloudy area fraction
`ᶜaʲ` [-] of updraft `j`, equal to `draft_area(ρaʲ, ρʲ)` where
`TD.has_condensate(thermo_params, ᶜqʲ)` holds and `0` otherwise.
"""
function _updraft_cloud_condensate_and_area(
    Y, p, j, thermo_params, microphysics_model,
)
    FT = eltype(p.params)
    ᶜq_liqʲ, ᶜq_iceʲ = _updraft_cloud_condensate(Y, p, j, microphysics_model)
    ᶜqʲ = @. lazy(max(zero(FT), ᶜq_liqʲ + ᶜq_iceʲ))
    ᶜaʲ = @. lazy(
        updraft_cloudy_area(
            thermo_params, ᶜqʲ, Y.c.sgsʲs.:($$j).ρa, p.precomputed.ᶜρʲs.:($$j),
        ),
    )
    return ᶜqʲ, ᶜaʲ
end

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
