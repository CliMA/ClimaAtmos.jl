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
    # (`set_covariance_cache!` writes `ᶜtemp_scalar` (l_d), `_2` (N²_disp),
    # `_3`, `_5`, `_6`; the QuadratureCloud `set_cloud_fraction!` writes
    # `ᶜtemp_scalar`, `_2`, `_3`).
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
"""
@inline displacement_length(l_mix, l_b, l_z) =
    min(max(l_mix, l_b), max(l_z, zero(l_z)))

"""
    materialized_displacement_length!(Y, p)

Materialize the vertical displacement length `l_d` of the variance closure
into `p.scratch.ᶜtemp_scalar` and return `(ᶜl_d, ᶜl_mix_disp)`, with
`ᶜl_mix_disp` the buoyancy-bounded turbulent mixing length
(`buoyancy_bounded_mixing_length`) as a lazy broadcast. Also refreshes
`p.precomputed.ᶜl_mix` for `AbstractEDMF` via `materialized_mixing_length!`.

`l_d` takes the maximum of `l_disp` (the mixing length `l_mix` bounded by
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
bounded by the wall distance (`displacement_length`).
The floor is kept out of the mixing length itself, because the
surface-layer wall scale `l_W` of the mixing length vanishes with the
friction velocity and in a stable surface layer, which would collapse `l_d`
in the free troposphere. The floor enters the variance only, not `ᶜl_mix` or
the diffusivity. Writes `p.scratch.ᶜtemp_scalar`, `p.scratch.ᶜtemp_scalar_2`,
and the Prandtl-number scratch of the mixing-length closure.
"""
function materialized_displacement_length!(Y, p)
    FT = eltype(p.params)
    turbconv_params = CAP.turbconv_params(p.params)
    c_b = FT(CAP.static_stab_coeff(turbconv_params))
    c_σ = FT(CAP.diagnostic_covariance_coeff(p.params))
    # `κ_b = κ_b_per_tke tke` is the vertical velocity variance of the
    # displacing eddies in the normalization of `l_disp`: the turbulent
    # displacement variance is `2 c_σ l_disp² = c_b tke τ_turb²`, and the
    # buoyancy bound `κ_b / N²` on `l_disp²` is the adiabatic excursion
    # `c_b tke / N²` of `2 c_σ l_disp²`.
    κ_b_per_tke = c_b / (2 * c_σ)
    w_min = floor_velocity(
        horizontal_filter_scale(axes(Y.c)),
        FT(CAP.sgs_variance_vertical_velocity_floor(p.params)),
        FT(CAP.sgs_variance_floor_reference_resolution(p.params)),
        FT(CAP.sgs_variance_floor_resolution_exponent(p.params)),
    )
    τ_0 = FT(CAP.sgs_variance_floor_time_scale(p.params))
    κ_min = w_min^2 / (2 * c_σ * c_b)
    ᶜl_mix = materialized_mixing_length!(Y, p)
    (; ᶜbg_coeffs, ᶠ∂θli∂z, ᶠ∂qt∂z) = p.precomputed
    ᶜN²_disp = p.scratch.ᶜtemp_scalar_2
    set_displacement_buoyancy_gradient!(ᶜN²_disp, ᶜbg_coeffs, ᶠ∂θli∂z, ᶠ∂qt∂z)
    ᶜz = Fields.coordinate_field(Y.c).z
    z_sfc = Fields.level(Fields.coordinate_field(Y.f).z, Fields.half)
    ᶜΔz = Fields.Δz_field(axes(Y.c))
    ᶜl_d = p.scratch.ᶜtemp_scalar
    if hasproperty(Y.c, :ρtke)
        ᶜtke = @. lazy(max(specific(Y.c.ρtke, Y.c.ρ), zero(FT)))
        ᶜl_mix_disp = @. lazy(
            buoyancy_bounded_mixing_length(
                ᶜl_mix,
                κ_b_per_tke * ᶜtke,
                ᶜN²_disp,
            ),
        )
        ᶜτ_turb = @. lazy(
            turbulent_lagrangian_time(ᶜl_mix_disp, κ_b_per_tke * ᶜtke),
        )
        @. ᶜl_d = displacement_length(
            ᶜl_mix_disp,
            floor_displacement_length(
                c_b * κ_min, ᶜN²_disp, ᶜτ_turb, τ_0, ᶜz - z_sfc, ᶜΔz,
            ),
            ᶜz - z_sfc,
        )
        return (ᶜl_d, ᶜl_mix_disp)
    else
        # Without prognostic TKE the grid-mean Smagorinsky length, materialized
        # in `ᶜtemp_scalar` by `materialized_mixing_length!`, enters `l_d`
        # pointwise in place (the broadcast reads and writes the same cell),
        # and the turbulence imposes no decorrelation limit on the floor.
        @. ᶜl_d = displacement_length(
            ᶜl_mix,
            floor_displacement_length(
                c_b * κ_min, ᶜN²_disp, FT(Inf), τ_0, ᶜz - z_sfc, ᶜΔz,
            ),
            ᶜz - z_sfc,
        )
        return (ᶜl_d, zero(FT))
    end
end

"""
    materialized_variance_fidelity!(Y, p, ᶜl_mix_disp)

Refresh `p.precomputed.ᶜα` from the current TKE, condensate state, and the
buoyancy-bounded mixing length `ᶜl_mix_disp` returned by
`materialized_displacement_length!` (see `sgs_variance_fidelity`). Called at
the end of `set_covariance_cache!` so `ᶜα` tracks each Picard iterate.

`σ_S` is estimated with the Gaussian-linear analytic formula
`σ_S² = σ_q² − 2γ Cov(T,q) + γ² σ_T²`, with `γ = ∂q_vap_sat_∂T`. This is
the fast estimate; the downstream quadrature closures still compute their
own `σ_S`.

Non-EDMF configurations without TKE substitute `TKE = 0` and `l_mix = 1 m`,
collapsing the Damköhler factor to 1.
"""
function materialized_variance_fidelity!(Y, p, ᶜl_mix_disp)
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
    # With prognostic TKE the eddy turnover rate `c_d √TKE / l_mix` uses the
    # mixing length bounded by the face-maximum dry buoyancy excursion of the
    # displacement closure (`buoyancy_bounded_mixing_length`), so it reflects
    # the inversion-bounded eddy size; without TKE, α_Da → 1.
    if hasproperty(Y.c, :ρtke) && hasproperty(p.precomputed, :ᶜl_mix)
        ᶜtke_or_zero = @. lazy(max(specific(Y.c.ρtke, Y.c.ρ), zero(FT)))
        ᶜl_mix_or_one = ᶜl_mix_disp
    else
        ᶜtke_or_zero = zero(FT)
        ᶜl_mix_or_one = one(FT)
    end
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
    # The equilibrium target of the modulator is the one the condensate relaxes
    # toward: curvature-aware where `ᶜdisp_S` is configured, Gaussian otherwise.
    ᶜlin, ᶜcurv = if p.atmos.sgs_variance_curvature
        (p.precomputed.ᶜdisp_S.lin, p.precomputed.ᶜdisp_S.curv)
    else
        (zero(FT), zero(FT))
    end
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
        ᶜlin,
        ᶜcurv,
    )
    return nothing
end

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
    set_displacement_curvature!(
        ᶜdisp_S, Y, p, thermo_params, ᶜl_d, ᶜ∂T_∂θ, ᶜ∂q∂z, ᶜ∂θ∂z,
    )

Fill `ᶜdisp_S = (; lin, curv)` (`displacement_curvature_coefficients`) from
the moisture vertical scale `ℓ_v = √(2 c_σ) l_d`, the Van Leer centre
gradients `ᶜ∂q∂z`, `ᶜ∂θ∂z`, and the second derivatives
`(∂ψ∂z⁺ − ∂ψ∂z⁻) / Δz` of the two bounding face gradients
`p.precomputed.ᶠ∂qt∂z`, `p.precomputed.ᶠ∂θli∂z`. At the domain boundaries,
the adjacent interior face gradient stands in for the boundary face, as in
`ᶜVanLeer_gradient!`, so the curvature is zero in the bottom and top cells.

`lin` uses the harmonic-mean-limited centre gradient that the covariance
uses, so `lin²` stays the vertical part of `σ_S²`; the limiter zeros it in
a cell whose two face gradients differ in sign, where `curv` is at its
largest and the whole vertical effect is the bound of the quadratic.
"""
function set_displacement_curvature!(
    ᶜdisp_S, Y, p, thermo_params, ᶜl_d, ᶜ∂T_∂θ, ᶜ∂q∂z, ᶜ∂θ∂z,
)
    FT = eltype(p.params)
    (; ᶜT, ᶠ∂θli∂z, ᶠ∂qt∂z) = p.precomputed
    c_σ = FT(CAP.diagnostic_covariance_coeff(p.params))
    r_v_θ = FT(CAP.sgs_variance_theta_vertical_scale_ratio(p.params))
    nc = Spaces.nlevels(axes(Y.c))
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
    ᶜΔz = Fields.Δz_field(axes(Y.c))
    ᶜχ = @. lazy(r_v_θ * TD.∂q_vap_sat_∂T(thermo_params, ᶜT, Y.c.ρ) * ᶜ∂T_∂θ)
    @. ᶜdisp_S = displacement_curvature_coefficients(
        sqrt(2 * c_σ) * ᶜl_d,
        ᶜχ,
        ᶜ∂q∂z,
        ᶜ∂θ∂z,
        (ᶜq_above(ᶠ∂qt∂z) - ᶜq_below(ᶠ∂qt∂z)) / ᶜΔz,
        (ᶜθ_above(ᶠ∂θli∂z) - ᶜθ_below(ᶠ∂θli∂z)) / ᶜΔz,
    )
    return nothing
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

"""
    set_tq_correlation!(p, correlation_model)

Fill `p.precomputed.ᶜcorr_Tq` with the SGS T–q correlation the quadrature samples.
`ConstantTqCorrelation`: the prescribed `Tq_correlation_coefficient`.
`DiagnosedTqCorrelation`: `T′q′ / √(T′T′ q′q′)` from the gradient covariance
accumulated in `p.precomputed.ᶜT′q′` by `set_covariance_cache!` (see
`sgs_correlation_Tq`); the fallback where a variance vanishes relative to its
grid mean is the prescribed value. `ᶜT′T′` and `ᶜq′q′` must already be in the T basis and
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
    (; ᶜT′T′, ᶜq′q′, ᶜT′q′, ᶜcorr_Tq, ᶜT, ᶜq_tot_nonneg) = p.precomputed
    fallback = correlation_Tq(p.params)
    @. ᶜcorr_Tq =
        sgs_correlation_Tq(ᶜT′T′, ᶜq′q′, ᶜT′q′, ᶜT, ᶜq_tot_nonneg, fallback)
    return nothing
end

"""
    set_covariance_cache!(Y, p, thermo_params)

Materializes T-based SGS covariances into cached fields for use by downstream
computations (SGS quadrature, cloud fraction). Populates
`p.precomputed.(ᶜT′T′, ᶜq′q′, ᶜcorr_Tq)` (and `ᶜT′q′` for the diagnosed correlation).

The covariance of `(θ_li, q_tot)` is that of a Gaussian subgrid displacement
acting on the resolved gradients (`displacement_covariance`), with moisture
horizontal scale `ℓ_{h,0}² = c_g (c_Δx Δx_h)²` from the horizontal grid scale,
moisture vertical displacement scale `ℓ_{v,0}² = 2 c_σ l_d²` from the
displacement length `l_d` (`materialized_displacement_length!`),
potential-temperature scale ratios `r_{Δx,θ} = ℓ_{h,θ} / ℓ_{h,0}`
(`sgs_variance_theta_horizontal_scale_ratio`) and
`r_{v,θ} = ℓ_{v,θ} / ℓ_{v,0}` (`sgs_variance_theta_vertical_scale_ratio`), and
horizontal displacement cross-correlation `r_{h,θq}`
(`sgs_variance_horizontal_cross_correlation`); the vertical displacements of
the two scalars are perfectly correlated. The resulting `2 × 2` covariance is
positive semi-definite for `|r_{h,θq}| ≤ 1`, which `SGSQuadratureParameters`
enforces.

Pipeline:

 1. Materialize `l_d` (`materialized_displacement_length!`, which also refreshes
    `ᶜl_mix` via `materialized_mixing_length!`).
 2. Horizontal gradient invariants of `θ_li` and `q_tot` (skipped when
    `sgs_variance_horizontal_scale_factor` is 0).
 3. θ′θ′, q′q′ and, for the diagnosed correlation, θ′q′ from the displacement.
 4. Transform θ→T using `compute_∂T_∂θ!`; with `sgs_variance_curvature`,
    the displacement coefficients `ᶜdisp_S` (`set_displacement_curvature!`).
 5. Apply the closure-validity bound `σ_q ≤ sgs_variance_max_rel_std * q_tot`.
 6. Set the sampled T–q correlation (`set_tq_correlation!`; for the diagnosed
    model the covariance is first rescaled with the bound so the correlation is
    invariant under it).
 7. Refresh the variance fidelity `ᶜα`.

Scratch: `ᶜtemp_scalar` holds `l_d`, `ᶜtemp_scalar_2` `N²_disp`,
`ᶜtemp_scalar_3` θ_li and then the θ→T Jacobian, `ᶜtemp_scalar_5` and `_6` the
moisture and cross invariants (`ᶜT′T′` holds the θ_li invariant before Step 3).
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
    ᶜlg = Fields.local_geometry_field(Y.c)

    # 1. Displacement scales.
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
    ᶜl_d, ᶜl_mix_disp = materialized_displacement_length!(Y, p)
    ᶜℓ_v_q² = @. lazy(2 * c_σ * ᶜl_d^2)
    ᶜℓ_v_θ² = @. lazy(r_v_θ^2 * ᶜℓ_v_q²)
    ᶜℓ_v_θq = @. lazy(r_v_θ * ᶜℓ_v_q²)

    # 2. Horizontal gradient invariants.
    if iszero(c_Δx)
        ᶜinv_q = ᶜinv_θ = ᶜinv_θq = zero(FT)
    else
        (; ᶜT, ᶜq_liq, ᶜq_ice) = p.precomputed
        ᶜθ_li = p.scratch.ᶜtemp_scalar_3
        @. ᶜθ_li = TD.liquid_ice_pottemp(
            thermo_params, ᶜT, Y.c.ρ, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice,
        )
        ᶜinv_q = p.scratch.ᶜtemp_scalar_5
        ᶜinv_θ = ᶜT′T′
        ᶜinv_θq = p.scratch.ᶜtemp_scalar_6
        hgrad_invariant!(ᶜinv_q, ᶜq_tot_nonneg, p)
        hgrad_invariant!(ᶜinv_θ, ᶜθ_li, p)
        hgrad_cross_invariant!(ᶜinv_θq, ᶜθ_li, ᶜq_tot_nonneg, p)
    end

    # 3. Covariances in the θ basis; ᶜT′T′ holds θ′θ′ until the transform.
    ᶜ∂q∂z = @. lazy(projected_vector_data(C3, ᶜgradᵥ_q_tot, ᶜlg))
    ᶜ∂θ∂z = @. lazy(projected_vector_data(C3, ᶜgradᵥ_θ_liq_ice, ᶜlg))
    @. ᶜq′q′ = displacement_covariance(ℓ_h_q², ᶜℓ_v_q², ᶜinv_q, ᶜ∂q∂z, ᶜ∂q∂z)
    @. ᶜT′T′ = displacement_covariance(ℓ_h_θ², ᶜℓ_v_θ², ᶜinv_θ, ᶜ∂θ∂z, ᶜ∂θ∂z)
    if diag_corr
        @. p.precomputed.ᶜT′q′ =
            displacement_covariance(ℓ_h_θq, ᶜℓ_v_θq, ᶜinv_θq, ᶜ∂θ∂z, ᶜ∂q∂z)
    end

    # 4. θ → T basis via the thermodynamic Jacobian ∂T/∂θ. q is unaffected.
    ᶜ∂T_∂θ = p.scratch.ᶜtemp_scalar_3
    compute_∂T_∂θ!(ᶜ∂T_∂θ, Y, p, thermo_params)
    @. ᶜT′T′ = ᶜ∂T_∂θ^2 * ᶜT′T′
    if diag_corr
        @. p.precomputed.ᶜT′q′ = ᶜ∂T_∂θ * p.precomputed.ᶜT′q′
    end
    if p.atmos.sgs_variance_curvature
        set_displacement_curvature!(
            p.precomputed.ᶜdisp_S, Y, p, thermo_params,
            ᶜl_d, ᶜ∂T_∂θ, ᶜ∂q∂z, ᶜ∂θ∂z,
        )
    end

    # 5. Closure-validity bound `σ_q ≤ sgs_variance_max_rel_std · q_tot`
    #    (default 0.5). A wider Gaussian puts a quadrature node at negative
    #    total water; the clamped quadrature then carries more total water
    #    than the grid mean holds, and condensing it can drive the grid-mean
    #    vapour negative. T′q′ is rescaled by the same factor as σ_q so the
    #    diagnosed correlation is invariant under the bound.
    r_max = CAP.sgs_variance_max_rel_std(p.params)
    if diag_corr
        @. p.precomputed.ᶜT′q′ *= sqrt(min(
            one(FT),
            (r_max * ᶜq_tot_nonneg)^2 / max(ᶜq′q′, eps(FT)^2),
        ))
    end
    @. ᶜq′q′ = min(ᶜq′q′, (r_max * ᶜq_tot_nonneg)^2)

    # 6. The correlation the quadrature samples.
    set_tq_correlation!(p, p.atmos.tq_correlation_model)

    # 7. Refresh `ᶜα` with the current bounded `ᶜl_mix` and state.
    materialized_variance_fidelity!(Y, p, ᶜl_mix_disp)
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
    q_sat_hat = TD.q_vap_saturation(_unwrapped(eval.tps), T_hat, eval.ρ)
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
    q_tot_hat - TD.q_vap_saturation(_unwrapped(eval.tps), T_hat, eval.ρ) -
    eval.mu_S


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
`C = q_c / (α·σ_S)` stays well-conditioned. A `ConditionalMomentSGS` takes
the linearized `σ_S` instead (`_sgs_linear_sigma_S`), without a quadrature
pass.

Returns `(; mu_S, sigma_S)`.
"""
@inline function _sgs_saturation_moments(
    thp, ρ, T_mean, q_tot_mean,
    sgs_quad, T′T′, q′q′, corr_Tq,
)
    mu_S = q_tot_mean - TD.q_vap_saturation(_unwrapped(thp), T_mean, ρ)
    sigma_S = _sgs_sigma_S(
        sgs_quad, thp, ρ, T_mean, q_tot_mean, mu_S, T′T′, q′q′, corr_Tq,
    )
    return (; mu_S, sigma_S)
end

# An `SGSQuadrature` of `ConditionalMomentSGS`, which takes the linearized `σ_S`.
const ConditionalMomentQuadrature =
    SGSQuadrature{N, A, W, <:ConditionalMomentSGS} where {N, A, W}

# `σ_S` of `_sgs_saturation_moments`: `E[(S − μ_S)²]` over the quadrature points,
# or the linearized width for `ConditionalMomentSGS`.
@inline function _sgs_sigma_S(
    sgs_quad, thp, ρ, T_mean, q_tot_mean, mu_S, T′T′, q′q′, corr_Tq,
)
    FT = typeof(ρ)
    sgs_quad_eff = isnothing(sgs_quad) ? GridMeanSGS() : sgs_quad
    evaluator = SGSVarianceEvaluator(thp, ρ, mu_S)
    sigma_S_sq = integrate_over_sgs(
        evaluator, sgs_quad_eff, q_tot_mean, T_mean, q′q′, T′T′, corr_Tq,
    )
    return max(sqrt(sigma_S_sq), ϵ_numerics(FT))
end
@inline _sgs_sigma_S(
    ::ConditionalMomentQuadrature, thp, ρ, T_mean, q_tot_mean, mu_S,
    T′T′, q′q′, corr_Tq,
) = _sgs_linear_sigma_S(thp, ρ, T_mean, T′T′, q′q′, corr_Tq)

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
# same linearization that justifies Gaussianity of S) and σ_S accumulated
# in one pass as E[(S − μ_S)²] (see `_sgs_saturation_moments`).
# Working with the centred excess S′ = S − μ_S ~ N(0, σ_S²), we seek a
# Lagrange multiplier λ that enforces mass conservation:
#
#     E[max(0, λ + α·S′)] = q_c,                                     (*)
#
# where α = `sgs_variance_fidelity` controls how much of the SGS variance is
# propagated into the local condensate (α = α_Da·α_eq ∈ (0, 1]; the
# equilibrium multiplier λ_eq is fitted separately with α = 1). The effective
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
    truncated_gaussian_condensate(mu_S, sigma_S)

Expected saturation excess `E[max(0, S)] = μ_S Φ(z) + σ_S φ(z)` with
`z = μ_S / σ_S` for a Gaussian saturation excess `S ~ N(μ_S, σ_S²)`.
"""
@inline function truncated_gaussian_condensate(mu_S, sigma_S)
    FT = typeof(mu_S)
    inv_sqrt2π = one(FT) / sqrt(FT(2) * FT(π))
    σ = max(FT(sigma_S), ϵ_numerics(FT))
    z = mu_S / σ
    return max(zero(FT), mu_S * normal_cdf(z) + σ * exp(-z * z / 2) * inv_sqrt2π)
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
    φa = ifelse(isfinite(a), exp(-a * a / 2) * inv_sqrt2π, zero(FT))
    φb = ifelse(isfinite(b), exp(-b * b / 2) * inv_sqrt2π, zero(FT))
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
    D = l * l - 2 * c * m₀
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
    m_curv = m_lin + f.curv * (x * x - 1) / 2
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
        lin = 0, curv = 0,
    )

Local SGS variance fidelity `α = α_Da · α_eq` for the condensate
reconstruction `E[max(0, λ + α·S′)] = q_c`.

  - Damköhler factor
    `α_Da = 1 / (1 + A · τ_mp · c_d · √TKE / max(l_mix, 1 m))`:
    quiescent (TKE → 0) → 1, well-mixed → 0.
  - Equilibrium-ratio modulator `α_eq = r / (w_eq + (1 − w_eq) r)` with
    `r = (q_c_actual + q_min) / (q_c_eq + q_min)` and `q_c_eq` the expected
    saturation excess under the SGS PDF (always ≥ 0 even for subsaturated
    means): the truncated-Gaussian expectation `μ_S Φ(z) + σ_S φ(z)`,
    `z = μ_S/σ_S`, or, with nonzero displacement coefficients `lin`, `curv`,
    the curvature-aware `displacement_condensate(μ_S, σ_S, lin, curv)`, the
    same target the condensate relaxes toward.
"""
@inline function sgs_variance_fidelity(
    A, τ_mp, c_d, tke, l_mix, w_eq, q_c_actual, mu_S, sigma_S, q_min,
    lin = zero(mu_S), curv = zero(mu_S),
)
    FT = typeof(tke)
    inv_τ_turb = FT(c_d) * sqrt(max(tke, zero(FT))) / max(l_mix, one(FT))
    α_Da = one(FT) / (one(FT) + FT(A) * FT(τ_mp) * inv_τ_turb)
    q_c_eq = displacement_condensate(FT(mu_S), FT(sigma_S), FT(lin), FT(curv))
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
    _sgs_augmented_sigma(mu_S, sigma_S, α, floor, λ_liq_eq = 1)

Augmented saturation-excess standard deviation

    σ_aug = sqrt((α · σ_S)² + (ε_rel_eff · max(0, −μ_S))² + σ_abs_eff²)

for the non-equilibrium condensate cloud-fraction closure, floored at
`ϵ_numerics(FT)`. In subsaturated air (`μ_S < 0`), the subsaturation deficit
`δ_sub = max(0, −μ_S)` provides a scale-aware non-equilibrium width
proportional to `q_sat (1 − RH)_+` that keeps trace detrained or residual
condensate patchy when the resolved gradient variance `σ_S` is small. As the
subdomain approaches saturation (`μ_S → 0⁻`), `δ_sub` vanishes with continuous
first derivative, and in saturated cloud decks (`μ_S ≥ 0`) the relative floor is
identically zero so quiescent decks reach full overcast. The ice-phase scaling
factors `s_ice_rel` and `s_ice_abs` modulate `ε_rel` and `σ_abs` across the
mixed-phase transition via `w_ice = (1 − λ_liq_eq)²`.
"""
@inline function _sgs_augmented_sigma(
    mu_S,
    sigma_S,
    α,
    floor,
    λ_liq_eq = one(mu_S),
)
    FT = typeof(mu_S)
    (; ε_rel, σ_abs, s_ice_abs, s_ice_rel) = floor
    δ_sub = max(zero(FT), -mu_S)
    λ_l = clamp(FT(λ_liq_eq), zero(FT), one(FT))
    w_ice = (one(FT) - λ_l)^2
    ε_rel_eff = FT(ε_rel) * ((one(FT) - w_ice) + w_ice * FT(s_ice_rel))
    σ_abs_eff = FT(σ_abs) * ((one(FT) - w_ice) + w_ice * FT(s_ice_abs))
    σ_var = FT(α) * FT(sigma_S)
    σ_floor_sq = (ε_rel_eff * δ_sub)^2 + σ_abs_eff^2
    return max(sqrt(σ_var^2 + σ_floor_sq), ϵ_numerics(FT))
end

"""
    _compute_cloud_fraction(q_c, mu_S, sigma_S, α, floor, λ_liq_eq = 1)

Cloud fraction `CF = Φ(z)`, where `z` solves the truncated-Gaussian condensate
relation `q_c / σ_aug = z·Φ(z) + φ(z)` (`_compute_z`) with the augmented width
`σ_aug = _sgs_augmented_sigma(mu_S, sigma_S, α, floor, λ_liq_eq)`.
"""
@inline function _compute_cloud_fraction(
    q_c,
    mu_S,
    sigma_S,
    α,
    floor,
    λ_liq_eq = one(mu_S),
)
    σ_aug = _sgs_augmented_sigma(mu_S, sigma_S, α, floor, λ_liq_eq)
    return normal_cdf(_compute_z(q_c / σ_aug))
end

"""
    _compute_cloud_fraction(
        thermo_params, T, ρ, q_tot, q_liq, q_ice,
        sgs_quad, T′T′, q′q′, corr_Tq, α, floor,
    )

Fused production overload: compute the hybrid cloud fraction in a single
inlined call that obtains `σ_S` (`_sgs_saturation_moments`: a quadrature pass,
or the linearized width for `ConditionalMomentSGS`) and evaluates the
truncated-Gaussian approximate closure in one broadcast kernel. Used by
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
    λ_liq_eq = TD.liquid_fraction_ramp(_unwrapped(thermo_params), T)
    return _compute_cloud_fraction(
        q_liq + q_ice,
        moments.mu_S,
        moments.sigma_S,
        α,
        floor,
        λ_liq_eq,
    )
end

"""
    CloudFractionFloorParams{FT}

Augmented-`σ` floor parameters for `_sgs_augmented_sigma` and
`_compute_cloud_fraction`, bundled into one isbits broadcast scalar.
"""
Base.@kwdef struct CloudFractionFloorParams{FT}
    ε_rel::FT
    σ_abs::FT
    s_ice_abs::FT = one(ε_rel)
    s_ice_rel::FT = one(ε_rel)
end
Base.broadcastable(x::CloudFractionFloorParams) = tuple(x)

"""
    cloud_fraction_floor_params(params)

Build the `CloudFractionFloorParams` bundle from the model parameter set.
"""
cloud_fraction_floor_params(params) = CloudFractionFloorParams(;
    ε_rel = CAP.cloud_fraction_eps_rel(params),
    σ_abs = CAP.cloud_fraction_sigma_abs(params),
    s_ice_abs = CAP.cloud_fraction_ice_sigma_abs_scale(params),
    s_ice_rel = CAP.cloud_fraction_ice_eps_rel_scale(params),
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
    _compute_sgs_moments(
        thp, ρ, T, q_tot, q_c, sgs_quad, T′T′, q′q′, corr_Tq, α,
        q_tot_cloud = q_tot, lin = 0, curv = 0,
    )

Single quadrature pass returning `(sigma_S, λ_lagrange, λ_eq)`:

  - `sigma_S = sqrt(Σᵢ wᵢ·S′ᵢ²)`: SGS standard deviation of the sampled
    centred excess, clipped at `ϵ_numerics(FT)`; for a `ConditionalMomentSGS`,
    the linearized `_sgs_linear_sigma_S`, without a quadrature pass.
  - `λ_lagrange`: Lagrange multiplier satisfying the mass-conservation
    constraint `Σᵢ wᵢ·max(0, λ + α·S′ᵢ) = q_c` exactly under the discrete
    quadrature measure — the same points and weights the microphysics
    evaluator integrates over (see `_fit_discrete_lagrange`; the analytic
    truncated-Gaussian inverse `_compute_z` provides the seed).
  - `λ_eq`: Lagrange multiplier satisfying the equilibrium condensate
    constraint `Σᵢ wᵢ·max(0, λ_eq + S′ᵢ) = q_c_eq` under the same discrete
    quadrature measure, with the unattenuated node fluctuations `S′ᵢ`: the
    fidelity `α` attenuates the actual condensate, which lags equilibrium,
    while the equilibrium condensate has the full fluctuation of its
    continuous target. Here,
    `q_c_eq = truncated_gaussian_condensate(μ_{S,cloud}, σ_S)` is the
    expected saturation excess of the continuous Gaussian PDF with
    cloud-available mean excess `μ_{S,cloud} = q_tot_cloud − q_sat(T, ρ)`
    (`q_tot_cloud = q_tot − q_rai − q_sno`), or, with nonzero displacement
    coefficients `(lin, curv)` (`displacement_curvature_coefficients`), the
    curvature-aware expectation `displacement_condensate(μ_{S,cloud}, σ_S, lin, curv)`.
    Used by `Microphysics1MEvaluator` so that condensation and evaporation
    relax toward `q_c_eq` without the discrete 3-point Gauss–Hermite support
    cutoff at `μ_S < −√3 σ_S`.

Both multipliers come from `_sgs_lagrange_multipliers`, which dispatches on
the distribution: the discrete fits above for the Gauss–Hermite sampled
distributions, the continuous multipliers for `ConditionalMomentSGS`, whose
node rule integrates the condensates exactly.

The SGS mean `μ_S = q_tot − q_sat(T, ρ)` is analytic under the closure's
linearization (see `_sgs_saturation_moments`) and is recomputed on demand
wherever it is needed downstream.

Without SGS sampling (`nothing`, `GridMeanSGS`), all mass sits at the mean:
`sigma_S = ϵ_numerics(FT)`, `λ_lagrange = q_c`, and `λ_eq = max(0, μ_{S,cloud})`
directly, matching the σ_S → 0 limit of the sampled branch.
"""
@inline function _compute_sgs_moments(
    thp, ρ, T, q_tot, q_c,
    sgs_quad, T′T′, q′q′, corr_Tq, α,
    q_tot_cloud = q_tot,
    lin = zero(ρ),
    curv = zero(ρ),
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

    mu_S = q_tot - q_sat
    (sigma_S, S′s, ws) = _sgs_excess_samples(
        sgs_quad, thp, ρ, T, q_tot, mu_S, T′T′, q′q′, corr_Tq,
    )

    # Equilibrium condensate `q_c_eq = E[max(0, S_cloud)]`: truncated Gaussian,
    # or the curvature-aware displacement expectation where `curv ≠ 0`.
    q_c_eq = displacement_condensate(mu_S_cloud, sigma_S, lin, curv)
    (λ_lagrange, λ_eq) = _sgs_lagrange_multipliers(
        sgs_quad.dist, q_c, α, sigma_S, mu_S_cloud, q_c_eq, curv, S′s, ws,
    )
    return (; sigma_S, λ_lagrange, λ_eq)
end

# `(σ_S, S′s, ws)` of `_compute_sgs_moments`: the centred excesses `S′s` at the
# quadrature points with their probability weights `ws`, which the discrete
# Lagrange fits need, and `σ_S` from them; `ConditionalMomentSGS` takes the
# linearized `σ_S` and needs no samples.
@inline function _sgs_excess_samples(
    sgs_quad, thp, ρ, T, q_tot, mu_S, T′T′, q′q′, corr_Tq,
)
    FT = typeof(ρ)
    transform =
        build_physical_transform(sgs_quad, q_tot, T, q′q′, T′T′, corr_Tq)
    S′s = quadrature_point_values(
        SGSExcessEvaluator(thp, ρ, mu_S), transform, sgs_quad,
    )
    ws = quadrature_prob_weights(sgs_quad)
    sigma_S = max(sqrt(sum(ws .* S′s .* S′s)), ϵ_numerics(FT))
    return (sigma_S, S′s, ws)
end
@inline _sgs_excess_samples(
    ::ConditionalMomentQuadrature, thp, ρ, T, q_tot, mu_S, T′T′, q′q′, corr_Tq,
) = (_sgs_linear_sigma_S(thp, ρ, T, T′T′, q′q′, corr_Tq), nothing, nothing)

"""
    _sgs_lagrange_multipliers(dist, q_c, α, sigma_S, mu_S_cloud, q_c_eq, curv, S′s, ws)

The Lagrange multipliers `(λ_lagrange, λ_eq)` of `_compute_sgs_moments` for the
SGS distribution `dist`.

For the Gauss–Hermite sampled distributions both are fitted to the discrete
quadrature measure (`_fit_discrete_lagrange`), `λ_lagrange` from the analytic
truncated-Gaussian seed (`_compute_z`, with `σ_S_eff = α σ_S` floored so the
seed stays finite where `q_c` and `α` both vanish) and `λ_eq` with the
unattenuated `S′ᵢ`, the fluctuations `q_c_eq` is evaluated with.

For `ConditionalMomentSGS` the node rule integrates the piecewise-linear
condensates exactly, so both are the continuous multipliers:
`λ_lagrange` solves `E[max(0, λ + α S′)] = q_c` under the Gaussian
(`_continuous_lagrange`), and `λ_eq = μ_{S,cloud}` is the mean cloud excess
itself, whose truncated-Gaussian expectation is `q_c_eq`; with curvature,
`λ_eq` is the mean of the Gaussian whose expectation matches the
curvature-aware `q_c_eq`.
"""
@inline function _sgs_lagrange_multipliers(
    ::AbstractSGSDistribution, q_c, α, sigma_S, mu_S_cloud, q_c_eq, curv, S′s, ws,
)
    FT = typeof(q_c)
    σ_S_eff = max(α * sigma_S, ϵ_numerics(FT))
    λ0 = _compute_z(q_c / σ_S_eff) * σ_S_eff
    λ_lagrange = _fit_discrete_lagrange(λ0, q_c, α, S′s, ws)
    λ_eq = _fit_discrete_lagrange(mu_S_cloud, q_c_eq, one(FT), S′s, ws)
    return (λ_lagrange, λ_eq)
end

@inline function _sgs_lagrange_multipliers(
    ::ConditionalMomentSGS, q_c, α, sigma_S, mu_S_cloud, q_c_eq, curv, S′s, ws,
)
    FT = typeof(q_c)
    σ_S_eff = max(α * sigma_S, ϵ_numerics(FT))
    λ_lagrange = _continuous_lagrange(q_c, σ_S_eff)
    λ_eq = ifelse(
        curv == zero(FT),
        mu_S_cloud,
        _continuous_lagrange(q_c_eq, sigma_S),
    )
    return (λ_lagrange, λ_eq)
end

"""
    set_sgs_moments_and_cloud_fraction!(Y, p)

Final post-Aitken update. No-op when `ᶜsgs_moments` is not allocated (dry / 0M).

Uses ONE quadrature pass via `_compute_sgs_moments` to fill
`ᶜsgs_moments = (sigma_S, λ_lagrange, λ_eq)`, then computes
`ᶜcloud_fraction` consistently with the augmented `σ_aug` closure (see
`_compute_cloud_fraction`) from the environment cloud condensate
(`_get_condensate_means`) and applies EDMF updraft weighting. Finally
runs `set_precip_fraction!`, the column overlap sweep that turns the fresh
cover into the precipitation fraction `ᶜprecip_frac`.

Overwrites `p.scratch.ᶜtemp_scalar`, `ᶜtemp_scalar_2`, `ᶜtemp_scalar_3`, and
`ᶜtemp_scalar_5`. It must not touch `ᶜtemp_scalar_4` or `ᶜtemp_scalar_7`, which
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
    ᶜq_rai_lazy, ᶜq_sno_lazy =
        _get_precip_means(Y, p, turbconv_model, microphysics_model)
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
    ᶜq_cloud = p.scratch.ᶜtemp_scalar_5
    @. ᶜq_cloud = max(0, ᶜq_mean - (ᶜq_rai_lazy + ᶜq_sno_lazy))

    # The displacement coefficients enter the loop as a field only when the
    # curvature-aware condensate is configured; otherwise as zeros, for which
    # `displacement_condensate` is the truncated-Gaussian expectation.
    FT = eltype(p.params)
    # Carried in by type, so the parameters fold to literals in the kernel.
    vthermo = _fold(ClimaComms.device(Y.c), thermo_params)
    if p.atmos.sgs_variance_curvature
        ᶜdisp_S = p.precomputed.ᶜdisp_S
        DataLayouts.foreach_point(
            ᶜsgs_moments, ᶜcloud_fraction, ᶜρ_env, ᶜT_mean, ᶜq_mean,
            ᶜq_cloud, ᶜq_lcl, ᶜq_icl, ᶜT′T′, ᶜq′q′, ᶜcorr_Tq, ᶜα, ᶜdisp_S,
        ) do ᶜsgs_moments, ᶜcloud_fraction, ᶜρ_env, ᶜT_mean, ᶜq_mean,
        ᶜq_cloud, ᶜq_lcl, ᶜq_icl, ᶜT′T′, ᶜq′q′, ᶜcorr_Tq, ᶜα, ᶜdisp_S

            _sgs_moments_and_cloud_fraction_point!(
                ᶜsgs_moments, ᶜcloud_fraction, ᶜρ_env, ᶜT_mean, ᶜq_mean,
                ᶜq_cloud, ᶜq_lcl, ᶜq_icl, ᶜT′T′, ᶜq′q′, ᶜcorr_Tq, ᶜα,
                ᶜdisp_S.lin, ᶜdisp_S.curv, vthermo, thermo_params, sgs_quad, floor,
            )
        end
    else
        # The closure captures the zero coefficients as values: a captured
        # `FT` is a `Type`, which is not isbits and cannot enter the GPU kernel.
        lin₀ = zero(FT)
        curv₀ = zero(FT)
        DataLayouts.foreach_point(
            ᶜsgs_moments, ᶜcloud_fraction, ᶜρ_env, ᶜT_mean, ᶜq_mean,
            ᶜq_cloud, ᶜq_lcl, ᶜq_icl, ᶜT′T′, ᶜq′q′, ᶜcorr_Tq, ᶜα,
        ) do ᶜsgs_moments, ᶜcloud_fraction, ᶜρ_env, ᶜT_mean, ᶜq_mean,
        ᶜq_cloud, ᶜq_lcl, ᶜq_icl, ᶜT′T′, ᶜq′q′, ᶜcorr_Tq, ᶜα

            _sgs_moments_and_cloud_fraction_point!(
                ᶜsgs_moments, ᶜcloud_fraction, ᶜρ_env, ᶜT_mean, ᶜq_mean,
                ᶜq_cloud, ᶜq_lcl, ᶜq_icl, ᶜT′T′, ᶜq′q′, ᶜcorr_Tq, ᶜα,
                lin₀, curv₀, vthermo, thermo_params, sgs_quad, floor,
            )
        end
    end

    _apply_edmf_cloud_weighting!(Y, p, turbconv_model, thermo_params)
    set_precip_fraction!(Y, p)
    return nothing
end

"""
    _sgs_moments_and_cloud_fraction_point!(
        ᶜsgs_moments, ᶜcloud_fraction, ᶜρ_env, ᶜT_mean, ᶜq_mean, ᶜq_cloud,
        ᶜq_lcl, ᶜq_icl, ᶜT′T′, ᶜq′q′, ᶜcorr_Tq, ᶜα, lin, curv,
        vthermo, thermo_params, sgs_quad, floor,
    )

Body of the `foreach_point` loop of `set_sgs_moments_and_cloud_fraction!`:
fill the SGS moments (`_compute_sgs_moments`) and the cloud fraction
(`_compute_cloud_fraction`) of one point. `lin` and `curv` are the
displacement coefficients, a point of `ᶜdisp_S` or zeros. `vthermo` carries the
thermodynamics parameters in its type, so they fold to literals in the quadrature
kernel; `thermo_params` is the same parameters as a value, for the saturation
calls beside it.
"""
@inline function _sgs_moments_and_cloud_fraction_point!(
    ᶜsgs_moments, ᶜcloud_fraction, ᶜρ_env, ᶜT_mean, ᶜq_mean, ᶜq_cloud,
    ᶜq_lcl, ᶜq_icl, ᶜT′T′, ᶜq′q′, ᶜcorr_Tq, ᶜα, lin, curv,
    vthermo, thermo_params, sgs_quad, floor,
)
    @. ᶜsgs_moments = _compute_sgs_moments(
        vthermo, ᶜρ_env, ᶜT_mean, ᶜq_mean, ᶜq_lcl + ᶜq_icl,
        $(sgs_quad), ᶜT′T′, ᶜq′q′, ᶜcorr_Tq, ᶜα, ᶜq_cloud, lin, curv,
    )
    # Recompute CF from q_c and σ_S using the augmented-σ closure. `Φ(λ/σ_aug)`
    # is not usable because λ was computed with the equilibrium σ_S_eff, not
    # σ_aug, so it would not match the truncated-Gaussian closure for the
    # augmented variance. This overwrites the Picard iterate
    # with a value consistent with the final SGS moments, so EDMF weighting
    # must be re-applied by the caller even though `set_cloud_fraction!`
    # already applied it during Picard.
    @. ᶜcloud_fraction = _compute_cloud_fraction(
        ᶜq_lcl + ᶜq_icl,
        ᶜq_cloud - TD.q_vap_saturation(thermo_params, ᶜT_mean, ᶜρ_env),
        ᶜsgs_moments.sigma_S,
        ᶜα,
        $(floor),
        TD.liquid_fraction_ramp(thermo_params, ᶜT_mean),
    )
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
    ᶜq_cloud = if microphysics_model isa NonEquilibriumMicrophysics
        ᶜq_rai_lazy, ᶜq_sno_lazy =
            _get_precip_means(Y, p, turbconv_model, microphysics_model)
        @. p.scratch.ᶜtemp_scalar_5 =
            max(0, ᶜq_mean - (ᶜq_rai_lazy + ᶜq_sno_lazy))
    else
        ᶜq_mean
    end

    sgs_quad = p.atmos.sgs_quadrature
    floor = cloud_fraction_floor_params(p.params)

    (; ᶜT′T′, ᶜq′q′, ᶜcorr_Tq, ᶜα) = p.precomputed
    ᶜcloud_fraction = p.precomputed.ᶜcloud_fraction

    # Carried in by type, so the parameters fold to literals in the kernel.
    vthermo = _fold(ClimaComms.device(Y.c), thermo_params)

    DataLayouts.foreach_point(
        ᶜcloud_fraction, ᶜT_mean, ᶜρ_env, ᶜq_cloud, ᶜq_lcl, ᶜq_icl,
        ᶜT′T′, ᶜq′q′, ᶜcorr_Tq, ᶜα,
    ) do ᶜcloud_fraction, ᶜT_mean, ᶜρ_env, ᶜq_cloud, ᶜq_lcl, ᶜq_icl,
    ᶜT′T′, ᶜq′q′, ᶜcorr_Tq, ᶜα

        @. ᶜcloud_fraction = _compute_cloud_fraction(
            $(vthermo), ᶜT_mean, ᶜρ_env, ᶜq_cloud, ᶜq_lcl, ᶜq_icl,
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
    _get_precip_means(Y, p, turbconv_model, microphysics_model)

Non-negative mean rain and snow `(ᶜq_rai, ᶜq_sno)` of the domain that carries
the SGS cloud closure, as lazy fields: the environment for PrognosticEDMFX,
the grid mean otherwise. Only non-equilibrium microphysics carries `ρq_rai`
and `ρq_sno`, and only those configurations allocate `ᶜsgs_moments`. The
clamp matches the `q_rai_nonneg`, `q_sno_nonneg` that
`microphysics_tendencies_1m` hands to `Microphysics1MEvaluator`, so the
cloud-available excess `μ_{S,cloud}` of `_compute_sgs_moments` and the node
excess `S_raw_cloud` of the evaluator subtract the same precipitation.
"""
function _get_precip_means(Y, p, turbconv_model, ::NonEquilibriumMicrophysics)
    if turbconv_model isa PrognosticEDMFX
        return (
            (@. lazy(max(0, $(ᶜspecific_env_value(@name(q_rai), Y, p))))),
            (@. lazy(max(0, $(ᶜspecific_env_value(@name(q_sno), Y, p))))),
        )
    else
        return (
            (@. lazy(max(0, specific(Y.c.ρq_rai, Y.c.ρ)))),
            (@. lazy(max(0, specific(Y.c.ρq_sno, Y.c.ρ)))),
        )
    end
end

"""
    _apply_edmf_cloud_weighting!(Y, p, turbconv_model, thermo_params)

Apply EDMF-specific adjustments to cloud diagnostics.

For PrognosticEDMFX:

 1. Weights environment cloud diagnostics by environment area fraction
 2. Adds updraft contributions weighted by their respective area fractions

Under equilibrium microphysics, updraft cloud fraction is binary (`1` where
condensate is present, `0` otherwise). Under non-equilibrium microphysics,
`_updraft_cloud_fraction` evaluates the truncated-Gaussian cover of the updraft
condensate against the updraft vapor subsaturation deficit floored by `σ_abs`,
so a saturated cloud updraft contributes its full area while trace condensate
entrained or diffused into a subsaturated sub-cloud updraft does not make the
sub-cloud plume overcast.
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
        (; ᶜρʲs, ᶜTʲs, ᶜq_tot_nonnegʲs, ᶜq_liqʲs, ᶜq_iceʲs) = p.precomputed
        microphysics_model = p.atmos.microphysics_model
        σ_abs = CAP.cloud_fraction_sigma_abs(p.params)
        for j in 1:n
            ᶜρaʲ = Y.c.sgsʲs.:($j).ρa
            ᶜq_liqʲ, ᶜq_iceʲ =
                _updraft_cloud_condensate(Y, p, j, microphysics_model)

            @. p.precomputed.ᶜcloud_fraction +=
                draft_area(ᶜρaʲ, ᶜρʲs.:($$j)) * _updraft_cloud_fraction(
                    thermo_params,
                    ᶜTʲs.:($$j),
                    ᶜρʲs.:($$j),
                    ᶜq_tot_nonnegʲs.:($$j),
                    ᶜq_liqʲs.:($$j),
                    ᶜq_iceʲs.:($$j),
                    ᶜq_liqʲ,
                    ᶜq_iceʲ,
                    σ_abs,
                    microphysics_model,
                )
        end
    end
end

"""
    _updraft_cloud_fraction(
        thermo_params, Tʲ, ρʲ, q_totʲ, q_liq_totʲ, q_ice_totʲ,
        q_lclʲ, q_iclʲ, σ_abs, microphysics_model,
    )

Cloud fraction within updraft `j` `[-]`.

For equilibrium microphysics, returns `1` where `TD.has_condensate` holds for
`q_lclʲ + q_iclʲ` and `0` otherwise. For non-equilibrium microphysics,
evaluates the truncated-Gaussian cover `Φ(_compute_z(q_cʲ / √(δ_subʲ² + σ_abs²)))`
with `q_cʲ = max(0, q_lclʲ + q_iclʲ)` and the updraft vapor subsaturation
deficit `δ_subʲ = max(0, q_sat(Tʲ, ρʲ, q_lclʲ, q_iclʲ) - q_vapʲ)`. Above the
lifting condensation level (`δ_subʲ = 0`, `q_cʲ ≫ σ_abs`), the updraft is
fully cloudy (`1`); below the lifting condensation level (`δ_subʲ ≫ q_cʲ`),
trace entrained or diffused condensate gives vanishing cover.
"""
@inline function _updraft_cloud_fraction(
    thermo_params,
    Tʲ,
    ρʲ,
    q_totʲ,
    q_liq_totʲ,
    q_ice_totʲ,
    q_lclʲ,
    q_iclʲ,
    σ_abs,
    ::NonEquilibriumMicrophysics,
)
    FT = typeof(Tʲ)
    q_cʲ = max(zero(FT), q_lclʲ + q_iclʲ)
    TD.has_condensate(thermo_params, q_cʲ) || return zero(FT)
    q_vapʲ = max(
        zero(FT),
        q_totʲ - max(zero(FT), q_liq_totʲ) - max(zero(FT), q_ice_totʲ),
    )
    q_satʲ = TD.q_vap_saturation(
        thermo_params,
        Tʲ,
        ρʲ,
        max(zero(FT), q_lclʲ),
        max(zero(FT), q_iclʲ),
    )
    δ_subʲ = max(zero(FT), q_satʲ - q_vapʲ)
    σ_upʲ = sqrt(δ_subʲ^2 + FT(σ_abs)^2)
    return normal_cdf(_compute_z(q_cʲ / σ_upʲ))
end
@inline function _updraft_cloud_fraction(
    thermo_params,
    Tʲ,
    ρʲ,
    q_totʲ,
    q_liq_totʲ,
    q_ice_totʲ,
    q_lclʲ,
    q_iclʲ,
    σ_abs,
    microphysics_model,
)
    FT = typeof(Tʲ)
    return ifelse(
        TD.has_condensate(thermo_params, max(zero(FT), q_lclʲ + q_iclʲ)),
        one(FT),
        zero(FT),
    )
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
