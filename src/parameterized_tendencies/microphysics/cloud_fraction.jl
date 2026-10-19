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
vertical motions with buoyancy-weighted kinetic energy `κ_b = c_b κ_min` in
stably stratified air (`N² > 0`), capped by the wall distance `l_z` and by
the cell depth `Δz`, over which the gradients the displacement acts on are
defined, and zero where the air is not stably stratified (`N² ≤ 0`). The
displacement lasts until the parcel's vertical velocity decorrelates: after a
time `1/N`, when buoyancy stops it, which gives the buoyancy scale
`√κ_b / N` of the mixing length; after the Lagrangian time scale of the
turbulence `τ_turb = l_mix / √(c_b κ_iso)` (`turbulent_lagrangian_time`); or in
weakly stratified quiescent air after `τ_0`. This is the vertical dispersion of
a stochastic Lagrangian model in stably stratified flow.
"""
@inline function floor_displacement_length(κ_b, N², τ_turb, τ_0, l_z, Δz)
    FT = typeof(N²)
    l_cap = min(max(l_z, zero(FT)), Δz)
    τ = min(τ_0, τ_turb, inv(sqrt(max(N², floatmin(FT)))))
    return ifelse(N² > zero(FT), min(sqrt(κ_b) * τ, l_cap), zero(FT))
end

"""
    turbulent_lagrangian_time(l_mix, κ_b)

Lagrangian time scale `l_mix / √κ_b` of the turbulence, with buoyancy-weighted
vertical velocity variance `κ_b = c_b κ_iso`; infinite where there is no TKE
(`κ_b ≤ 0`), so that it imposes no limit there. Where `l_mix` is at the
buoyancy scale `√κ_b / N`, `turbulent_lagrangian_time` equals `1 / N`, matching
the buoyancy period of `floor_displacement_length`.
"""
@inline function turbulent_lagrangian_time(l_mix, κ_b)
    FT = typeof(κ_b)
    return ifelse(κ_b > eps(FT), l_mix / sqrt(max(κ_b, eps(FT))), FT(Inf))
end

"""
    buoyancy_bounded_mixing_length(l_mix, κ_b, N²)

Turbulent mixing length `l_mix` bounded by the buoyancy scale `√(κ_b / N²)`
with `κ_b = c_b κ_iso` where `N² > 0` and `κ_b > 0`. Bounding `l_mix` with the
same face-maximum dry `N²` used by `floor_displacement_length` ensures that a
cell adjacent to a capping inversion does not displace parcels across the
inversion over a distance larger than its face buoyancy scale allows when the
Van Leer harmonic-mean center gradient vanishes on the neutral side of the cell.
"""
@inline function buoyancy_bounded_mixing_length(l_mix, κ_b, N²)
    FT = typeof(N²)
    return ifelse(
        (N² > eps(FT)) & (κ_b > eps(FT)),
        min(l_mix, sqrt(max(κ_b, zero(FT)) / max(N², eps(FT)))),
        l_mix,
    )
end

"""
    displacement_length(l_mix, l_b, l_z)

Vertical displacement length `min(√(l_mix² + l_b²), l_z)`: the mixing length
and the floored buoyancy displacement scale combined in quadrature, capped by
the wall distance `l_z`, which bounds each of them and must bound their
combination too, since air cannot be displaced below the ground.
"""
@inline displacement_length(l_mix, l_b, l_z) =
    min(sqrt(l_mix^2 + l_b^2), max(l_z, zero(l_z)))

"""
    boundary_layer_mixing_weight(l_mix, l_d, l_z, H_p, c_H)

Boundary-layer mixing weight `w_bl = max(w_turb, w_z) ∈ [0, 1]` that relaxes
the horizontal and vertical moisture displacement scales toward the
potential-temperature displacement scales in the planetary boundary layer:

    w_turb = min((l_mix / l_d)², 1),    w_z = exp(-l_z / (c_H H_p)),

from the 3D turbulent mixing weight `w_turb` (`0` when `l_d ≤ 0`) and the
surface-coupled scale-height taper `w_z` (`0` when `c_H H_p ≤ 0`), with wall
distance `l_z = max(z - z_sfc, 0)`, local pressure scale height
`H_p = R_d T / g`, and dimensionless scale-height fraction `c_H`
(`sgs_variance_moisture_bl_transition_scale`).
"""
@inline function boundary_layer_mixing_weight(l_mix, l_d, l_z, H_p, c_H)
    FT = typeof(l_d)
    w_turb = ifelse(
        l_d > eps(FT),
        min((max(l_mix, zero(FT)) / max(l_d, eps(FT)))^2, one(FT)),
        zero(FT),
    )
    h_bl = c_H * H_p
    w_z = ifelse(
        h_bl > eps(FT),
        exp(-max(l_z, zero(FT)) / max(h_bl, eps(FT))),
        zero(FT),
    )
    return max(w_turb, w_z)
end

"""
    moisture_scale_ratio(r_θ, w_bl)

Ratio `r_q = 1 - (1 - r_θ) w_bl` of the horizontal or vertical moisture
displacement scale to its free-tropospheric value, relaxing from `1` in the
free troposphere (`w_bl = 0`) to the potential-temperature scale ratio
`r_θ ∈ {r_{Δx,θ}, r_{v,θ}}` in the boundary layer (`w_bl = 1`), where
boundary-layer eddies mix moisture and potential temperature alike.
"""
@inline moisture_scale_ratio(r_θ, w_bl) =
    one(w_bl) - (one(w_bl) - r_θ) * w_bl

"""
    horizontal_moisture_scale_ratio(r_Δx_θ, l_mix, l_d, l_z = Inf, H_p = 0, c_H = 0)

Ratio `r_{h,q} = ℓ_{h,q} / (√c_g c_Δx Δx_h)` of the moisture horizontal
displacement scale to its free-tropospheric value `√c_g c_Δx Δx_h`, evaluated
from `moisture_scale_ratio` and `boundary_layer_mixing_weight`.
"""
@inline function horizontal_moisture_scale_ratio(
    r_Δx_θ,
    l_mix,
    l_d,
    l_z = oftype(l_d, Inf),
    H_p = zero(l_d),
    c_H = zero(l_d),
)
    w_bl = boundary_layer_mixing_weight(l_mix, l_d, l_z, H_p, c_H)
    return moisture_scale_ratio(r_Δx_θ, w_bl)
end

"""
    materialized_displacement_length!(Y, p)

Materialize the vertical displacement length `l_d` of the variance closure
into `p.scratch.ᶜtemp_scalar` and return it. Also refreshes `p.precomputed.ᶜl_mix`
for `AbstractEDMF` via `materialized_mixing_length!`.

`l_d` combines the buoyancy-bounded turbulent mixing length and the floor
displacement scale `floor_displacement_length` in quadrature, evaluated with the
face-maximum dry buoyancy gradient `N²_disp` (`set_displacement_buoyancy_gradient!`),
with `κ_min = w_min² / (2 c_σ c_b)`, the velocity scale `w_min` of the unresolved
vertical motions at the horizontal grid scale (`floor_velocity`;
`sgs_variance_vertical_velocity_floor`,
`sgs_variance_floor_reference_resolution`,
`sgs_variance_floor_resolution_exponent`) and the time over which their
displacement persists, the smallest of the buoyancy period `1/N_disp`, the
Lagrangian time scale of the turbulence `l_mix / √(c_b κ_iso)`
(`turbulent_lagrangian_time`) and the quiescent limit `τ_0`
(`sgs_variance_floor_time_scale`). In turbulent layers `l_mix` dominates and
the closure is the production–dissipation balance, bounded by the face dry
buoyancy scale `√(c_b κ_iso) / N_disp` (`buoyancy_bounded_mixing_length`). In
stably stratified air without TKE, `l_d = w_min / (√(2 c_σ) N_disp)`, so the
vertical displacement variance `2 c_σ l_d²` is `(w_min / N_disp)²`: the
displacement of unresolved vertical motions (gravity waves, mesoscale
circulations) with velocity scale `w_min`, which displace air without mixing it.
Where the air is not stably stratified (`N²_disp ≤ 0`) the floor displacement
vanishes, so `l_d` reduces to `l_mix`. The cell-depth cap applies to the floor
scale only; `l_mix` keeps its own limits, so `l_d` exceeds the cell depth
wherever the mixing length does, as the turbulent variance
`2 c_σ l_mix² (∂_z ψ)²` always has. The two scales are combined in quadrature
and the combination is capped by the wall distance (`displacement_length`).
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
    if hasproperty(Y.c, :ρtke)
        ᶜtke = @. lazy(max(specific(Y.c.ρtke, Y.c.ρ), zero(FT)))
        ᶜl_mix_disp = @. lazy(
            buoyancy_bounded_mixing_length(ᶜl_mix, c_b * ᶜtke, ᶜN²_disp),
        )
        ᶜτ_turb = @. lazy(turbulent_lagrangian_time(ᶜl_mix_disp, c_b * ᶜtke))
    else
        ᶜl_mix_disp = ᶜl_mix
        ᶜτ_turb = FT(Inf)
    end
    ᶜz = Fields.coordinate_field(Y.c).z
    z_sfc = Fields.level(Fields.coordinate_field(Y.f).z, Fields.half)
    ᶜΔz = Fields.Δz_field(axes(Y.c))
    ᶜl_d = p.scratch.ᶜtemp_scalar
    @. ᶜl_d = displacement_length(
        ᶜl_mix_disp,
        floor_displacement_length(
            c_b * κ_min, ᶜN²_disp, ᶜτ_turb, τ_0, ᶜz - z_sfc, ᶜΔz,
        ),
        ᶜz - z_sfc,
    )
    return ᶜl_d
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
    c_b = FT(CAP.static_stab_coeff(turbconv_params))
    # Non-EDMF fallbacks: TKE → 0 drives α_Da → 1. When turbulent mixing is
    # active, bound `ᶜl_mix` by the face-maximum dry buoyancy scale (as in
    # `materialized_displacement_length!`) so the eddy turnover rate
    # `c_d √TKE / l_mix` reflects the inversion-bounded eddy size.
    if hasproperty(Y.c, :ρtke) && hasproperty(p.precomputed, :ᶜl_mix)
        ᶜtke_or_zero = @. lazy(max(specific(Y.c.ρtke, Y.c.ρ), zero(FT)))
        if hasproperty(p.precomputed, :ᶜbg_coeffs)
            (; ᶜbg_coeffs, ᶠ∂θli∂z, ᶠ∂qt∂z) = p.precomputed
            ᶜN²_disp = p.scratch.ᶜtemp_scalar_2
            set_displacement_buoyancy_gradient!(
                ᶜN²_disp, ᶜbg_coeffs, ᶠ∂θli∂z, ᶠ∂qt∂z,
            )
            ᶜl_mix_or_one = @. lazy(
                buoyancy_bounded_mixing_length(
                    p.precomputed.ᶜl_mix,
                    c_b * ᶜtke_or_zero,
                    ᶜN²_disp,
                ),
            )
        else
            ᶜl_mix_or_one = p.precomputed.ᶜl_mix
        end
    else
        ᶜtke_or_zero =
            hasproperty(Y.c, :ρtke) ?
            (@. lazy(specific(Y.c.ρtke, Y.c.ρ))) : zero(FT)
        ᶜl_mix_or_one =
            hasproperty(p.precomputed, :ᶜl_mix) ? p.precomputed.ᶜl_mix : one(FT)
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
    displacement_covariance(ℓ_h², ℓ_v², inv_h, ∂ψ∂z, ∂φ∂z)

Covariance `ℓ_h² ∇_h ψ · ∇_h φ + ℓ_v² ∂_z ψ ∂_z φ` of two scalars under a
subgrid displacement with covariance `diag(ℓ_h², ℓ_h², ℓ_v²)`, from the
horizontal invariant `inv_h = ∇_h ψ · ∇_h φ` and the vertical gradients.
"""
@inline displacement_covariance(ℓ_h², ℓ_v², inv_h, ∂ψ∂z, ∂φ∂z) =
    ℓ_h² * inv_h + ℓ_v² * ∂ψ∂z * ∂φ∂z

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

In exact arithmetic the covariance comes from one displacement acting on
both gradients, so it satisfies the Cauchy-Schwarz inequality and
`|ρ| ≤ 1` by construction. The explicit clamp is a cheap safeguard against
Float32 roundoff at the Cauchy-Schwarz boundary (which can produce
`|ρ| = 1 + O(ε)` and feed NaN into downstream `√(1 − ρ²)` in the bivariate
quadrature) and a user-supplied `Tq_correlation_coefficient` outside `[-1, 1]`.
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

The covariance of `(θ_li, q_tot)` is that of a Gaussian subgrid displacement
acting on the resolved gradients (`displacement_covariance`), with
free-tropospheric moisture horizontal scale `ℓ_{h,0}² = c_g (c_Δx Δx_h)²` from
the horizontal grid scale, free-tropospheric moisture vertical displacement
scale `ℓ_{v,0}² = 2 c_σ l_d²` from the displacement length `l_d`
(`materialized_displacement_length!`), potential-temperature scale ratios
`r_{Δx,θ} = ℓ_{h,θ} / ℓ_{h,0}` (`sgs_variance_theta_horizontal_scale_ratio`) and
`r_{v,θ} = ℓ_{v,θ} / ℓ_{v,0}` (`sgs_variance_theta_vertical_scale_ratio`),
moisture scale ratios `r_{h,q} = ℓ_{h,q} / ℓ_{h,0}` and
`r_{v,q} = ℓ_{v,q} / ℓ_{v,0}` (`moisture_scale_ratio`, which relaxes `ℓ_{h,q}`
to `ℓ_{h,θ}` and `ℓ_{v,q}` to `ℓ_{v,θ}` in the boundary layer with weight
`boundary_layer_mixing_weight`), and horizontal displacement cross-correlation
`r_{h,θq}` (`sgs_variance_horizontal_cross_correlation`).

Pipeline:

 1. Materialize `l_d` (`materialized_displacement_length!`, which also refreshes
    `ᶜl_mix` via `materialized_mixing_length!`).
 2. Horizontal gradient invariants of `θ_li` and `q_tot` (skipped when
    `sgs_variance_horizontal_scale_factor` is 0).
 3. θ′θ′, q′q′ and, for the diagnosed correlation, θ′q′ from the displacement.
 4. Transform θ→T using `compute_∂T_∂θ!`.
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
    c_H = FT(CAP.sgs_variance_moisture_bl_transition_scale(p.params))
    R_d = FT(TD.Parameters.R_d(thermo_params))
    grav = FT(TD.Parameters.grav(thermo_params))
    (; ᶜT) = p.precomputed
    ᶜz = Fields.coordinate_field(Y.c).z
    z_sfc = Fields.level(Fields.coordinate_field(Y.f).z, Fields.half)
    Δx_h = FT(
        Spaces.node_horizontal_length_scale(
            Spaces.horizontal_space(axes(Y.c)),
        ),
    )
    ℓ_h_0² = c_g * (c_Δx * Δx_h)^2
    ℓ_h_θ² = r_Δx_θ^2 * ℓ_h_0²
    c_σ = FT(CAP.diagnostic_covariance_coeff(p.params))
    ᶜl_d = materialized_displacement_length!(Y, p)
    if hasproperty(Y.c, :ρtke) && hasproperty(p.precomputed, :ᶜl_mix)
        turbconv_params = CAP.turbconv_params(p.params)
        c_b = FT(CAP.static_stab_coeff(turbconv_params))
        ᶜtke = @. lazy(max(specific(Y.c.ρtke, Y.c.ρ), zero(FT)))
        ᶜN²_disp = p.scratch.ᶜtemp_scalar_2
        ᶜl_mix_disp = @. lazy(
            buoyancy_bounded_mixing_length(
                p.precomputed.ᶜl_mix,
                c_b * ᶜtke,
                ᶜN²_disp,
            ),
        )
    else
        ᶜl_mix_disp = zero(FT)
    end
    ᶜw_bl = @. lazy(
        boundary_layer_mixing_weight(
            ᶜl_mix_disp,
            ᶜl_d,
            ᶜz - z_sfc,
            R_d * ᶜT / grav,
            c_H,
        ),
    )
    ᶜr_h_q = @. lazy(moisture_scale_ratio(r_Δx_θ, ᶜw_bl))
    ᶜr_v_q = @. lazy(moisture_scale_ratio(r_v_θ, ᶜw_bl))
    ᶜℓ_h_q² = @. lazy(ᶜr_h_q^2 * ℓ_h_0²)
    ᶜℓ_h_θq = @. lazy(r_h_θq * r_Δx_θ * ᶜr_h_q * ℓ_h_0²)
    ᶜℓ_v_0² = @. lazy(2 * c_σ * ᶜl_d^2)
    ᶜℓ_v_q² = @. lazy(ᶜr_v_q^2 * ᶜℓ_v_0²)
    ᶜℓ_v_θ² = @. lazy(r_v_θ^2 * ᶜℓ_v_0²)
    ᶜℓ_v_θq = @. lazy(r_v_θ * ᶜr_v_q * ᶜℓ_v_0²)

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
    @. ᶜq′q′ = displacement_covariance(ᶜℓ_h_q², ᶜℓ_v_q², ᶜinv_q, ᶜ∂q∂z, ᶜ∂q∂z)
    @. ᶜT′T′ = displacement_covariance(ℓ_h_θ², ᶜℓ_v_θ², ᶜinv_θ, ᶜ∂θ∂z, ᶜ∂θ∂z)
    if diag_corr
        @. p.precomputed.ᶜT′q′ =
            displacement_covariance(ᶜℓ_h_θq, ᶜℓ_v_θq, ᶜinv_θq, ᶜ∂θ∂z, ᶜ∂q∂z)
    end

    # 4. θ → T basis via the thermodynamic Jacobian ∂T/∂θ. q is unaffected.
    ᶜ∂T_∂θ = p.scratch.ᶜtemp_scalar_3
    compute_∂T_∂θ!(ᶜ∂T_∂θ, Y, p, thermo_params)
    @. ᶜT′T′ = ᶜ∂T_∂θ^2 * ᶜT′T′
    if diag_corr
        @. p.precomputed.ᶜT′q′ = ᶜ∂T_∂θ * p.precomputed.ᶜT′q′
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

    # 7. Refresh `ᶜα` with the current `ᶜl_mix` and state.
    materialized_variance_fidelity!(Y, p)
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
    c_qc =
        hasproperty(floor, :condensate_release) ? floor.condensate_release :
        zero(FT)
    # Release the relative floor where the mean saturation excess `mu_S` or
    # condensate margin `c_qc * q_c` exceeds the release width `w` — by default
    # the *equilibrium* PDF width, so the floor is released where the unfloored
    # closure already predicts overcast. The denominator guard covers only the
    # w → 0 limit: the smallness of the equilibrium width relative to
    # ε_rel·q_sat is what drives the release in a quiescent deck, so it must
    # not be floored away.
    w = sqrt(
        margin^2 * ((α * sigma_S)^2 + σ_abs^2) +
        (abs_margin * ε_rel * q_sat)^2,
    )
    x = max(mu_S, c_qc * q_c, zero(FT)) / max(w, ϵ_numerics(FT))
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
  - `margin`, `abs_margin`, `sharpness`, `residual`, `condensate_release`:
    release shape (see the saturation-margin release section of
    `_compute_cloud_fraction`).
"""
Base.@kwdef struct CloudFractionFloorParams{FT}
    ε_rel::FT
    σ_abs::FT
    margin::FT
    abs_margin::FT
    sharpness::FT
    residual::FT
    condensate_release::FT = FT(0)
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
    condensate_release = CAP.cloud_fraction_floor_condensate_release(params),
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
