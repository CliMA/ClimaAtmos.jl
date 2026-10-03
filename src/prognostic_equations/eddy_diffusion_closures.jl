#####
##### TKE-based eddy diffusion closures
#####

import StaticArrays as SA
import Thermodynamics.Parameters as TDP
import ClimaCore.Geometry as Geometry
import ClimaCore.Fields as Fields
import SurfaceFluxes.UniversalFunctions as UF

"""
    buoyancy_gradient_coefficients(thermo_params, T, ρ, q_tot, q_liq, q_ice)

Return the pointwise thermodynamic coefficients of the moist buoyancy-gradient
chain rule.

The buoyancy gradient is *linear* in the vertical gradients of the
prognostic state,

    ∂b/∂z = C_θ(state, cf) ∂θli/∂z + C_q(state, cf) ∂qt/∂z,

with the cloud-fraction blend also linear:
`C_θ = Cθ_unsat + cf ΔCθ`, `C_q = Cq_unsat + cf ΔCq`.

The coefficients contain all of the expensive pointwise thermodynamics
(saturation vapor pressure, latent heat, potential temperatures); evaluating
them once per state update and reusing them for the centered, one-sided, and
face-native gradient stencils — via `blended_N²` — avoids recomputing
that thermodynamics for every stencil.

# Arguments

  - `thermo_params`: Thermodynamic parameters.
  - `T`: Air temperature [K].
  - `ρ`: Air density [kg/m³].
  - `q_tot`, `q_liq`, `q_ice`: Total, liquid, and ice specific humidities
    [kg/kg].

# Returns

The `NamedTuple` `(; Cθ_unsat, ΔCθ, Cq_unsat, ΔCq)` of cloud-fraction-
independent coefficients: `∂b/∂θ_li` in unsaturated air [m/s²/K] with its
saturated increment `ΔCθ`, and `∂b/∂q_tot` in unsaturated air [m/s²] with its
saturated increment `ΔCq`.
"""
@inline function buoyancy_gradient_coefficients(
    thermo_params,
    T,
    ρ,
    q_tot,
    q_liq,
    q_ice,
)
    g = TDP.grav(thermo_params)
    Rv_over_Rd = TDP.Rv_over_Rd(thermo_params)
    R_v = TDP.R_v(thermo_params)

    ∂b∂θv = g / TD.virtual_pottemp(thermo_params, T, ρ, q_tot, q_liq, q_ice)

    lh = TD.latent_heat(thermo_params, T, q_liq, q_ice)
    cp_m = TD.cp_m(thermo_params, q_tot, q_liq, q_ice)
    q_sat = TD.q_vap_saturation(thermo_params, T, ρ, q_liq, q_ice)
    θ = TD.potential_temperature(thermo_params, T, ρ, q_tot, q_liq, q_ice)
    ∂b∂θli_unsat = ∂b∂θv * (1 + (Rv_over_Rd - 1) * q_tot)
    ∂b∂qt_unsat = ∂b∂θv * (Rv_over_Rd - 1) * θ
    ∂b∂θli_sat = (
        ∂b∂θv *
        (1 + Rv_over_Rd * (1 + lh / R_v / T) * q_sat - q_tot) /
        (1 + lh^2 / cp_m / R_v / T^2 * q_sat)
    )
    ∂b∂qt_sat = (lh / cp_m / T * ∂b∂θli_sat - ∂b∂θv) * θ

    return (;
        Cθ_unsat = ∂b∂θli_unsat,
        ΔCθ = ∂b∂θli_sat - ∂b∂θli_unsat,
        Cq_unsat = ∂b∂qt_unsat,
        ΔCq = ∂b∂qt_sat - ∂b∂qt_unsat,
    )
end

"""
    blended_N²(coeffs, cf, ∂θli∂z, ∂qt∂z)

Return the moist buoyancy gradient `N² = ∂b/∂z` [1/s²] from precomputed
chain-rule coefficients (`buoyancy_gradient_coefficients`), the local
cloud fraction `cf` [-], and the projected vertical gradients of `θ_li` [K/m]
and `q_tot` [1/m], for physical scalars:

    ∂b/∂z = (Cθ_unsat + cf ΔCθ) ∂θli/∂z + (Cq_unsat + cf ΔCq) ∂qt/∂z.
"""
@inline blended_N²(coeffs, cf, ∂θli∂z, ∂qt∂z) =
    (coeffs.Cθ_unsat + cf * coeffs.ΔCθ) * ∂θli∂z +
    (coeffs.Cq_unsat + cf * coeffs.ΔCq) * ∂qt∂z

"""
    buoyancy_gradients(ebgc::AbstractEnvBuoyGradClosure, thermo_params,
                       bg_model::EnvBuoyGradVars)

Compute the mean vertical buoyancy gradient `∂b/∂z` [1/s²] in the environment
from the state and the vertical gradients `∂θₗᵢ/∂z`, `∂qₜ/∂z` bundled in
`bg_model`.

The gradient blends the unsaturated and saturated responses with the
environmental cloud fraction: `buoyancy_gradient_coefficients` supplies
the partial derivatives of buoyancy, and
`buoyancy_gradient_chain_rule` applies them to the gradients and
blends.

# Arguments

  - `ebgc`: Environmental buoyancy-gradient closure (e.g. `BuoyGradMean`).
  - `thermo_params`: Thermodynamic parameters.
  - `bg_model`: `EnvBuoyGradVars` bundling `T`, `ρ`, `q_tot`, `q_liq`, `q_ice`,
    `cf`, `∂qt∂z`, and `∂θli∂z`.

# Returns

The mean vertical buoyancy gradient [1/s²].

# Notes

The production pipeline evaluates the same quantity through the fused
`blended_N²` broadcast; this bundled form is retained as the reference
path exercised by the unit tests.
"""
function buoyancy_gradients(
    ebgc::AbstractEnvBuoyGradClosure,
    thermo_params,
    bg_model::EnvBuoyGradVars,
)
    (; T, ρ, q_tot, q_liq, q_ice) = bg_model
    coeffs = buoyancy_gradient_coefficients(
        thermo_params,
        T,
        ρ,
        q_tot,
        q_liq,
        q_ice,
    )
    ∂b∂z = buoyancy_gradient_chain_rule(
        ebgc,
        bg_model,
        thermo_params,
        coeffs.Cθ_unsat,
        coeffs.Cq_unsat,
        coeffs.Cθ_unsat + coeffs.ΔCθ,
        coeffs.Cq_unsat + coeffs.ΔCq,
    )
    return ∂b∂z
end

"""
    buoyancy_gradient_chain_rule(
        closure::AbstractEnvBuoyGradClosure,
        bg_model::EnvBuoyGradVars,
        thermo_params,
        ∂b∂θli_unsat,
        ∂b∂qt_unsat,
        ∂b∂θli_sat,
        ∂b∂qt_sat,
    )

Apply the buoyancy chain rule to the vertical gradients in `bg_model` and blend
the unsaturated and saturated results with the environmental cloud fraction.

Each pair of partial derivatives is contracted with `∂θli∂z` and `∂qt∂z` to give
`∂b∂z_unsat` and `∂b∂z_sat`, which are combined as
`(1 - cf) ∂b∂z_unsat + cf ∂b∂z_sat`.

# Arguments

  - `closure`: Environmental buoyancy-gradient closure.
  - `bg_model`: `EnvBuoyGradVars` supplying `∂θli∂z`, `∂qt∂z`, and `cf`.
  - `thermo_params`: Thermodynamic parameters (unused by the current method,
    retained for closures that need them).
  - `∂b∂θli_unsat`, `∂b∂θli_sat`: Partial derivatives of buoyancy with respect to
    liquid-ice potential temperature, unsaturated and saturated [m/s²/K].
  - `∂b∂qt_unsat`, `∂b∂qt_sat`: Partial derivatives of buoyancy with respect to
    total specific humidity, unsaturated and saturated [m/s²].

# Returns

The mean vertical buoyancy gradient [1/s²]. Called from
`buoyancy_gradients`.
"""
function buoyancy_gradient_chain_rule(
    ::AbstractEnvBuoyGradClosure,
    bg_model::EnvBuoyGradVars,
    thermo_params,
    ∂b∂θli_unsat,
    ∂b∂qt_unsat,
    ∂b∂θli_sat,
    ∂b∂qt_sat,
)
    ∂b∂z_θli_unsat = ∂b∂θli_unsat * bg_model.∂θli∂z
    ∂b∂z_qt_unsat = ∂b∂qt_unsat * bg_model.∂qt∂z
    ∂b∂z_unsat = ∂b∂z_θli_unsat + ∂b∂z_qt_unsat
    ∂b∂z_θl_sat = ∂b∂θli_sat * bg_model.∂θli∂z
    ∂b∂z_qt_sat = ∂b∂qt_sat * bg_model.∂qt∂z
    ∂b∂z_sat = ∂b∂z_θl_sat + ∂b∂z_qt_sat

    ∂b∂z = (1 - bg_model.cf) * ∂b∂z_unsat + bg_model.cf * ∂b∂z_sat

    return ∂b∂z
end

"""
    surface_flux_tke(turbconv_params, ρ_sfc, ustar, surface_local_geometry)

Return the surface flux of TKE as a `C3` vector for use in operator boundary
conditions.

The magnitude is `c_k ρ_sfc ustar³`, directed along the surface upward normal,
where `c_k` (`tke_surf_flux_coeff`) is a dimensionless coefficient. The `ustar³`
scaling makes the flux a shear-production input: the TKE generated at the
surface by unresolved roughness elements.

# Arguments

  - `turbconv_params`: Turbulence-convection parameters.
  - `ρ_sfc`: Air density at the surface [kg/m³]; currently the first
    cell-center density.
  - `ustar`: Friction velocity [m/s].
  - `surface_local_geometry`: `ClimaCore.Geometry.LocalGeometry` at the surface.

# Returns

A `ClimaCore.Geometry.C3` vector, the TKE flux normal to the surface
[kg/s³].
"""
function surface_flux_tke(
    turbconv_params,
    ρ_sfc,
    ustar,
    surface_local_geometry,
)

    c_k = CAP.tke_surf_flux_coeff(turbconv_params)
    # Determine the direction of the flux (normal to the surface)
    # c3_unit is a unit vector in the direction of the surface normal (e.g., C3(0,0,1) for a flat surface)
    c3_unit = C3(unit_basis_vector_data(C3, surface_local_geometry))
    return c_k * ρ_sfc * ustar^3 * c3_unit
end

"""
    mixing_length_lopez_gomez_2020(
        turbconv_params, sf_params, vkc, ustar, z, z_sfc, Δ_f, sfc_tke,
        N²_prod, tke, obukhov_length, strain_rate_norm, Pr,
        scale_blending_method,
    ) -> MixingLength

Compute the turbulent mixing length pointwise from a modified version of the
[Lopez2020](@cite) closure with an *empirical* TKE-scale form that unifies
the P = ε balance and the bounded amplitude limits.

Three physical scales are formed and blended by `blend_scales`:

  - `l_W`: wall scale `κ (z - z_sfc) ustar / (c_m √e_sfc φ_m(ζ))`, matching
    Monin-Obukhov similarity in the surface layer.
  - `l_TKE`: empirical scale `l_inf · √x · (1 + x) · exp(−x)` with
    `x = TKE / (l_inf / τ_ε)²` and reference eddy turnover time `τ_ε`. In the
    small-`x` limit this reduces to the Lopez-Gomez P = ε balance mixing length.
  - `l_N`: buoyancy-limited scale `√(c_b e) / N`, capped by the wall
    distance and used only where `N²_prod > 0`.

The blend is then limited by the wall distance and by the resolvability filter
scale `Δ_f`, and clamped to non-negative values.

The same closure is evaluated at cell centers (`ᶜmixing_length`) and at faces
(`set_face_diffusivities!`), with the corresponding inputs.

# Arguments

  - `turbconv_params`: Turbulence-convection parameters (`c_m`, `c_b`, `c_d`).
  - `sf_params`: Surface-flux parameters (Businger universal functions).
  - `vkc`: Von Kármán constant [-].
  - `ustar`: Friction velocity [m/s].
  - `z`: Height of the evaluation point [m].
  - `z_sfc`: Surface elevation [m].
  - `Δ_f`: Resolvability filter scale [m] that caps the mixing length (see
    `resolvability_filter_scale`; `Inf` where the grid imposes no scale,
    as in single columns).
  - `sfc_tke`: TKE near the surface (first cell center) [m²/s²].
  - `N²_prod`: Squared buoyancy frequency entering the production-dissipation
    balance for `l_TKE` and the buoyancy-limited length `l_N` [1/s²].
  - `tke`: Turbulent kinetic energy at the evaluation point [m²/s²].
  - `obukhov_length`: Surface Monin-Obukhov length [m].
  - `strain_rate_norm`: Squared Frobenius norm of the strain-rate tensor,
    `SᵢⱼSᵢⱼ` [1/s²].
  - `Pr`: Turbulent Prandtl number [-].
  - `scale_blending_method`: Blending method for the physical scales
    (`blend_scales`).

# Returns

A `MixingLength{FT}` holding the blended mixing length (`master`) and the
constituent scales `wall`, `tke`, `buoy`, and `l_grid`, all [m].
"""
function mixing_length_lopez_gomez_2020(
    turbconv_params,
    sf_params,
    vkc,
    ustar,
    z,
    z_sfc,
    Δ_f,
    sfc_tke,
    N²_prod,
    tke,
    obukhov_length,
    strain_rate_norm,
    Pr,
    scale_blending_method,
)

    FT = eltype(z)
    eps_FT = eps(FT)

    c_m = CAP.tke_ed_coeff(turbconv_params)
    c_d = tke_dissipation_coefficient(turbconv_params)
    c_b = CAP.static_stab_coeff(turbconv_params)
    c_tke = CAP.mixing_length_tke_coeff(turbconv_params)
    l_inf = CAP.mixing_length_tke_l_inf(turbconv_params)
    τ_ε_max = CAP.mixing_length_tke_tau_max(turbconv_params)

    # l_z: Geometric distance from the surface
    l_z = z - z_sfc
    # Ensure l_z is non-negative when z is numerically smaller than z_sfc.
    l_z = max(l_z, FT(0))

    # l_W: Wall-constrained length scale (near-surface limit, to match
    # Monin-Obukhov Similarity Theory in the surface layer, with Businger-Dyer
    # type stability functions)
    tke_sfc_safe = max(sfc_tke, eps_FT)
    ustar_sq_safe = max(ustar * ustar, eps_FT) # ustar^2 may vanish in certain LES setups

    # Denominator of the base length scale (always positive):
    #     c_m * √(tke_sfc / u_*²) = c_m * √(e_sfc) / u_*
    # The value increases when u_* is small and decreases when e_sfc is small.
    l_W_denom_factor = sqrt(tke_sfc_safe / ustar_sq_safe)
    l_W_denom = max(c_m * l_W_denom_factor, eps_FT)

    # Base length scale (neutral, but adjusted for TKE level)
    # l_W_base = κ * l_z / (c_m * sqrt(e_sfc) / u_star)
    # This can be Inf if l_W_denom is eps_FT and l_z is large.
    # This can be 0 if l_z is 0.
    # The expression approaches ∞ when l_W_denom ≈ eps_FT and l_z > eps_FT,
    # and approaches 0 when l_z → 0.
    l_W_base = vkc * l_z / l_W_denom

    obukhov_len_safe =
        obukhov_length < FT(0) ? min(obukhov_length, -eps_FT) : max(obukhov_length, eps_FT)
    zeta = l_z / obukhov_len_safe # Stability parameter zeta
    phi_m = UF.phi(sf_params.ufp, zeta, UF.MomentumTransport())
    l_W = l_W_base / max(phi_m, eps_FT)

    l_W = max(l_W, FT(0)) # Ensure non-negative

    # --- l_TKE: empirical mixing length ---
    tke_pos = max(tke, FT(0))
    # `a_pd = c_m·(2|S̃|² − N²/Pr)` is the shear+buoyancy production
    # coefficient; `√(c_d/a_pd)` is the eddy turnover time at P = ε balance.
    a_pd = c_m * (2 * strain_rate_norm - N²_prod / Pr)

    # Empirical form:
    #     l_TKE = l_inf · √x · (1+x)·exp(−x),   x = TKE / (l_inf/τ_ε)²,
    #     τ_ε = c_tke · √(c_d/a_pd)
    # `τ_ε` is the reference eddy turnover time: `c_tke` times the eddy
    # turnover time at P = ε balance.
    # For small x, (1+x)·exp(−x) → 1 and l_TKE → τ_ε · √TKE =
    # c_tke · √(c_d/a_pd) · √TKE — the Lopez-Gomez P = ε balance mixing length
    # scaled by `c_tke`. `x` is *adaptive*: stronger forcing (smaller τ_ε)
    # raises `(l_inf/τ_ε)²` so higher TKE is allowed before the decay
    # kicks in. Peak: l_TKE is maximized at x = 1 with max(l_TKE) = (2/e)·l_inf
    # ≈ 0.74·l_inf.
    # τ_ε is capped at `τ_ε_max`, the residual eddy turnover from
    # background processes; the cap is inert wherever local production
    # is strong (c_tke·√(c_d/a_pd) ≪ τ_ε_max).
    τ_ε = min(τ_ε_max, c_tke * sqrt(c_d / max(a_pd, eps_FT)))
    tke_nondim = tke_pos / (l_inf / τ_ε)^2
    l_TKE = l_inf * sqrt(tke_nondim) *
            (FT(1) + tke_nondim) * exp(-tke_nondim)

    # --- l_N: Static-stability length scale (buoyancy limit), constrained by l_z ---
    N_eff_sq = max(N²_prod, FT(0)) # Use N^2 only if stable (N^2 > 0)
    l_N = l_z # Default to wall distance if not stably stratified or TKE is zero
    # if N_eff_sq > eps_FT && tke_pos > eps_FT
    #     N_eff = sqrt(N_eff_sq)
    #     # l_N ~ sqrt(c_b * TKE) / N_eff
    #     l_N_physical = sqrt(c_b * tke_pos) / N_eff
    #     # Limit by distance from wall
    #     l_N = min(l_N_physical, l_z)
    # end
    l_N = max(l_N, FT(0)) # Ensure non-negative


    # --- Combine Scales ---

    # Vector of *physical* scales (wall, TKE, stability). All ≥ 0.
    # l_N is already limited by l_z; l_W and l_TKE are not, but the master
    # wall + grid caps below still apply.
    l_physical_scales = SA.SVector(l_W, l_TKE, l_N)

    l_smin =
        blend_scales(scale_blending_method, l_physical_scales, turbconv_params)

    # 1. Limit the combined physical scale by the distance from the wall.
    #    This step mitigates excessive values of l_W or l_TKE.
    l_limited_phys_wall = min(l_smin, l_z)

    # 2. Impose the resolvability filter scale (see
    #    resolvability_filter_scale for the rationale and regimes).
    l_grid = Δ_f
    l_final = min(l_limited_phys_wall, l_grid)

    # `l_final` is not floored here: leaving it at its physical value
    # (possibly zero) avoids introducing artificial diffusivities where
    # eddy activity is genuinely absent. The dissipation-rate consumers
    # (`tke_dissipation` in `edmfx_tke.jl`, and the implicit Jacobian
    # inline in `manual_sparse_jacobian.jl`) apply a small local floor
    # only where division-by-zero would matter.
    l_final = max(l_final, FT(0))    # ensure non-negative only

    return MixingLength(l_final, l_W, l_TKE, l_N, l_grid)
end

"""
    set_buoyancy_gradient_inputs!(Y, p, thermo_params)

Materialize, once per state update, everything the buoyancy-gradient stencils
share:

  - `p.precomputed.ᶜbg_coeffs`: the pointwise chain-rule coefficients of
    `buoyancy_gradient_coefficients` (all of the expensive
    saturation thermodynamics lives here);
  - `p.precomputed.ᶠ∂θli∂z`, `p.precomputed.ᶠ∂qt∂z`: exact two-point face
    gradients of `θ_li` and `q_tot`, projected to physical scalars;
  - `p.precomputed.ᶜgradᵥ_θ_liq_ice`, `p.precomputed.ᶜgradᵥ_q_tot`: the
    corresponding centered gradients, obtained by applying the Van Leer
    harmonic-mean limiter (`ᶜVanLeer_gradient!`) to the physical face
    gradients and stored as `Covariant3Vector`s.

The centered and face-native (`set_face_diffusivities!`) buoyancy gradients
then reduce to
`blended_N²` FMA broadcasts, which may be evaluated repeatedly (e.g.,
per cloud-fraction Picard iteration, where only `cf` changes) at negligible
cost. The coefficients depend on `(T, ρ, q)` but not on `cf`, so they are
fixed during the Picard iteration.

Mutates `p.precomputed` and uses `p.scratch.ᶜtemp_scalar`;
returns `nothing`.
"""
NVTX.@annotate function set_buoyancy_gradient_inputs!(Y, p, thermo_params)
    (; ᶜbg_coeffs, ᶠ∂θli∂z, ᶠ∂qt∂z, ᶜgradᵥ_θ_liq_ice, ᶜgradᵥ_q_tot) = p.precomputed
    (; ᶜT, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice) = p.precomputed
    ᶠlg = Fields.local_geometry_field(Y.f)
    @. ᶜbg_coeffs = buoyancy_gradient_coefficients(
        thermo_params,
        ᶜT,
        Y.c.ρ,
        ᶜq_tot_nonneg,
        ᶜq_liq,
        ᶜq_ice,
    )
    # θ_li materialized once; the lazy form would re-evaluate the (pow-heavy)
    # Exner function at every gradient stencil point across face and center gradients.
    ᶜθ_li = p.scratch.ᶜtemp_scalar
    @. ᶜθ_li = TD.liquid_ice_pottemp(
        thermo_params,
        ᶜT,
        Y.c.ρ,
        ᶜq_tot_nonneg,
        ᶜq_liq,
        ᶜq_ice,
    )
    # Domain-boundary faces carry zero gradient (ᶠgradᵥ BCs).
    @. ᶠ∂θli∂z = projected_vector_data(C3, ᶠgradᵥ(ᶜθ_li), ᶠlg)
    @. ᶠ∂qt∂z = projected_vector_data(C3, ᶠgradᵥ(ᶜq_tot_nonneg), ᶠlg)

    ᶜVanLeer_gradient!(ᶜgradᵥ_θ_liq_ice, ᶠ∂θli∂z, ᶜθ_li)
    ᶜVanLeer_gradient!(ᶜgradᵥ_q_tot, ᶠ∂qt∂z, ᶜq_tot_nonneg)

    # @show "###########################"
    # @show parent(Y.c.ρq_tot)[16:20]
    # @show parent(ᶜθ_li)[16:20]
    # @show parent(ᶠ∂θli∂z)[16:20]
    # @show parent(ᶜgradᵥ_θ_liq_ice.components.data.:1)[16:20]
    # @show parent(ᶜq_tot_nonneg)[16:20]
    # @show parent(ᶠ∂qt∂z)[16:20]
    # @show parent(ᶜgradᵥ_q_tot.components.data.:1)[16:20]
    return nothing
end

"""
    ᶜVanLeer_gradient!(ᶜout, ᶠ∂ψ∂z, ᶜψ)

Write the Van Leer-limited vertical gradient of a center scalar field,
derived from the already-materialized physical face gradient `ᶠ∂ψ∂z`, into
the `Covariant3Vector` center field `ᶜout`. At each center the two adjacent
*physical* face gradients are combined using the harmonic-mean limiter
`2ab/(a+b)` (zero when they differ in sign), and the result is converted
back into the covariant basis.

For the bottom and top cells, the center gradient is taken from the adjacent
interior face gradient rather than blending the gradients at the two bounding
faces, since the boundary face gradients are imposed by the zero-gradient BC
and do not represent the physical slope.
"""
function ᶜVanLeer_gradient!(ᶜout, ᶠ∂ψ∂z, ᶜψ)
    ᶜlg = Fields.local_geometry_field(axes(ᶜout))
    nc = Spaces.nlevels(axes(ᶜout))
    FT = Spaces.undertype(axes(ᶜout))
    # TODO: pull Δz from the local geometry instead of hardcoding.
    # Δz = FT(50)
    # c_threshold = FT(10)
    ᶜbias_below = Operators.BottomBiasedF2C(
        bottom = Operators.SetValue(
            Fields.level(ᶠ∂ψ∂z, 1 + Fields.half),
        ),
    )
    ᶜbias_above = Operators.TopBiasedF2C(
        top = Operators.SetValue(
            Fields.level(ᶠ∂ψ∂z, nc - Fields.half),
        ),
    )
    @. ᶜout = Geometry.Covariant3Vector(
        harmonic_mean(
            ᶜbias_below(ᶠ∂ψ∂z),
            ᶜbias_above(ᶠ∂ψ∂z),
            # c_threshold * eps(ᶜψ) / Δz,
        ) * unit_basis_vector_data(C3, ᶜlg),
    )
    return nothing
end

"""
    harmonic_mean(a, b, τ = zero(a))

Van Leer's harmonic-mean slope limiter: `2ab/(a+b)` when `a` and `b` share
a sign *and* both have magnitude above `τ`, else zero. Written as
`2*a*(b/denom)` rather than `2*a*b/denom` so that small-but-equal arguments
at Float32 (e.g. humidity slopes on coarse grids) do not underflow through
the intermediate `a*b`.
"""
@inline function harmonic_mean(a, b)
    same_sign = ((a > zero(a)) & (b > zero(b))) |
                ((a < zero(a)) & (b < zero(b)))
    # above_floor = (abs(a) > τ) & (abs(b) > τ)
    use_hm = same_sign# & above_floor
    denom = ifelse(use_hm, a + b, one(a))
    return ifelse(use_hm, 2 * a * (b / denom), zero(a))
end

"""
    set_face_diffusivities!(Y, p)

Fill the face-native turbulence pipeline: at the cell faces where the
diffusive fluxes live, this sets

  - `p.precomputed.ᶠbuoygrad`: the moist buoyancy gradient from the *exact*
    two-point face differences of `(θ_li, q_tot)` with the pointwise
    chain-rule coefficients interpolated to the face (see `blended_N²`);

  - `p.precomputed.ᶠK_h`, `p.precomputed.ᶠK_u`: eddy diffusivity/viscosity
    evaluated natively at the face from `ᶠbuoygrad`, the face turbulent
    Prandtl number, and the face mixing length (the same
    `mixing_length_lopez_gomez_2020` closure evaluated with face inputs).

No-op (fields remain at their previous values) unless
`p.atmos.turbconv_model isa AbstractEDMF`, which is also the condition under
which `Y.c.ρtke` is available. Mutates `p.precomputed.ᶠbuoygrad`, `ᶠK_h`,
`ᶠK_u` and uses three `p.scratch` face scalars; returns `nothing`.

# Extended help

Pointwise face inputs (`κ = ᶠinterp(tke)`, strain, coefficients) use
arithmetic interpolation: it is the second-order-accurate choice in the
resolved limit.
"""
NVTX.@annotate function set_face_diffusivities!(Y, p)
    p.atmos.turbconv_model isa AbstractEDMF || return nothing
    (; ᶠbuoygrad, ᶠK_h, ᶠK_u) = p.precomputed
    (; ᶜbg_coeffs, ᶠ∂θli∂z, ᶠ∂qt∂z) = p.precomputed
    (; ᶜcloud_fraction, ᶜstrain_rate_norm) = p.precomputed
    (; ustar, obukhov_length) = p.precomputed.sfc_conditions
    (; params) = p
    turbconv_params = CAP.turbconv_params(params)
    sf_params = CAP.surface_fluxes_params(params)
    vkc = CAP.von_karman_const(params)

    ᶠΔ_f = resolvability_filter_scale(axes(Y.f))
    ᶠz = Fields.coordinate_field(Y.f).z
    z_sfc = Fields.level(Fields.coordinate_field(Y.f).z, Fields.half)
    ᶜtke = @. lazy(specific(Y.c.ρtke, Y.c.ρ))
    sfc_tke = Fields.level(ᶜtke, 1)

    # Face-native moist buoyancy gradient: the vertical differences of the
    # prognostic state are exactly defined at the face by the two-point
    # gradient stencil; the pointwise chain-rule coefficients vary smoothly
    # and are interpolated.
    @. ᶠbuoygrad = blended_N²(
        ᶠinterp(ᶜbg_coeffs),
        ᶠinterp(ᶜcloud_fraction),
        ᶠ∂θli∂z,
        ᶠ∂qt∂z,
    )
    # All face inputs of the mixing-length closure are materialized: nesting
    # an operator broadcast inside the closure's lazy tree would turn it into
    # a stencil broadcast, whose interior-window logic cannot handle the
    # point-space surface fields (sfc_tke, z_sfc, ustar) the closure needs.
    ᶠκ = p.scratch.ᶠtemp_scalar
    @. ᶠκ = ᶠinterp(max(ᶜtke, 0))
    ᶠstrain = p.scratch.ᶠtemp_scalar_2
    @. ᶠstrain = ᶠinterp(ᶜstrain_rate_norm)
    ᶠPr = p.scratch.ᶠtemp_scalar_3
    @. ᶠPr = turbulent_prandtl_number(params, ᶠbuoygrad, ᶠstrain)

    # Face mixing length: same closure and constants as the center pipeline,
    # evaluated with face inputs.
    ᶠml = @. lazy(
        mixing_length_lopez_gomez_2020(
            turbconv_params,
            sf_params,
            vkc,
            ustar,
            ᶠz,
            z_sfc,
            ᶠΔ_f,
            sfc_tke,
            ᶠbuoygrad,
            ᶠκ,
            obukhov_length,
            ᶠstrain,
            ᶠPr,
            p.atmos.edmfx_model.scale_blending_method,
        ),
    )
    val_master = Val{:master}()
    @. ᶠK_u = eddy_viscosity(
        turbconv_params,
        ᶠκ,
        get_mixing_length_field(ᶠml, val_master),
    )
    @. ᶠK_h = eddy_diffusivity(ᶠK_u, ᶠPr)
    return nothing
end

"""
    get_mixing_length_field(ml::MixingLength, ::Val{P})

Extract one length scale [m] from a `MixingLength`, selected by the `Val`
property tag `P` (GPU-safe field access without runtime symbol lookup).

Tags: `:master` (the blended scale), `:wall`, `:tke`, `:buoy`, `:l_grid`.
"""
@inline get_mixing_length_field(ml::MixingLength, ::Val{:master}) = ml.master
@inline get_mixing_length_field(ml::MixingLength, ::Val{:wall}) = ml.wall
@inline get_mixing_length_field(ml::MixingLength, ::Val{:tke}) = ml.tke
@inline get_mixing_length_field(ml::MixingLength, ::Val{:buoy}) = ml.buoy
@inline get_mixing_length_field(ml::MixingLength, ::Val{:l_grid}) = ml.l_grid

"""
    ᶜmixing_length(Y, p, property::Val{P} = Val{:master}(); grid_scale)

Return a lazy cell-center field of the PROPHET (`EDMFX` in code) mixing length,
selected by `property` (`get_mixing_length_field`).

Evaluates `mixing_length_lopez_gomez_2020` with center inputs.
Only valid for `AbstractEDMF` configurations, which always carry `Y.c.ρtke`.
Writes `p.scratch.ᶜtemp_scalar_5` (the Prandtl number) as a side effect.

# Keyword Arguments

  - `grid_scale`: upper bound on the mixing length. By default, the
    resolvability filter scale `max(Δx_h, Δz)` (see
    `resolvability_filter_scale`).
"""
function ᶜmixing_length(
    Y, p, property::Val{P} = Val{:master}();
    grid_scale = resolvability_filter_scale(axes(Y.c)),
) where {P}
    (; params) = p
    (; ustar, obukhov_length) = p.precomputed.sfc_conditions
    # Centered buoyancy gradient feeds both `l_N` and Pr_t(Ri), and the TKE
    # production-dissipation balance for `l_TKE` (consistent with the actual
    # TKE budget).
    (; ᶜbuoygrad, ᶜstrain_rate_norm) = p.precomputed
    ᶜz = Fields.coordinate_field(Y.c).z
    z_sfc = Fields.level(Fields.coordinate_field(Y.f).z, Fields.half)
    ᶜΔ_f = grid_scale

    # ᶜmixing_length is only evaluated for AbstractEDMF, which always carries
    # Y.c.ρtke.
    ᶜtke = @. lazy(specific(Y.c.ρtke, Y.c.ρ))
    sfc_tke = Fields.level(ᶜtke, 1)

    ᶜprandtl_nvec = p.scratch.ᶜtemp_scalar_5
    @. ᶜprandtl_nvec =
        turbulent_prandtl_number(params, ᶜbuoygrad, ᶜstrain_rate_norm)

    # Extract sub-parameters before the lazy broadcast to avoid capturing
    # the full ClimaAtmosParameters struct (~4 KiB) in GPU kernel parameters.
    turbconv_params = CAP.turbconv_params(params)
    sf_params = CAP.surface_fluxes_params(params)
    vkc = CAP.von_karman_const(params)

    ᶜmixing_length_tuple = @. lazy(
        mixing_length_lopez_gomez_2020(
            turbconv_params,
            sf_params,
            vkc,
            ustar,
            ᶜz,
            z_sfc,
            ᶜΔ_f,
            sfc_tke,
            ᶜbuoygrad,
            ᶜtke,
            obukhov_length,
            ᶜstrain_rate_norm,
            ᶜprandtl_nvec,
            p.atmos.edmfx_model.scale_blending_method,
        ),
    )
    return @. lazy(get_mixing_length_field(ᶜmixing_length_tuple, property))
end

"""
    set_horizontal_diffusivities!(Y, p)

Compute and cache the horizontal eddy viscosity `ᶜK_u_h` and eddy diffusivity
`ᶜK_h_h` of the TKE-based closure, with the mixing length limited by the
horizontal node spacing, `l_h = min(l_phys, Δx_h)`.
"""
function set_horizontal_diffusivities!(Y, p)
    (; params) = p
    (; ᶜK_u_h, ᶜK_h_h, ᶜbuoygrad, ᶜstrain_rate_norm) = p.precomputed
    turbconv_params = CAP.turbconv_params(params)
    Δx_h = horizontal_filter_scale(axes(Y.c))
    ᶜl_h = ᶜmixing_length(Y, p; grid_scale = Δx_h)
    ᶜtke = @. lazy(specific(Y.c.ρtke, Y.c.ρ))
    @. ᶜK_u_h = eddy_viscosity(turbconv_params, ᶜtke, ᶜl_h)
    ᶜprandtl_nvec =
        @. lazy(turbulent_prandtl_number(params, ᶜbuoygrad, ᶜstrain_rate_norm))
    @. ᶜK_h_h = eddy_diffusivity(ᶜK_u_h, ᶜprandtl_nvec)
    return nothing
end

"""
    ᶜdiffusive_flux_divergenceᵥ(ᶠcoef, ᶜχ)

Return the lazy vertical divergence of the diffusive scalar flux
`F = -ᶠcoef ∇χ`, with zero-flux top and bottom boundaries.

`ᶠcoef` must be a face field or a `lazy` broadcast, not a bare `Field * Field`
product. Fold `ρ`, `K`, and any scaling factor into `ᶠcoef` in left-to-right order.
"""
ᶜdiffusive_flux_divergenceᵥ(ᶠcoef, ᶜχ) = @. lazy(ᶜdiffdivᵥ(-(ᶠcoef * ᶠgradᵥ(ᶜχ))))

"""
    ᶜh_eff_plus_Φ!(ᶜout, thermo_params, ᶜT, ᶜΦ, ᶜq_vap, ᶜq_liq, ᶜq_ice)

Write `h_eff + Φ` into the center field `ᶜout` and return it, where

    h_eff = (h_v q_v + h_l q_l + h_i q_i) / max(q_v + q_l + q_i, ε)

is the mass-weighted specific enthalpy of the suspended water. Every specific
humidity is clipped at zero, so a limiter undershoot cannot change the sign of
the weights or of the denominator.

`h_eff + Φ` is the coefficient of the aggregate water gradient in the
single-gradient enthalpy flux `F_h = -K [∇s_d + (h_eff + Φ) ∇q_tot_eff]` shared
by vertical diffusion, horizontal diffusion and hyperdiffusion, and it is the
same coefficient the implicit Jacobian holds frozen.

The result is written into a field rather than returned lazily because every
caller passes it to a divergence operator or a `DiagonalMatrixRow`, where the
nested expression exceeds GPU kernel parameter limits.
"""
function ᶜh_eff_plus_Φ!(ᶜout, thermo_params, ᶜT, ᶜΦ, ᶜq_vap, ᶜq_liq, ᶜq_ice)
    FT = eltype(ᶜout)
    ϵ_FT = eps(FT)
    @. ᶜout =
        (
            TD.enthalpy_vapor(thermo_params, ᶜT) * max(FT(0), ᶜq_vap) +
            TD.enthalpy_liquid(thermo_params, ᶜT) * max(FT(0), ᶜq_liq) +
            TD.enthalpy_ice(thermo_params, ᶜT) * max(FT(0), ᶜq_ice)
        ) / max(
            max(FT(0), ᶜq_vap) + max(FT(0), ᶜq_liq) + max(FT(0), ᶜq_ice),
            ϵ_FT,
        ) + ᶜΦ
    return ᶜout
end

"""
    ᶜh_eff_plus_Φ!(ᶜout, Y, p)

Write `h_eff + Φ` into `ᶜout` and return it, taking the temperature and
geopotential from the cache and the suspended water from `ᶜsuspended_water`.
"""
function ᶜh_eff_plus_Φ!(ᶜout, Y, p)
    thermo_params = CAP.thermodynamics_params(p.params)
    (; ᶜT) = p.precomputed
    (; ᶜΦ) = p.core
    ᶜq_vap, ᶜq_lcl, ᶜq_icl = ᶜsuspended_water(Y, p)
    return ᶜh_eff_plus_Φ!(ᶜout, thermo_params, ᶜT, ᶜΦ, ᶜq_vap, ᶜq_lcl, ᶜq_icl)
end

"""
    ᶜdry_static_energy(p)

Return the lazy dry static energy `s_d = h_d + Φ` from the cached temperature
and geopotential, the scalar whose gradient the diffusive enthalpy flux
`F_h = -K [∇s_d + (h_eff + Φ) ∇q_tot_eff]` acts on besides the diffusing water.
"""
function ᶜdry_static_energy(p)
    thermo_params = CAP.thermodynamics_params(p.params)
    (; ᶜT) = p.precomputed
    (; ᶜΦ) = p.core
    return @. lazy(TD.dry_static_energy(thermo_params, ᶜT, ᶜΦ))
end

"""
    ᶜsuspended_water(Y, p)

Return the lazy specific humidities `(q_vap, q_lcl, q_icl)` of the suspended
water: vapor, cloud liquid and cloud ice. With a non-equilibrium scheme the
cloud species are prognostic and are read from `Y`; otherwise they are the
diagnostic `ᶜq_liq` and `ᶜq_ice` of the equilibrium partition.

These are the weights of `ᶜh_eff_plus_Φ!` and, together with
`ᶜdiffusing_water`, define which water the diffusive and hyperdiffusive
fluxes act on.
"""
function ᶜsuspended_water(Y, p)
    (; ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice) = p.precomputed
    ᶜq_vap = @. lazy(TD.vapor_specific_humidity(ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice))
    return p.atmos.microphysics_model isa
           Union{NonEquilibriumMicrophysics1M, NonEquilibriumMicrophysics2M} ?
           (
        ᶜq_vap,
        (@. lazy(specific(Y.c.ρq_lcl, Y.c.ρ))),
        (@. lazy(specific(Y.c.ρq_icl, Y.c.ρ))),
    ) : (ᶜq_vap, ᶜq_liq, ᶜq_ice)
end

"""
    ᶜdiffusing_water(Y, p)

Return the lazy specific humidity of the water that diffuses,
`q_tot_eff = q_tot - q_rai - q_sno`. Rain and snow are excluded because they
sediment rather than follow the turbulent flow; with an equilibrium scheme
there is no separate precipitation mass and this is `q_tot`.
"""
ᶜdiffusing_water(Y, p) =
    p.atmos.microphysics_model isa
    Union{NonEquilibriumMicrophysics1M, NonEquilibriumMicrophysics2M} ?
    (@. lazy(specific(Y.c.ρq_tot - Y.c.ρq_rai - Y.c.ρq_sno, Y.c.ρ))) :
    (@. lazy(specific(Y.c.ρq_tot, Y.c.ρ)))

"""
    gradient_richardson_number(params, ᶜN², ᶜstrain_rate_norm)

Compute the gradient Richardson number, the ratio of the buoyancy to the shear
term in the TKE budget:

    Ri = ᶜN² / max(2 SᵢⱼSᵢⱼ, eps).

# Arguments

  - `params`: Parameter set; used only for the floating-point type.
  - `ᶜN²`: Effective squared buoyancy frequency [1/s²].
  - `ᶜstrain_rate_norm`: Squared Frobenius norm of the strain-rate tensor,
    `SᵢⱼSᵢⱼ` [1/s²].

# Returns

The gradient Richardson number [-].
"""
function gradient_richardson_number(params, ᶜN², ᶜstrain_rate_norm)
    FT = eltype(params)

    # Calculate the denominator term for Ri, ensuring it's not zero
    # Based on the formulation Ri = N^2 / max(2*|S|, eps)
    ᶜshear_term_safe = max(2 * ᶜstrain_rate_norm, eps(FT))
    ᶜRi_grad = ᶜN² / ᶜshear_term_safe

    return ᶜRi_grad
end


"""
    turbulent_prandtl_number(params, ᶜN², ᶜstrain_rate_norm)

Compute the turbulent Prandtl number as a function of the gradient Richardson
number (`gradient_richardson_number`).

The formula is from Li et al. (JAS 2015, DOI: 10.1175/JAS-D-14-0335.1, their
Eq. 39), reformulated and with an algebraic error in their expression
corrected:

    Pr_t(Ri) = (X + sqrt(max(X² - 4 Pr_n Ri, 0))) / 2,

with `X = Pr_n + ω_pr Ri`, the neutral Prandtl number `Pr_n`
(`Prandtl_number_0`), and the scale coefficient `ω_pr`
(`Prandtl_number_scale`). It applies in both stable (`Ri > 0`) and unstable
(`Ri < 0`) conditions; the result is limited to `[eps(FT), Pr_max]`.

# Arguments

  - `params`: Parameter set.
  - `ᶜN²`: Effective squared buoyancy frequency [1/s²].
  - `ᶜstrain_rate_norm`: Squared Frobenius norm of the strain-rate tensor,
    `SᵢⱼSᵢⱼ` [1/s²].

# Returns

The turbulent Prandtl number [-].

The strong-stability limit of this closure (`Pr(Ri) → ∞`) is what closes the
TKE dissipation coefficient; see `tke_dissipation_coefficient` for that
derivation.
"""
function turbulent_prandtl_number(params, ᶜN², ᶜstrain_rate_norm)
    FT = eltype(params)
    turbconv_params = CAP.turbconv_params(params)
    eps_FT = eps(FT)

    # Parameters from CliMAParams
    Pr_n = CAP.Prandtl_number_0(turbconv_params) # Neutral Prandtl number
    ω_pr = CAP.Prandtl_number_scale(turbconv_params) # Prandtl number scale coefficient
    Pr_max = CAP.Pr_max(turbconv_params) # Maximum Prandtl number limit

    # Calculate the raw gradient Richardson number using the new helper function
    ᶜRi_grad = gradient_richardson_number(params, ᶜN², ᶜstrain_rate_norm)

    # --- Apply the Pr_t(Ri) formula valid for stable and unstable conditions ---

    # Calculate the intermediate term X = Pr_n + ω_pr * Ri
    X = Pr_n + ω_pr * ᶜRi_grad

    # Calculate the discriminant term: (Pr_n + ω_pr*Ri)^2 - 4*Pr_n*Ri = X^2 - 4*Pr_n*Ri
    discriminant = X * X - 4 * Pr_n * ᶜRi_grad
    # Ensure the discriminant is non-negative before taking the square root
    discriminant_safe = max(discriminant, FT(0))

    # Calculate the Prandtl number using the positive root solution of the quadratic eq.
    # Pr_t = ( X + sqrt(discriminant_safe) ) / 2
    prandtl_nvec = (X + sqrt(discriminant_safe)) / 2

    # Optional safety: ensure Pr_t is not excessively small or negative,
    # though the formula should typically yield positive values if Pr_n > 0.
    # Also ensure that it's not larger than the Pr_max parameter.
    return min(max(prandtl_nvec, eps_FT), Pr_max)
end

"""
    tke_dissipation_coefficient(turbconv_params)

Return the TKE dissipation coefficient `c_d = c_m c_b / Ri_c`.

This derived closure coefficient combines the eddy-viscosity coefficient `c_m`
(`tke_ed_coeff`), the static-stability coefficient `c_b`
(`static_stab_coeff`), and the critical gradient Richardson number `Ri_c`
(`Ri_crit`), all dimensionless. Used by `tke_dissipation`.

# Extended help

Derivation of `c_d = c_m c_b / Ri_c`. Consider the local TKE balance
(production = buoyancy destruction + dissipation, no transport) in stably
stratified air, where the mixing length is buoyancy-limited,
`l = l_N = √(c_b e)/N`, with `e` the TKE and `N` the buoyancy frequency:

    2 K_u S² - K_h N² = c_d e^{3/2} / l,
    K_u = c_m l √e,   K_h = K_u / Pr,

with `S² = SᵢⱼSᵢⱼ` the squared strain-rate norm. Substituting `l = l_N` makes
every term linear in `e`,

    c_m √c_b (2S² - N²/Pr) e / N = c_d e N / √c_b,

so the TKE amplitude cancels: there is no local equilibrium level, only a sharp
threshold — TKE grows or decays exponentially according to the sign of the
balance. Dividing by `N²` and using the gradient Richardson number
`Ri = N²/(2S²)` gives the marginal condition

    1/Ri = 1/Pr + c_d/(c_m c_b).

In strong stability `Pr(Ri)` grows without bound (see
`turbulent_prandtl_number`), so `1/Pr → 0` and turbulence is maintained
for `Ri < Ri_c` with

    Ri_c = c_m c_b / c_d.

This combination of the three coefficients controls the stable-regime cutoff, so
`(c_m, c_b, Ri_c)` are calibrated and `c_d` follows. The basis is nearly
orthogonal: `c_m/c_d = Ri_c/c_b`, so the neutral-limit equilibrium TKE
(`e = 2 (c_m/c_d) l² S²`) is independent of `c_m`, which acts as a
flux-magnitude scaling, while `c_b` partitions TKE amplitude against the
buoyancy length and `Ri_c` sets the stability cutoff.
"""
tke_dissipation_coefficient(turbconv_params) =
    CAP.tke_ed_coeff(turbconv_params) * CAP.static_stab_coeff(turbconv_params) /
    CAP.Ri_crit(turbconv_params)

"""
    blend_scales(
        method::AbstractScaleBlending,
        l::SA.SVector,
        turbconv_params,
    )

Combine the physical mixing-length scales in `l` (wall, TKE balance,
stability) into a single non-negative scale [m].

Dispatches on `method`:

  - `SmoothMinimumBlending`: differentiable smooth minimum
    (`lamb_smooth_minimum`) with the parameters `smin_ub` and `smin_rm`.
  - `HardMinimumBlending`: plain `minimum(l)`.

# Arguments

  - `method`: Blending method.
  - `l`: `SVector` of candidate length scales [m].
  - `turbconv_params`: Turbulence-convection parameters.

# Returns

The blended mixing length [m], floored at zero.
"""
function blend_scales(
    method::SmoothMinimumBlending,
    l::SA.SVector,
    turbconv_params,
)
    FT = eltype(l)
    smin_ub = CAP.smin_ub(turbconv_params)
    smin_rm = CAP.smin_rm(turbconv_params)
    l_final = lamb_smooth_minimum(l, smin_ub, smin_rm)
    return max(l_final, FT(0))
end

function blend_scales(
    method::HardMinimumBlending,
    l::SA.SVector,
    turbconv_params,
)
    FT = eltype(l)
    return max(minimum(l), FT(0))
end

"""
    lamb_smooth_minimum(l, smoothness_param, λ_floor)

Compute a differentiable approximation to `minimum(l)` as an exponentially
weighted average,

    smin = Σᵢ lᵢ exp(-(lᵢ - x_min)/λ₀) / Σᵢ exp(-(lᵢ - x_min)/λ₀),

with `x_min = minimum(l)` and the smoothness scale
`λ₀ = max(x_min * smoothness_param / W(2/e), λ_floor)`, where `W(2/e) ≈ 0.463`
is the Lambert W function (hard-coded for type stability). The result is
slightly larger than the true minimum; larger `λ₀` gives a smoother
approximation.

# Arguments

  - `l`: `SVector` of values to minimize over, here length scales [m].
  - `smoothness_param`: Scaling of the smoothness parameter `λ₀`; larger values
    give a smoother minimum [-].
  - `λ_floor`: Lower bound on `λ₀`, in the units of `l`. Must be positive.

# Returns

The smooth minimum, in the units of `l`.
"""
function lamb_smooth_minimum(l, smoothness_param, λ_floor)
    FT = typeof(smoothness_param)

    # Precomputed constant value of LambertW(2/e) for efficiency.
    # LambertW.lambertw(FT(2) / FT(MathConstants.e)) ≈ 0.46305551336554884
    lambert_2_over_e = FT(0.46305551336554884)

    # Ensure the floor for the smoothness parameter is positive
    @assert λ_floor > 0 "λ_floor must be positive"

    # 1. Find the minimum value in the vector
    x_min = minimum(l)

    # 2. Calculate the smoothing parameter λ_0.
    # It scales with the minimum value and smoothness_param, bounded below by λ_floor.
    # Using a precomputed value for lambertw(2/e) for type stability and efficiency.
    lambda_scaling_term = x_min * smoothness_param / lambert_2_over_e
    λ_0 = max(lambda_scaling_term, λ_floor)

    # 3. Ensure λ_0 is numerically positive (should be guaranteed by λ_floor > 0)
    λ_0_safe = max(λ_0, eps(FT))

    # Calculate the numerator and denominator for the weighted average.
    # The exponent is -(l_i - x_min)/λ_0_safe, which is <= 0.
    numerator = sum(l_i -> l_i * exp(-(l_i - x_min) / λ_0_safe), l)
    denominator = sum(l_i -> exp(-(l_i - x_min) / λ_0_safe), l)

    # 4. Calculate the smooth minimum.
    # The denominator is guaranteed to be >= 1 because the term with l_i = x_min
    # contributes exp(0) = 1. Add a safeguard for (unlikely) underflow issues.
    return numerator / max(eps(FT), denominator)
end

"""
    eddy_viscosity(params, tke, mixing_length)

Compute the eddy viscosity for momentum, `K_u = c_m l √max(tke, 0)`.

# Arguments

  - `params`: Turbulence-convection parameters (`c_m` is `tke_ed_coeff`).
  - `tke`: Turbulent kinetic energy [m²/s²].
  - `mixing_length`: Turbulent mixing length [m].

# Returns

The eddy viscosity `K_u` [m²/s].
"""
function eddy_viscosity(params, tke, mixing_length)
    c_m = CAP.tke_ed_coeff(params)
    return c_m * mixing_length * sqrt(max(tke, 0))
end

"""
    eddy_diffusivity(K_u, prandtl_number)

Compute the eddy diffusivity for scalars, `K_h = K_u / Pr_t` [m²/s], from the
eddy viscosity `K_u` [m²/s] and the turbulent Prandtl number `Pr_t` [-].

`Pr_t` from `turbulent_prandtl_number` is already bounded away from
zero, so no guard is needed here.
"""
function eddy_diffusivity(K_u, prandtl_number)
    return K_u / prandtl_number # prandtl_nvec is already bounded by eps_FT and Pr_max
end
