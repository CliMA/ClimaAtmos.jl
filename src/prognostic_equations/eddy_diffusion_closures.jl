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

Pointwise thermodynamic coefficients of the moist buoyancy-gradient chain
rule. The buoyancy gradient is *linear* in the vertical gradients of the
prognostic state,

    ∂b/∂z = C_θ(state, cf) ∂θli/∂z + C_q(state, cf) ∂qt/∂z,

with the cloud-fraction blend also linear:
`C_θ = Cθ_unsat + cf ΔCθ`, `C_q = Cq_unsat + cf ΔCq`. This function returns
the four cf-independent coefficients as a `NamedTuple`
`(; Cθ_unsat, ΔCθ, Cq_unsat, ΔCq)`.

The coefficients contain all of the expensive pointwise thermodynamics
(saturation vapor pressure, latent heat, potential temperatures); evaluating
them once per state update and reusing them for the centered, one-sided, and
face-native gradient stencils — via [`blended_N²`](@ref) — avoids recomputing
that thermodynamics for every stencil.
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

Moist buoyancy gradient from precomputed chain-rule coefficients
(see [`buoyancy_gradient_coefficients`](@ref)), the local cloud fraction, and
projected vertical gradients of `θ_li` and `q_tot` (physical scalars):

    ∂b/∂z = (Cθ_unsat + cf ΔCθ) ∂θli/∂z + (Cq_unsat + cf ΔCq) ∂qt/∂z.
"""
@inline blended_N²(coeffs, cf, ∂θli∂z, ∂qt∂z) =
    (coeffs.Cθ_unsat + cf * coeffs.ΔCθ) * ∂θli∂z +
    (coeffs.Cq_unsat + cf * coeffs.ΔCq) * ∂qt∂z

"""
    buoyancy_gradients(closure, thermo_params, bg_model::EnvBuoyGradVars)

Calculates the mean vertical buoyancy gradient (`∂b/∂z`) in the environment,
from the state and the prognostic vertical gradients (`∂θₗᵢ/∂z`, `∂qₜ/∂z`)
bundled in `bg_model`.

This gradient is determined by considering contributions from both the unsaturated
and saturated portions of the environment, weighted by the environmental cloud
fraction. The calculation involves:

 1. Determining partial derivatives of buoyancy with respect to virtual potential
    temperature (`θᵥ`) for the unsaturated part, and with respect to liquid-ice
    potential temperature (`θₗᵢ`) and total specific humidity (`qₜ`) for the
    saturated part ([`buoyancy_gradient_coefficients`](@ref)).
 2. Applying the chain rule using the provided vertical gradients of these
    thermodynamic variables ([`buoyancy_gradient_chain_rule`](@ref)).
 3. Blending the resulting unsaturated and saturated buoyancy gradients based on
    the environmental cloud fraction.

Arguments:

  - `closure`: The environmental buoyancy gradient closure type (e.g., `BuoyGradMean`).
  - `thermo_params`: Thermodynamic parameters from `CLIMAParameters`.
  - `bg_model`: An `EnvBuoyGradVars` bundling `T`, `ρ`, `q_tot`, `q_liq`,
    `q_ice`, `cf`, `∂qt∂z`, and `∂θli∂z`.

Returns:

  - `∂b∂z`: The mean vertical buoyancy gradient [s⁻²].
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
        ∂b∂θli_sat::FT,
        ∂b∂qt_sat::FT,
    ) where {FT}

Calculates the mean vertical buoyancy gradient (`∂b∂z`) by applying the chain rule
to the partial derivatives of buoyancy and then blending based on cloud fraction.

This function takes the partial derivatives of buoyancy with respect to:

  - virtual potential temperature (`∂b/∂θᵥ`) for the unsaturated part,
  - liquid-ice potential temperature (`∂b/∂θₗᵢ,sat`) for the saturated part,
  - total specific humidity (`∂b/∂qₜ,sat`) for the saturated part.

It then multiplies these by the respective vertical gradients of `θᵥ`, `θₗᵢ`, and `qₜ`
(obtained from `bg_model`)
to get the buoyancy gradients for the unsaturated (`∂b∂z_unsat`) and saturated
(`∂b∂z_sat`) parts of the environment.
Finally, it returns a single mean buoyancy gradient by linearly combining
`∂b∂z_unsat` and `∂b∂z_sat` weighted by the environmental cloud fraction
(also obtained from `bg_model`).

Arguments:

  - `closure`: The environmental buoyancy gradient closure type.
  - `bg_model`: Precomputed environmental buoyancy gradient variables (`EnvBuoyGradVars`).
  - `thermo_params`: Thermodynamic parameters from `CLIMAParameters`.
  - `∂b∂θli_sat`: Partial derivative of buoyancy w.r.t. liquid-ice potential temperature (saturated part).
  - `∂b∂qt_sat`: Partial derivative of buoyancy w.r.t. total specific humidity (saturated part).

Returns:

  - `∂b∂z`: The mean vertical buoyancy gradient [s⁻²].
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
    surface_flux_tke(
        turbconv_params,
        ρa_sfc,
        ustar,
        surface_local_geometry,
    )

Calculates the surface flux of TKE, a C3 vector used by
ClimaAtmos operator boundary conditions.

The flux magnitude is modeled as
c_k * ρa_sfc * ustar^3`,
directed along the surface upward normal.

Details:

  - `c_k`: A dimensionless coefficient (`tke_surface_flux_coeff`) scaling the surface flux of TKE.
  - The formulation `ustar^3` implies that the TKE flux is primarily driven by
    shear production at the surface.

This flux represents the net input of TKE into the atmosphere from the surface,
arising from turbulent generation processes by unresolved roughness elements.

Arguments:

  - `turbconv_params`: Set of turbulence and convection model parameters.
  - `ρa_sfc`: Area-fraction weighted air density at the surface [kg/m^3].
  - `ustar`: Friction velocity [m/s].
  - `surface_local_geometry`: The `LocalGeometry` object at the surface.

Returns:

  - A `ClimaCore.Geometry.C3` vector representing the TKE flux normal to the surface.
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
    turbconv_params,
    sf_params,
    vkc,
    ustar,
    ᶜz,
    z_sfc,
    ᶜΔ_f,
    sfc_tke,
    ᶜN²_eff,
    ᶜN²_prod,
    ᶜtke,
    obukhov_length,
    ᶜstrain_rate_norm,
    ᶜPr,
    scale_blending_method,
)

where:
- `turbconv_params`: Turbulence-convection parameter set.
- `sf_params`: Surface flux parameter set (Businger parameters).
- `vkc`: Von Kármán constant.
- `ustar`: Friction velocity [m/s].
- `ᶜz`: Cell center height [m].
- `z_sfc`: Surface elevation [m].
- `ᶜΔ_f`: Resolvability filter scale [m] that caps the mixing length (see
  [`resolvability_filter_scale`](@ref); `Inf` where the grid imposes no
  scale, as in single columns).
- `sfc_tke`: TKE near the surface (e.g., first cell center) [m^2/s^2].
- `ᶜN²_eff`: Effective squared Brunt-Väisälä frequency [1/s^2], used for the
  buoyancy-limited scale `l_N` (may include the unresolved-jump augmentation
  of `interface_effective_N²`).
- `ᶜN²_prod`: Squared Brunt-Väisälä frequency entering the TKE
  production-dissipation balance for `l_TKE` [1/s^2]. Passed separately so
  the balance uses the same (centered) stability as the actual TKE buoyancy
  production, keeping `l_TKE` stencil-consistent with the budget it
  parameterizes even when `ᶜN²_eff` carries the interface augmentation.
- `ᶜtke`: Turbulent kinetic energy at cell center [m^2/s^2].
- `obukhov_length`: Surface Monin-Obukhov length [m].
- `ᶜstrain_rate_norm`: Frobenius norm of strain rate tensor [1/s].
- `ᶜPr`: Turbulent Prandtl number [-].
- `scale_blending_method`: The method to use for blending physical scales.

Point-wise calculation of the turbulent mixing length, limited by physical constraints (wall distance,
TKE balance, stability) and by the resolvability filter scale
(see [`resolvability_filter_scale`](@ref)). Based on
Lopez‐Gomez, I., Cohen, Y., He, J., Jaruga, A., & Schneider, T. (2020).
A generalized mixing length closure for eddy‐diffusivity mass‐flux schemes of turbulence and convection.
Journal of Advances in Modeling Earth Systems, 12, e2020MS002161. https://doi.org/ 10.1029/2020MS002161

Returns a `MixingLength{FT}` struct containing the final blended mixing length (`master`)
and its constituent physical scales.
"""

function mixing_length_lopez_gomez_2020(
    turbconv_params,
    sf_params,
    vkc,
    ustar,
    ᶜz,
    z_sfc,
    ᶜΔ_f,
    sfc_tke,
    ᶜN²_eff,
    ᶜN²_prod,
    ᶜtke,
    obukhov_length,
    ᶜstrain_rate_norm,
    ᶜPr,
    scale_blending_method,
)

    FT = eltype(ᶜz)
    eps_FT = eps(FT)

    c_m = CAP.tke_ed_coeff(turbconv_params)
    c_d = tke_dissipation_coefficient(turbconv_params)
    c_b = CAP.static_stab_coeff(turbconv_params)

    # l_z: Geometric distance from the surface
    l_z = ᶜz - z_sfc
    # Ensure l_z is non-negative when ᶜz is numerically smaller than z_sfc.
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

    # --- l_TKE: TKE production-dissipation balance scale ---
    tke_pos = max(ᶜtke, FT(0)) # Ensure TKE is not negative
    sqrt_tke_pos = sqrt(tke_pos)

    # Net production of TKE from shear and buoyancy is approximated by
    #     (S² − N²/Pr_t) · √TKE · l,
    # where S² denotes the gradient involved in shear production and
    # N²/Pr_t denotes the gradient involved in buoyancy production.
    # The factor below corresponds to that production term normalised by l.
    a_pd = c_m * (2 * ᶜstrain_rate_norm - ᶜN²_prod / ᶜPr) * sqrt_tke_pos

    # Dissipation is modelled as c_d · k^{3/2} / l.
    # For the quadratic expression below, c_neg ≡ c_d · k^{3/2}.
    c_neg = c_d * tke_pos * sqrt_tke_pos

    # Solve for l_TKE in
    #     a_pd · l_TKE − c_neg / l_TKE = 0
    #  ⇒  a_pd · l_TKE² − c_neg = 0
    # yielding
    #     l_TKE = √c_neg / a_pd.
    l_TKE = ifelse(tke_pos > eps_FT, sqrt(c_neg / max(a_pd, eps_FT)), FT(0))

    # --- l_N: Static-stability length scale (buoyancy limit), constrained by l_z ---
    N_eff_sq = max(ᶜN²_eff, FT(0)) # Use N^2 only if stable (N^2 > 0)
    l_N = l_z # Default to wall distance if not stably stratified or TKE is zero
    if N_eff_sq > eps_FT && tke_pos > eps_FT
        N_eff = sqrt(N_eff_sq)
        # l_N ~ sqrt(c_b * TKE) / N_eff
        l_N_physical = sqrt(c_b * tke_pos) / N_eff
        # Limit by distance from wall
        l_N = min(l_N_physical, l_z)
    end
    l_N = max(l_N, FT(0)) # Ensure non-negative


    # --- Combine Scales ---

    # Vector of *physical* scales (wall, TKE, stability)
    # These scales (l_W, l_TKE, l_N) are already ensured to be non-negative.
    # l_N is already limited by l_z. l_W and l_TKE are not necessarily.
    l_physical_scales =
        (tke_pos > eps_FT && a_pd <= 0) ? SA.SVector(l_W, l_N) : SA.SVector(l_W, l_TKE, l_N)

    l_smin =
        blend_scales(scale_blending_method, l_physical_scales, turbconv_params)

    # 1. Limit the combined physical scale by the distance from the wall.
    #    This step mitigates excessive values of l_W or l_TKE.
    l_limited_phys_wall = min(l_smin, l_z)

    # 2. Impose the resolvability filter scale (see
    #    resolvability_filter_scale for the rationale and regimes).
    l_grid = ᶜΔ_f
    l_final = min(l_limited_phys_wall, l_grid)

    # Final check: guarantee that the mixing length is at least a small positive
    # value.  This prevents division-by-zero in
    #     ε_d = C_d · TKE^{3/2} / l_mix
    # when TKE > 0.  When TKE = 0, l_mix is inconsequential, but eps_FT
    # provides a conservative lower bound.
    # minimum mixing length
    l_final = max(l_final, FT(1)) # TODO: make a climaparam

    return MixingLength(l_final, l_W, l_TKE, l_N, l_grid)
end

"""
    set_buoyancy_gradient_inputs!(Y, p, thermo_params)

Materializes, once per state update, everything the buoyancy-gradient
stencils share:

  - `p.precomputed.ᶜbg_coeffs`: the pointwise chain-rule coefficients of
    [`buoyancy_gradient_coefficients`](@ref) (all of the expensive
    saturation thermodynamics lives here);
  - `p.precomputed.ᶠ∂θli∂z`, `p.precomputed.ᶠ∂qt∂z`: exact two-point face
    gradients of `θ_li` and `q_tot`, projected to physical scalars.

The centered, one-sided (`set_stability_buoyancy_gradient!`), and face-native
(`set_face_diffusivities!`) buoyancy gradients then reduce to
[`blended_N²`](@ref) FMA broadcasts, which may be evaluated repeatedly (e.g.,
per cloud-fraction Picard iteration, where only `cf` changes) at negligible
cost. The coefficients depend on `(T, ρ, q)` but not on `cf`, so they are
fixed during the Picard iteration.
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

    @. ᶜgradᵥ_θ_liq_ice = ᶜgradᵥ(ᶠinterp(ᶜθ_li))
    @. ᶜgradᵥ_q_tot = ᶜgradᵥ(ᶠinterp(ᶜq_tot_nonneg))
    return nothing
end

"""
    set_stability_buoyancy_gradient!(Y, p, thermo_params)

Fills `p.precomputed.ᶜN²_eff` with an interface-aware effective
stability: at each cell center, the buoyancy gradient is evaluated twice, with
upward- and downward-biased one-sided vertical gradients of `θ_li` and `q_tot`
(i.e., the exact two-point gradients of the two adjacent faces), each is
augmented by the unresolved-jump term of [`interface_effective_N²`](@ref)
(when prognostic TKE is available), and the more stable (larger) of the two
face values is kept.

Rationale: centered two-cell gradients average across unresolved, strongly
stable interfaces such as boundary-layer capping inversions, biasing N²_eff
low — and hence the stability mixing length `l_N` and turbulent Prandtl
number toward too much mixing — exactly in the entrainment zone. The one-sided
evaluation lets a single-cell jump register at both adjacent cell centers, and
the jump term of `interface_effective_N²` additionally accounts for the limit
in which the jump is a sheet interface thinner than the grid: eddy excursions
are then capped by the work against the full jump `Δb`, independent of `Δz`.
Away from sharp interfaces both one-sided gradients agree with the centered
one and the jump term is `O((Δz/l_N)²)`, so the correction is inactive.

This field feeds the mixing-length and `Pr_t(Ri)` closures only; the TKE
buoyancy production keeps the centered `ᶜbuoygrad`, so convective
production in unstable layers is unaffected. Without prognostic TKE the jump
term is unavailable and the pure one-sided max is used.
"""
NVTX.@annotate function set_stability_buoyancy_gradient!(Y, p, thermo_params)
    (; ᶜN²_eff, ᶜcloud_fraction) = p.precomputed
    (; ᶜbg_coeffs, ᶠ∂θli∂z, ᶠ∂qt∂z) = p.precomputed
    # One-sided center gradients: the exact face gradients (see
    # `set_buoyancy_gradient_inputs!`) brought to centers from the upper
    # (ᶜright_bias) and lower (ᶜleft_bias) adjacent faces. Domain-boundary
    # faces carry zero gradient, so the biased estimates fall back to neutral
    # there and the max picks the interior side.
    ᶜN²_up = @. lazy(
        blended_N²(
            ᶜbg_coeffs,
            ᶜcloud_fraction,
            ᶜright_bias(ᶠ∂θli∂z),
            ᶜright_bias(ᶠ∂qt∂z),
        ),
    )
    ᶜN²_dn = @. lazy(
        blended_N²(
            ᶜbg_coeffs,
            ᶜcloud_fraction,
            ᶜleft_bias(ᶠ∂θli∂z),
            ᶜleft_bias(ᶠ∂qt∂z),
        ),
    )
    if MatrixFields.has_field(Y, @name(c.ρtke))
        # Interface-aware effective stability: each one-sided face gradient is
        # augmented by the unresolved-jump term of `interface_effective_N²`
        # before taking the max, so an inversion concentrated at a face limits
        # eddy excursions through the work against the full jump Δb = N² Δz
        # rather than the Δz-diluted gradient.
        turbconv_params = CAP.turbconv_params(p.params)
        c_b = CAP.static_stab_coeff(turbconv_params)
        ᶜtke_pos = @. lazy(max(specific(Y.c.ρtke, Y.c.ρ), 0))
        ᶠΔz = Fields.Δz_field(axes(Y.f))
        @. ᶜN²_eff = max(
            interface_effective_N²(ᶜN²_up, ᶜright_bias(ᶠΔz), ᶜtke_pos, c_b),
            interface_effective_N²(ᶜN²_dn, ᶜleft_bias(ᶠΔz), ᶜtke_pos, c_b),
        )
    else
        @. ᶜN²_eff = max(ᶜN²_up, ᶜN²_dn)
    end
    return nothing
end

"""
    interface_effective_N²(N², Δz, κ_iso, c_b)

Interface-aware effective squared buoyancy frequency at a face,

    N²_eff = N² + [(Δb)₊]² / (c_b κ_iso),    Δb = N² Δz,

where `N²` is the two-point face buoyancy gradient, `Δz` the face-adjacent
grid spacing, `κ_iso` the isotropic TKE, and `c_b` the static-stability
mixing-length coefficient (`mixing_length_static_stab_coeff`).

The face jump `Δb` is compatible with any subgrid profile between a uniform
gradient over `Δz` (which centered differencing assumes) and a sheet interface
at the face. An eddy of energy `κ_iso` crossing a sheet performs work `Δb ℓ`
over a penetration distance `ℓ`, capping excursions at `ℓ_p = c_b κ_iso / Δb`;
the jump term makes the buoyancy-limited length `l_N = √(c_b κ_iso)/N_eff`
interpolate between the standard resolved limit and `ℓ_p`. In smooth regions
the correction is relatively `O((Δz/l_N)²)` — quadratically small wherever the
stratification is resolved — so `N²_eff` is a consistent, second-order-accurate
discretization of the same continuum stability and acts as a smooth interface
indicator without a mode switch. The positive-part clamp restricts the
correction to stable jumps, leaving convectively unstable layers untouched.
"""
@inline function interface_effective_N²(N², Δz, κ_iso, c_b)
    FT = typeof(N²)
    Δb_pos = max(N² * Δz, FT(0))
    return N² + Δb_pos^2 / (c_b * max(κ_iso, eps(FT)))
end

"""
    set_face_diffusivities!(Y, p)

Face-native turbulence pipeline: fills, at cell faces where the diffusive
fluxes live,

  - `p.precomputed.ᶠbuoygrad`: the moist buoyancy gradient from the *exact*
    two-point face differences of `(θ_li, q_tot)` with the pointwise
    chain-rule coefficients interpolated to the face (see `blended_N²`);

  - `p.precomputed.ᶠK_h`, `p.precomputed.ᶠK_u`: eddy diffusivity/viscosity
    evaluated natively at the face from the face effective stability
    `N²_eff = ᶠbuoygrad + [(Δb)₊]²/(c_b κ)` ([`interface_effective_N²`](@ref)),
    the face turbulent Prandtl number, and the face mixing length (the same
    `mixing_length_lopez_gomez_2020` closure evaluated with face inputs);

  - `p.precomputed.ᶠK_entr`: the interfacial entrainment diffusivity

        K_e = γ w_e Δz,   w_e = A √κ / max(Ri_b, 1),   Ri_b = ℓ_e Δb / κ,

    with `Δb = (ᶠbuoygrad Δz)₊` the stable face buoyancy jump, `ℓ_e` the
    face-native energy-containing eddy scale (minimum of the wall and
    TKE-balance components, which — unlike `l_N` — are not suppressed by the
    interface), `A` the entrainment efficiency
    (`EDMF_interface_entr_efficiency`), and the gate
    `γ = jt/(ᶠbuoygrad + jt)` the fraction of the effective stability carried
    by the unresolved-jump term.

Evaluating the stability closure *at the face* keeps the collapse of `K` at
an unresolved inversion confined to the jump face: a center-based evaluation
(where the max over adjacent faces registers the jump at the whole cell)
necessarily leaks the collapse to the cell's opposite face through
interpolation, under-mixing the interior of the entrainment-zone cell.

The discrete face flux `K_e Δψ/Δz = γ w_e Δψ` represents interfacial
entrainment at velocity `w_e` in down-gradient form: collapsing the
down-gradient diffusivity at a sheet interface (via `N²_eff`) is correct for
turbulent mixing but leaves finite-velocity entrainment unrepresented; `K_e`
restores it. `γ → 1` at sheet interfaces and vanishes as `(Δz/l_N)²` where
the stratification is resolved, so `K_e → 0` doubly — through `γ` and through
`Δz` — as `Δz → 0`, recovering the standard local closure. At coarse `Δz`
over a sharp inversion the entrainment flux `w_e Δψ` is
resolution-independent by construction. (When an inversion is smeared over
two faces, each face carries its partial jump and the summed entrainment flux
under-recovers; the full sub-cell reconstruction that would remove this
residual `Δz`-sensitivity is left to future work.)

Pointwise face inputs (`κ = ᶠinterp(tke)`, strain, coefficients) use
arithmetic interpolation: it is the second-order-accurate choice in the
resolved limit, and the O(1) factor it introduces at sheet interfaces (the
face `κ` mixes the turbulent and quiescent sides) is absorbed by the
calibration of `c_b` and `A`.

`K_e` is added to the face diffusivities for all scalars and momentum in
`edmfx_sgs_diffusive_flux_tendency!` (and its Jacobian), keeping energy,
water, and momentum transport conservative and mutually consistent. The TKE
buoyancy production/destruction is evaluated from the same face diffusivities
and the same `ᶠbuoygrad` (see `edmfx_tke_tendency!`), so the interfacial
sink `−γ w_e Δb` per face — bounded by `A κ^{3/2}/ℓ_e`, a fixed multiple of
the dissipation — is carried automatically and the discrete energy
conversions mirror the fluxes term by term.

Validity domain: the closure targets strong, mixed-layer-capping inversions
(large stable buoyancy jump `Δb`), where the restored diffusive exchange
represents mixed-layer entrainment — DYCOMS and BOMEX cloud cover and
inversion height converge under vertical refinement with it active. At
weak-`Δb`, moisture-dominated (trade-cumulus) inversions on coarse grids,
the down-gradient form transports moisture up the jump faster than it warms
and dries, and the thick inversion-base cell can saturate: 24 h RICO at
`Δz = 160 m` relapses to overcast for *any* `A` (at `A = 0` by moisture
trapping under the unresolved inversion; at larger `A` faster, by diffusive
moistening of the inversion layer), while the fine grid holds trade-cumulus
cover. Cumulus-top entrainment is localized to penetrating plumes and
belongs to the entrainment/detrainment closures, not to `K_e`. Consequently,
calibrate `A` against equilibrium (≳ 24 h) targets — spin-up snapshots
reward values that fail at equilibrium — and treat coarse-grid
weak-inversion cloud cover as outside this closure's convergence guarantee.

No-op (fields remain zero) for non-EDMF configurations or without prognostic
TKE.
"""
NVTX.@annotate function set_face_diffusivities!(Y, p)
    p.atmos.turbconv_model isa AbstractEDMF || return nothing
    (; ᶠbuoygrad, ᶠK_h, ᶠK_u, ᶠK_entr) = p.precomputed
    (; ᶜbg_coeffs, ᶠ∂θli∂z, ᶠ∂qt∂z) = p.precomputed
    (; ᶜcloud_fraction, ᶜstrain_rate_norm) = p.precomputed
    (; ustar, obukhov_length) = p.precomputed.sfc_conditions
    (; params) = p
    turbconv_params = CAP.turbconv_params(params)
    c_b = CAP.static_stab_coeff(turbconv_params)
    A_entr = CAP.interface_entr_efficiency(turbconv_params)
    sf_params = CAP.surface_fluxes_params(params)
    vkc = CAP.von_karman_const(params)

    ᶠΔz = Fields.Δz_field(axes(Y.f))
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
    ᶠN²_eff = p.scratch.ᶠtemp_scalar_2
    @. ᶠN²_eff = interface_effective_N²(ᶠbuoygrad, ᶠΔz, ᶠκ, c_b)
    ᶠstrain = p.scratch.ᶠtemp_scalar_3
    @. ᶠstrain = ᶠinterp(ᶜstrain_rate_norm)
    ᶠPr = p.scratch.ᶠtemp_scalar_4
    @. ᶠPr = turbulent_prandtl_number(params, ᶠN²_eff, ᶠstrain)

    # Face mixing length: same closure and constants as the center pipeline,
    # evaluated with face inputs. The augmented N²_eff limits l_N (and the
    # eddy excursions it represents); the un-augmented face gradient enters
    # the production-dissipation balance for l_TKE, consistent with the TKE
    # budget stencils.
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
            ᶠN²_eff,
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

    # Interfacial entrainment diffusivity; `A` is constant over a run, so the
    # (comparatively expensive) closure is skipped when interfacial entrainment
    # is disabled. ᶠK_entr is still zeroed on every update in that case, so
    # downstream reads (SGS fluxes, TKE budget, diffusion Jacobian) always see
    # a defined value regardless of how the field was allocated.
    if !iszero(A_entr)
        val_energy_containing = Val{:energy_containing}()
        @. ᶠK_entr = interface_entrainment_diffusivity(
            ᶠbuoygrad,
            ᶠΔz,
            ᶠκ,
            get_mixing_length_field(ᶠml, val_energy_containing),
            c_b,
            A_entr,
        )
    else
        @. ᶠK_entr = 0
    end
    return nothing
end
"""
    interface_entrainment_diffusivity(N²_face, Δz, κ_iso, ℓ_e, c_b, A)

Pointwise interfacial entrainment diffusivity `K_e = γ w_e Δz`; see
[`set_face_diffusivities!`](@ref) for the closure. Returns zero
where the face jump is not stable (`Δb ≤ 0`) or turbulence is absent.
"""
@inline function interface_entrainment_diffusivity(
    N²_face,
    Δz,
    κ_iso,
    ℓ_e,
    c_b,
    A,
)
    FT = typeof(Δz)
    κ_safe = max(κ_iso, eps(FT))
    Δb_pos = max(N²_face * Δz, FT(0))
    jt = Δb_pos^2 / (c_b * κ_safe)
    # Gate: fraction of the effective stability carried by the jump term.
    # Branchless (avoids warp divergence). The division is guarded by the
    # `jt > 0` branch: `jt > 0` implies `Δb_pos > 0`, hence `N²_face > 0`, so
    # the denominator `N²_face + jt > 0` and no zero-guard is needed.
    γ = ifelse(jt > zero(jt), jt / (N²_face + jt), zero(jt))
    Ri_b = ℓ_e * Δb_pos / κ_safe
    w_e = A * sqrt(κ_safe) / max(Ri_b, FT(1))
    return γ * w_e * Δz
end

# GPU-safe field access using Val dispatch
@inline get_mixing_length_field(ml::MixingLength, ::Val{:master}) = ml.master
@inline get_mixing_length_field(ml::MixingLength, ::Val{:wall}) = ml.wall
@inline get_mixing_length_field(ml::MixingLength, ::Val{:tke}) = ml.tke
@inline get_mixing_length_field(ml::MixingLength, ::Val{:buoy}) = ml.buoy
@inline get_mixing_length_field(ml::MixingLength, ::Val{:l_grid}) = ml.l_grid
# Energy-containing eddy scale ℓ_e for the interfacial entrainment closure:
# the scales of the eddies that scour an interface (wall and TKE-balance),
# which — unlike l_N — are not suppressed by the interface itself. Also
# bounded by the resolvability filter scale l_grid where the grid imposes a
# finite one (l_grid = max(Δx_h, Δz); Inf for single columns, so the bound
# is inert there and binds only in the gray zone / LES). Branchless ifelse
# avoids GPU warp divergence.
@inline function get_mixing_length_field(
    ml::MixingLength,
    ::Val{:energy_containing},
)
    ℓ_phys = ifelse(ml.tke > zero(ml.tke), min(ml.wall, ml.tke), ml.wall)
    return min(ℓ_phys, ml.l_grid)
end

"""
    ᶜmixing_length(Y, p, property = Val(:master); grid_scale)

Compute the center-space mixing length of the TKE-based closure, with
`property` selecting a `MixingLength` component (see `get_mixing_length_field`).

# Keyword Arguments

  - `grid_scale`: upper bound on the mixing length. By default, the
    resolvability filter scale `max(Δx_h, Δz)` (see
    [`resolvability_filter_scale`](@ref)).
"""
function ᶜmixing_length(
    Y, p, property::Val{P} = Val{:master}();
    grid_scale = resolvability_filter_scale(axes(Y.c)),
) where {P}
    (; params) = p
    (; ustar, obukhov_length) = p.precomputed.sfc_conditions
    # Stability-biased buoyancy gradient: registers unresolved inversions
    # (see set_stability_buoyancy_gradient!); feeds l_N and Pr_t(Ri). The
    # centered gradient feeds the TKE production-dissipation balance for
    # l_TKE, consistent with the actual TKE budget.
    (; ᶜN²_eff, ᶜbuoygrad, ᶜstrain_rate_norm) = p.precomputed
    ᶜz = Fields.coordinate_field(Y.c).z
    z_sfc = Fields.level(Fields.coordinate_field(Y.f).z, Fields.half)
    ᶜΔ_f = grid_scale

    # ᶜmixing_length is only evaluated for AbstractEDMF, which always carries
    # Y.c.ρtke.
    ᶜtke = @. lazy(specific(Y.c.ρtke, Y.c.ρ))
    sfc_tke = Fields.level(ᶜtke, 1)

    ᶜprandtl_nvec = p.scratch.ᶜtemp_scalar_5
    @. ᶜprandtl_nvec =
        turbulent_prandtl_number(params, ᶜN²_eff, ᶜstrain_rate_norm)

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
            ᶜN²_eff,
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
    (; ᶜK_u_h, ᶜK_h_h, ᶜN²_eff, ᶜstrain_rate_norm) = p.precomputed
    turbconv_params = CAP.turbconv_params(params)
    Δx_h = horizontal_filter_scale(axes(Y.c))
    ᶜl_h = ᶜmixing_length(Y, p; grid_scale = Δx_h)
    ᶜtke = @. lazy(specific(Y.c.ρtke, Y.c.ρ))
    @. ᶜK_u_h = eddy_viscosity(turbconv_params, ᶜtke, ᶜl_h)
    ᶜprandtl_nvec =
        @. lazy(turbulent_prandtl_number(params, ᶜN²_eff, ᶜstrain_rate_norm))
    @. ᶜK_h_h = eddy_diffusivity(ᶜK_u_h, ᶜprandtl_nvec)
    return nothing
end

"""
    ᶜdiffusive_flux_divergenceᵥ(ᶠcoef, ᶜχ)

Lazy vertical divergence of the diffusive scalar flux `F = -ᶠcoef ∇χ`, with
zero-flux top and bottom boundaries.

`ᶠcoef` must be a face field or a `lazy` broadcast, not a bare `Field * Field`
product. Fold `ρ`, `K`, and any scaling factor into `ᶠcoef` in left-to-right order.
"""
ᶜdiffusive_flux_divergenceᵥ(ᶠcoef, ᶜχ) = @. lazy(ᶜdiffdivᵥ(-(ᶠcoef * ᶠgradᵥ(ᶜχ))))

"""
    ᶠtotal_enthalpy_gradientᵥ(thermo_params, ᶜT, ᶜΦ, ᶜq_vap, ᶜq_liq, ᶜq_ice)

Lazy face gradient of total enthalpy in dry-static-energy + water-enthalpy form,
`∇s_d + Σ_μ (h_μ + Φ) ∇q_μ` for `μ ∈ {vap, liq, ice}`.

Summands are combined in a fixed order: dry static energy, then vapor, liquid, ice.
"""
ᶠtotal_enthalpy_gradientᵥ(thermo_params, ᶜT, ᶜΦ, ᶜq_vap, ᶜq_liq, ᶜq_ice) = @. lazy(
    ᶠgradᵥ(TD.dry_static_energy(thermo_params, ᶜT, ᶜΦ)) +
    ᶠinterp(TD.enthalpy_vapor(thermo_params, ᶜT) + ᶜΦ) * ᶠgradᵥ(ᶜq_vap) +
    ᶠinterp(TD.enthalpy_liquid(thermo_params, ᶜT) + ᶜΦ) * ᶠgradᵥ(ᶜq_liq) +
    ᶠinterp(TD.enthalpy_ice(thermo_params, ᶜT) + ᶜΦ) * ᶠgradᵥ(ᶜq_ice),
)

"""
    ᶜtotal_enthalpy_gradientₕ!(ᶜ∇h, thermo_params, ᶜT, ᶜΦ, ᶜq_vap, ᶜq_liq, ᶜq_ice)

Write the horizontal gradient of total enthalpy in dry-static-energy +
water-enthalpy form, `∇ₕs_d + Σ_μ (h_μ + Φ) ∇ₕq_μ` for `μ ∈ {vap, liq, ice}`,
into the center field `ᶜ∇h` and return it.

Summands are combined in a fixed order: dry static energy, then vapor, liquid,
ice. The gradient is accumulated in two broadcasts; see the horizontal EDMF
diffusion documentation page.
"""
function ᶜtotal_enthalpy_gradientₕ!(
    ᶜ∇h, thermo_params, ᶜT, ᶜΦ, ᶜq_vap, ᶜq_liq, ᶜq_ice,
)
    @. ᶜ∇h =
        gradₕ(TD.dry_static_energy(thermo_params, ᶜT, ᶜΦ)) +
        (TD.enthalpy_vapor(thermo_params, ᶜT) + ᶜΦ) * gradₕ(ᶜq_vap)
    @. ᶜ∇h +=
        (TD.enthalpy_liquid(thermo_params, ᶜT) + ᶜΦ) * gradₕ(ᶜq_liq) +
        (TD.enthalpy_ice(thermo_params, ᶜT) + ᶜΦ) * gradₕ(ᶜq_ice)
    return ᶜ∇h
end

"""
    gradient_richardson_number(params, ᶜN²_eff, ᶜstrain_rate_norm)

Calculates the gradient Richardson number (Ri).

The gradient Richardson number is a dimensionless parameter that represents the ratio
of buoyancy term to the shear term in the TKE equation. It is calculated as:

    Ri = ᶜN²_eff / max(2 * |S|, ε)

where:

  - `params`: Parameter set (e.g., CLIMAParameters.AbstractParameterSet), used to determine floating point type.
  - `ᶜN²_eff`: Effective squared Brunt-Väisälä frequency [1/s²].
  - `ᶜstrain_rate_norm`: Frobenius norm of the strain rate tensor, |S| [1/s].
  - `ε` is a small machine epsilon value to prevent division by zero.

Returns:

  - The gradient Richardson number (dimensionless scalar).
"""
function gradient_richardson_number(params, ᶜN²_eff, ᶜstrain_rate_norm)
    FT = eltype(params)

    # Calculate the denominator term for Ri, ensuring it's not zero
    # Based on the formulation Ri = N^2 / max(2*|S|, eps)
    ᶜshear_term_safe = max(2 * ᶜstrain_rate_norm, eps(FT))
    ᶜRi_grad = ᶜN²_eff / ᶜshear_term_safe

    return ᶜRi_grad
end


"""
    turbulent_prandtl_number(params, ᶜN²_eff, ᶜstrain_rate_norm)

where:

  - `params`: Parameters set
  - `ᶜN²_eff`: Effective squared Brunt-Väisälä frequency [1/s^2].
  - `ᶜstrain_rate_norm`: Frobenius norm of strain rate tensor, |S| [1/s].

Returns the turbulent Prandtl number based on the gradient Richardson number.

The formula implemented is from Li et al. (JAS 2015, DOI: 10.1175/JAS-D-14-0335.1, their Eq. 39),
with a reformulation and correction of an algebraic error in their expression:

    Pr_t(Ri) = (X + sqrt(max(X^2 - 4*Pr_n*Ri, 0))) / 2

where X = Pr_n + ω_pr * Ri and Ri = N^2 / max(2*|S|, eps).
Parameters used are Pr_n = Prandtl_number_0 (neutral Prandtl number) and
ω_pr = Prandtl_number_scale (Prandtl number scale coefficient).
This formula applies in both stable (Ri > 0) and unstable (Ri < 0) conditions.
The returned turbulent Prandtl number is limited to be between eps(FT) and Pr_max.

The strong-stability limit of this closure (`Pr(Ri) → ∞`) is what closes the
TKE dissipation coefficient; see [`tke_dissipation_coefficient`](@ref) for that
derivation.
"""
function turbulent_prandtl_number(params, ᶜN²_eff, ᶜstrain_rate_norm)
    FT = eltype(params)
    turbconv_params = CAP.turbconv_params(params)
    eps_FT = eps(FT)

    # Parameters from CliMAParams
    Pr_n = CAP.Prandtl_number_0(turbconv_params) # Neutral Prandtl number
    ω_pr = CAP.Prandtl_number_scale(turbconv_params) # Prandtl number scale coefficient
    Pr_max = CAP.Pr_max(turbconv_params) # Maximum Prandtl number limit

    # Calculate the raw gradient Richardson number using the new helper function
    ᶜRi_grad = gradient_richardson_number(params, ᶜN²_eff, ᶜstrain_rate_norm)

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

The TKE dissipation coefficient `c_d = c_m c_b / Ri_c`, a derived closure
coefficient combining the eddy-viscosity coefficient `c_m` (`tke_ed_coeff`),
the static-stability coefficient `c_b` (`static_stab_coeff`), and the critical
gradient Richardson number `Ri_c` (`Ri_crit`). Used by [`tke_dissipation`](@ref).

# Derivation of `c_d = c_m c_b / Ri_c`

Consider the local TKE balance (production = buoyancy destruction +
dissipation, no transport) in stably stratified air, where the mixing length is
buoyancy-limited, `l = l_N = √(c_b e)/N`, with `e` the TKE and `N` the buoyancy
frequency:

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
[`turbulent_prandtl_number`](@ref)), so `1/Pr → 0` and turbulence is maintained
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

Calculates the blended mixing length scale based on the specified blending method.

This function dispatches to specific implementations based on the type of
`method`, which can be `SmoothMinimumBlending` or `HardMinimumBlending`. Each
method combines the physical scales (wall, TKE, stability) in a different way
to produce a single representative mixing length scale.

See also: `SmoothMinimumBlending`, `HardMinimumBlending`, `lamb_smooth_minimum`
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
    lamb_smooth_minimum(l::SA.SVector{N, FT}, smoothness_param::FT, λ_floor::FT) where {N, FT}

Calculates a smooth minimum of the elements in the StaticVector `l`.

This function provides a differentiable approximation to the `minimum` function,
yielding a value slightly larger than the true minimum, weighted towards the
smallest elements. The degree of smoothness is controlled by an internally
calculated parameter `λ₀`, which depends on the input parameters
`smoothness_param` and `λ_floor`. A larger `λ₀` results in a smoother
(less sharp) minimum approximation.

This implementation is based on an exponentially weighted average, with `λ₀`
determined involving the minimum element `x_min` and a factor related to the
Lambert W function evaluated at 2/e.

Arguments:

  - `l`: An `SVector{N, FT}` of N numbers for which to find the smooth minimum.
  - `smoothness_param`: A parameter (`FT`) influencing the scaling of the smoothness
    parameter `λ₀`. A larger value generally leads to a larger `λ₀`
    and a smoother minimum.
  - `λ_floor`: The minimum value (`FT`) allowed for the smoothness parameter `λ₀`.
    Ensures a minimum level of smoothing and prevents `λ₀` from
    becoming zero or negative. Must be positive.
    Returns:
  - The smooth minimum value (`FT`).

Algorithm:

 1. Find the hard minimum `x_min = minimum(l)`.
 2. Calculate the smoothness scale:
    `λ₀ = max(x_min * smoothness_param / W(2/e), λ_floor)`,
    where `W(2/e)` is the Lambert W function evaluated at 2/e.
 3. Ensure `λ₀` is positive (`>= eps(FT)`).
 4. Compute the exponentially weighted average:
    `smin = Σᵢ(lᵢ * exp(-(lᵢ - x_min) / λ₀)) / Σᵢ(exp(-(lᵢ - x_min) / λ₀))`
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

Calculates the eddy viscosity (K_u) for momentum based on the turbulent
kinetic energy (TKE) and the mixing length.

Returns K_u in units of [m^2/s].
"""
function eddy_viscosity(params, tke, mixing_length)
    c_m = CAP.tke_ed_coeff(params)
    return c_m * mixing_length * sqrt(max(tke, 0))
end

"""
    eddy_diffusivity(K_u, prandtl_number)

Calculates the eddy diffusivity (K_h) for scalars given the eddy viscosity (K_u)
and the turbulent Prandtl number.

Returns K_h in units of [m^2/s].
"""
function eddy_diffusivity(K_u, prandtl_number)
    return K_u / prandtl_number # prandtl_nvec is already bounded by eps_FT and Pr_max
end
