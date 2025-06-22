#####
##### TKE-based eddy diffusion closures
#####

import StaticArrays as SA
import Thermodynamics.Parameters as TDP
import ClimaCore.Geometry as Geometry
import ClimaCore.Fields as Fields

get_t_sat(thermo_params, x::EnvBuoyGradVars) =
    TD.air_temperature(thermo_params, x.ts)
get_p(thermo_params, x::EnvBuoyGradVars) = TD.air_pressure(thermo_params, x.ts)
get_ρ(thermo_params, x::EnvBuoyGradVars) = TD.air_density(thermo_params, x.ts)
get_en_cld_frac(thermo_params, x::EnvBuoyGradVars) =
    ifelse(TD.has_condensate(thermo_params, x.ts), 1, 0)
get_θ_liq_ice_sat(thermo_params, x::EnvBuoyGradVars) =
    TD.liquid_ice_pottemp(thermo_params, x.ts)
get_qt_sat(thermo_params, x::EnvBuoyGradVars) =
    TD.total_specific_humidity(thermo_params, x.ts)
get_ql_sat(thermo_params, x::EnvBuoyGradVars) =
    TD.liquid_specific_humidity(thermo_params, x.ts)
get_qi_sat(thermo_params, x::EnvBuoyGradVars) =
    TD.ice_specific_humidity(thermo_params, x.ts)
get_qv_sat(thermo_params, x::EnvBuoyGradVars) =
    TD.vapor_specific_humidity(thermo_params, x.ts)
get_θ_sat(thermo_params, x::EnvBuoyGradVars) =
    TD.dry_pottemp(thermo_params, x.ts)
get_∂θli∂z_sat(_, x::EnvBuoyGradVars) = x.∂θli∂z_sat
get_∂qt∂z_sat(_, x::EnvBuoyGradVars) = x.∂qt∂z_sat
get_∂θv∂z_unsat(_, x::EnvBuoyGradVars) = x.∂θv∂z_unsat

"""
    buoyancy_gradients(
        closure::AbstractEnvBuoyGradClosure,
        thermo_params,
        moisture_model,
        # Arguments for the first method (most commonly called):
        ts::TD.ThermodynamicState,
        ::Type{C3}, # Covariant3 vector type, for projecting gradients
        ∂θv∂z_unsat::AbstractField, # Vertical gradient of virtual potential temperature in unsaturated part
        ∂qt∂z_sat::AbstractField,   # Vertical gradient of total specific humidity in saturated part
        ∂θli∂z_sat::AbstractField,   # Vertical gradient of liquid-ice potential temperature in saturated part
        local_geometry::Fields.LocalGeometry,
        # Argument for the second method (internal use with precomputed EnvBuoyGradVars):
        # bg_model::EnvBuoyGradVars
    )

Calculates the mean vertical buoyancy gradient (`∂b/∂z`) in the environment.

This gradient is determined by considering contributions from both the unsaturated
and saturated portions of the environment, weighted by the environmental cloud
fraction. The calculation involves:
1. Determining partial derivatives of buoyancy with respect to virtual potential
   temperature (`θᵥ`) for the unsaturated part, and with respect to liquid-ice
   potential temperature (`θₗᵢ`) and total specific humidity (`qₜ`) for the
   saturated part.
2. Applying the chain rule using the provided vertical gradients of these
   thermodynamic variables (`∂θᵥ/∂z`, `∂θₗᵢ/∂z`, `∂qₜ/∂z`), obtained from
   the input fields after projection.
3. Blending the resulting unsaturated and saturated buoyancy gradients based on
   the environmental cloud fraction derived from the thermodynamic state `ts`.

The dispatch on `EnvBuoyGradVars` (used internally by the first method after
constructing it from `ts` and projected gradients) occurs during its construction.
The analytical solutions employed are consistent for both mean fields and
conditional fields derived from assumed distributions over conserved thermodynamic
variables.

Arguments:
- `closure`: The environmental buoyancy gradient closure type (e.g., `BuoyGradMean`).
- `thermo_params`: Thermodynamic parameters from `CLIMAParameters`.
- `moisture_model`: Moisture model (e.g., `EquilMoistModel`, `NonEquilMoistModel`).
- `ts`: Center-level thermodynamic state of the environment.
- `C3`: The `ClimaCore.Geometry.Covariant3Vector` type, used for projecting input vertical gradients.
- `∂θv∂z_unsat`: Field of vertical gradients of virtual potential temperature in the unsaturated part.
- `∂qt∂z_sat`: Field of vertical gradients of total specific humidity in the saturated part.
- `∂θli∂z_sat`: Field of vertical gradients of liquid-ice potential temperature in the saturated part.
- `local_geometry`: Field of local geometry at cell centers, used for gradient projection.
The second method takes a precomputed `EnvBuoyGradVars` object instead of `ts` and gradient fields.

Returns:
- `∂b∂z`: The mean vertical buoyancy gradient [s⁻²], as a field of scalars.
"""
function buoyancy_gradients end

function buoyancy_gradients(
    ebgc::AbstractEnvBuoyGradClosure,
    thermo_params,
    moisture_model,
    ts,
    ::Type{C3},
    ∂θv∂z_unsat,
    ∂qt∂z_sat,
    ∂θli∂z_sat,
    ᶜlg,
) where {C3}
    return buoyancy_gradients(
        ebgc,
        thermo_params,
        moisture_model,
        EnvBuoyGradVars(
            ts,
            projected_vector_buoy_grad_vars(
                C3,
                ∂θv∂z_unsat,
                ∂qt∂z_sat,
                ∂θli∂z_sat,
                ᶜlg,
            ),
        ),
    )
end

function buoyancy_gradients(
    ebgc::AbstractEnvBuoyGradClosure,
    thermo_params,
    moisture_model,
    bg_model::EnvBuoyGradVars,
)
    FT = eltype(bg_model)

    g = TDP.grav(thermo_params)
    molmass_ratio = TDP.molmass_ratio(thermo_params)
    R_d = TDP.R_d(thermo_params)
    R_v = TDP.R_v(thermo_params)

    phase_part = TD.PhasePartition(FT(0), FT(0), FT(0)) # assuming R_m = R_d
    p = get_p(thermo_params, bg_model)
    Π = TD.exner_given_pressure(thermo_params, p, phase_part)

    ∂b∂θv = g * (R_d * get_ρ(thermo_params, bg_model) / p) * Π
    θ_liq_ice_sat = get_θ_liq_ice_sat(thermo_params, bg_model)
    qt_sat = get_qt_sat(thermo_params, bg_model)

    if get_en_cld_frac(thermo_params, bg_model) > 0.0
        ts_sat = if moisture_model isa DryModel
            TD.PhaseDry_pθ(thermo_params, p, θ_liq_ice_sat)
        elseif moisture_model isa EquilMoistModel
            TD.PhaseEquil_pθq(thermo_params, p, θ_liq_ice_sat, qt_sat)
        elseif moisture_model isa NonEquilMoistModel
            TD.PhaseNonEquil_pθq(
                thermo_params,
                p,
                θ_liq_ice_sat,
                TD.PhasePartition(
                    qt_sat,
                    get_ql_sat(thermo_params, bg_model),
                    get_qi_sat(thermo_params, bg_model),
                ),
            )
        else
            error("Unsupported moisture model")
        end

        phase_part = TD.PhasePartition(thermo_params, ts_sat)
        lh = TD.latent_heat_liq_ice(thermo_params, phase_part)
        cp_m = TD.cp_m(thermo_params, ts_sat)
        t_sat = get_t_sat(thermo_params, bg_model)
        qv_sat = get_qv_sat(thermo_params, bg_model)
        ∂b∂θli_sat = (
            ∂b∂θv *
            (1 + molmass_ratio * (1 + lh / R_v / t_sat) * qv_sat - qt_sat) /
            (1 + lh * lh / cp_m / R_v / t_sat / t_sat * qv_sat)
        )
        ∂b∂qt_sat =
            (lh / cp_m / t_sat * ∂b∂θli_sat - ∂b∂θv) *
            get_θ_sat(thermo_params, bg_model)
    else
        ∂b∂θli_sat = FT(0)
        ∂b∂qt_sat = FT(0)
    end

    ∂b∂z = buoyancy_gradient_chain_rule(
        ebgc,
        bg_model,
        thermo_params,
        ∂b∂θv,
        ∂b∂θli_sat,
        ∂b∂qt_sat,
    )
    return ∂b∂z
end

"""
    buoyancy_gradient_chain_rule(
        closure::AbstractEnvBuoyGradClosure,
        bg_model::EnvBuoyGradVars,
        thermo_params,
        ∂b∂θv::FT,
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
(obtained from `bg_model` via `get_∂θv∂z_unsat`, `get_∂θli∂z_sat`, `get_∂qt∂z_sat`)
to get the buoyancy gradients for the unsaturated (`∂b∂z_unsat`) and saturated
(`∂b∂z_sat`) parts of the environment.
Finally, it returns a single mean buoyancy gradient by linearly combining
`∂b∂z_unsat` and `∂b∂z_sat` weighted by the environmental cloud fraction
(also obtained from `bg_model` via `get_en_cld_frac`).

Arguments:
- `closure`: The environmental buoyancy gradient closure type.
- `bg_model`: Precomputed environmental buoyancy gradient variables (`EnvBuoyGradVars`).
- `thermo_params`: Thermodynamic parameters from `CLIMAParameters`.
- `∂b∂θv`: Partial derivative of buoyancy w.r.t. virtual potential temperature (unsaturated part).
- `∂b∂θli_sat`: Partial derivative of buoyancy w.r.t. liquid-ice potential temperature (saturated part).
- `∂b∂qt_sat`: Partial derivative of buoyancy w.r.t. total specific humidity (saturated part).

Returns:
- `∂b∂z`: The mean vertical buoyancy gradient [s⁻²].
"""
function buoyancy_gradient_chain_rule(
    ::AbstractEnvBuoyGradClosure,
    bg_model::EnvBuoyGradVars,
    thermo_params,
    ∂b∂θv,
    ∂b∂θli_sat,
    ∂b∂qt_sat,
)
    FT = eltype(thermo_params)
    en_cld_frac = get_en_cld_frac(thermo_params, bg_model)
    if en_cld_frac > FT(0)
        ∂b∂z_θl_sat = ∂b∂θli_sat * get_∂θli∂z_sat(thermo_params, bg_model)
        ∂b∂z_qt_sat = ∂b∂qt_sat * get_∂qt∂z_sat(thermo_params, bg_model)
    else
        ∂b∂z_θl_sat = FT(0)
        ∂b∂z_qt_sat = FT(0)
    end

    ∂b∂z_unsat =
        en_cld_frac < FT(1) ? ∂b∂θv * get_∂θv∂z_unsat(thermo_params, bg_model) :
        FT(0)

    ∂b∂z_sat = ∂b∂z_θl_sat + ∂b∂z_qt_sat
    ∂b∂z = (1 - en_cld_frac) * ∂b∂z_unsat + en_cld_frac * ∂b∂z_sat

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
    ρa_sfc,
    ustar,
    surface_local_geometry,
)

    c_k = CAP.tke_surf_flux_coeff(turbconv_params)
    # Determine the direction of the flux (normal to the surface)
    # c3_unit is a unit vector in the direction of the surface normal (e.g., C3(0,0,1) for a flat surface)
    c3_unit = C3(unit_basis_vector_data(C3, surface_local_geometry))
    return c_k * ρa_sfc * ustar^3 * c3_unit
end

"""
    mixing_length(
    params,
    ustar,
    ᶜz,
    z_sfc,
    ᶜdz,
    sfc_tke,
    ᶜN²_eff,
    ᶜtke,
    obukhov_length,
    ᶜstrain_rate_norm,
    ᶜPr,
    ᶜtke_exch,
    scale_blending_method,
)

where:
- `params`: Parameter set (e.g., CLIMAParameters.AbstractParameterSet).
- `ustar`: Friction velocity [m/s].
- `ᶜz`: Cell center height [m].
- `z_sfc`: Surface elevation [m].
- `ᶜdz`: Cell vertical thickness [m].
- `sfc_tke`: TKE near the surface (e.g., first cell center) [m^2/s^2].
- `ᶜN²_eff`: Effective squared Brunt-Väisälä frequency [1/s^2].
- `ᶜtke`: Turbulent kinetic energy at cell center [m^2/s^2].
- `obukhov_length`: Surface Monin-Obukhov length [m].
- `ᶜstrain_rate_norm`: Frobenius norm of strain rate tensor [1/s].
- `ᶜPr`: Turbulent Prandtl number [-].
- `ᶜtke_exch`: TKE exchange term [m^2/s^3].
- `scale_blending_method`: The method to use for blending physical scales.

Calculates the turbulent mixing length, limited by physical constraints (wall distance,
TKE balance, stability) and grid resolution. 

Returns a `MixingLength{FT}` struct containing the final blended mixing length (`master`) 
and its constituent physical scales.
"""
function mixing_length(
    params,
    ustar,
    ᶜz,
    z_sfc,
    ᶜdz,
    sfc_tke,
    ᶜN²_eff,
    ᶜtke,
    obukhov_length,
    ᶜstrain_rate_norm,
    ᶜPr,
    ᶜtke_exch,
    scale_blending_method,
)

    FT = eltype(params)
    eps_FT = eps(FT)

    turbconv_params = CAP.turbconv_params(params)
    sf_params = CAP.surface_fluxes_params(params) # Businger params

    c_m = CAP.tke_ed_coeff(turbconv_params)
    c_d = CAP.tke_diss_coeff(turbconv_params)
    smin_ub = CAP.smin_ub(turbconv_params)
    smin_rm = CAP.smin_rm(turbconv_params)
    c_b = CAP.static_stab_coeff(turbconv_params)
    vkc = CAP.von_karman_const(params)

    # MOST stability function coefficients
    most_a_m = sf_params.ufp.a_m # Businger a_m
    most_b_m = sf_params.ufp.b_m # Businger b_m
    most_g_m = CAP.coefficient_b_m_gryanik(params)  # Gryanik b_m

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

    if obukhov_length < FT(0) # Unstable case
        obukhov_len_safe = min(obukhov_length, -eps_FT) # Ensure L < 0
        zeta = l_z / obukhov_len_safe # Stability parameter zeta = z/L (<0)

        # Calculate MOST term (1 - b_m * zeta)
        # Since zeta is negative, this term is > 1
        inner_term = 1 - most_b_m * zeta

        # Numerical safety check – by theory the value is ≥ 1.
        inner_term_safe = max(inner_term, eps_FT)

        # Unstable-regime correction factor:
        #     (1 − b_m ζ)^(1/4) = φ_m⁻¹,
        # where φ_m is the Businger stability function φ_m = (1 − b_m ζ)^(-1/4).
        stability_correction = sqrt(sqrt(inner_term_safe))
        l_W = l_W_base * stability_correction

    else # Neutral or stable case
        # Ensure L > 0 for Monin-Obukhov length
        obukhov_len_safe_stable = max(obukhov_length, eps_FT)
        zeta = l_z / obukhov_len_safe_stable # zeta >= 0

        # Stable/neutral-regime correction after Gryanik (2020):
        #     φ_m = 1 + a_m ζ / (1 + g_m ζ)^(2/3),
        # a nonlinear refinement to the Businger formulation.
        phi_m_denom_term = (1 + most_g_m * zeta)
        # Guard against a negative base in the fractional power
        # (theoretically impossible for ζ ≥ 0 and g_m > 0, retained for robustness).
        phi_m_denom_cubed_sqrt = cbrt(phi_m_denom_term)
        phi_m_denom =
            max(phi_m_denom_cubed_sqrt * phi_m_denom_cubed_sqrt, eps_FT) # (val)^(2/3)

        phi_m = 1 + (most_a_m * zeta) / phi_m_denom

        # Stable-regime correction factor: 1 / φ_m.
        # phi_m should be >= 1 for stable/neutral
        stability_correction = 1 / max(phi_m, eps_FT)

        # Apply the correction factor
        l_W = l_W_base * stability_correction
    end
    l_W = max(l_W, FT(0)) # Ensure non-negative

    # --- l_TKE: TKE production-dissipation balance scale ---
    tke_pos = max(ᶜtke, FT(0)) # Ensure TKE is not negative
    sqrt_tke_pos = sqrt(tke_pos)

    # Net production of TKE from shear and buoyancy is approximated by
    #     (S² − N²/Pr_t) · √TKE · l,
    # where S² denotes the gradient involved in shear production and
    # N²/Pr_t denotes the gradient involved in buoyancy production.
    # The factor below corresponds to that production term normalised by l.
    a_pd = c_m * (2 * ᶜstrain_rate_norm - ᶜN²_eff / ᶜPr) * sqrt_tke_pos

    # Dissipation is modelled as c_d · k^{3/2} / l.
    # For the quadratic expression below, c_neg ≡ c_d · k^{3/2}.
    c_neg = c_d * tke_pos * sqrt_tke_pos

    l_TKE = FT(0)
    # Solve for l_TKE in
    #     a_pd · l_TKE − c_neg / l_TKE + ᶜtke_exch = 0
    #  ⇒  a_pd · l_TKE² + ᶜtke_exch · l_TKE − c_neg = 0
    # yielding
    #     l_TKE = (−ᶜtke_exch ± √(ᶜtke_exch² + 4 a_pd c_neg)) / (2 a_pd).
    if abs(a_pd) > eps_FT # If net of shear and buoyancy production (a_pd) is non-zero
        discriminant = ᶜtke_exch * ᶜtke_exch + 4 * a_pd * c_neg
        if discriminant >= FT(0) # Ensure real solution exists
            # Select the physically admissible (positive) root for l_TKE.
            # When a_pd > 0 (production exceeds dissipation) the root
            #     (−ᶜtke_exch + √D) / (2 a_pd)
            # is positive.  For a_pd < 0 the opposite root is required.
            l_TKE_sol1 = (-(ᶜtke_exch) + sqrt(discriminant)) / (2 * a_pd)
            # For a_pd < 0 (local destruction exceeds production) use
            #     (−ᶜtke_exch − √D) / (2 a_pd).
            if a_pd > FT(0)
                l_TKE = l_TKE_sol1
            else # a_pd < FT(0)
                l_TKE = (-(ᶜtke_exch) - sqrt(discriminant)) / (2 * a_pd)
            end
            l_TKE = max(l_TKE, FT(0)) # Ensure it's non-negative
        end
    elseif abs(ᶜtke_exch) > eps_FT # If a_pd is zero, balance is between exchange and dissipation
        # ᶜtke_exch = c_neg / l_TKE  => l_TKE = c_neg / ᶜtke_exch
        # Ensure division is safe and result is positive
        if ᶜtke_exch > eps_FT # Assuming positive exchange means TKE sink from env perspective
            l_TKE = c_neg / ᶜtke_exch # if c_neg is positive, l_TKE is positive
        elseif ᶜtke_exch < -eps_FT # Negative exchange means TKE source for env
            # -|ᶜtke_exch| = c_neg / l_TKE. If c_neg > 0, this implies l_TKE < 0, which is unphysical.
            # This case (a_pd=0, tke_exch < 0, c_neg > 0) implies TKE source and dissipation, no production.
            # Dissipation = Source. So, c_d * k_sqrt_k / l = -tke_exch. l = c_d * k_sqrt_k / (-tke_exch)
            l_TKE = c_neg / (-(ᶜtke_exch))
        end
        l_TKE = max(l_TKE, FT(0))
    end
    # If a_pd = 0 and ᶜtke_exch = 0 (or c_neg = 0), l_TKE remains zero.

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
    l_physical_scales = SA.SVector(l_W, l_TKE, l_N)

    l_smin =
        blend_scales(scale_blending_method, l_physical_scales, turbconv_params)

    # 1. Limit the combined physical scale by the distance from the wall.
    #    This step mitigates excessive values of l_W or l_TKE.
    l_limited_phys_wall = min(l_smin, l_z)

    # 2. Impose the grid-scale limit (TODO: replace by volumetric grid scale)
    l_grid = ᶜdz   # TODO include costant rescaling factor
    l_final = min(l_limited_phys_wall, l_grid)

    # Final check: guarantee that the mixing length is at least a small positive
    # value.  This prevents division-by-zero in
    #     ε_d = C_d · TKE^{3/2} / l_mix
    # when TKE > 0.  When TKE = 0, l_mix is inconsequential, but eps_FT
    # provides a conservative lower bound.
    # minimum mixing length
    l_final = max(l_final, FT(1)) # TODO: make a climaparam

    return MixingLength{FT}(l_final, l_W, l_TKE, l_N, l_grid)
end

function master_mixing_length(    
    params,
    ustar,
    ᶜz,
    z_sfc,
    ᶜdz,
    sfc_tke,
    ᶜN²_eff,
    ᶜtke,
    obukhov_length,
    ᶜstrain_rate_norm,
    ᶜPr,
    ᶜtke_exch,
    scale_blending_method,
)

    ᶜmixing_length_tuple = @. lazy(mixing_length(
        params,
        ustar,
        ᶜz,
        z_sfc,
        ᶜdz,
        max(sfc_tke, eps(FT)),
        ᶜlinear_buoygrad,
        max(ᶜtke⁰, 0),
        obukhov_length,
        ᶜstrain_rate_norm,
        ᶜprandtl_nvec,
        ᶜtke_exch,
        scale_blending_method,
    ))

    return getproperty(ᶜmixing_length_tuple, :master)
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
    eddy_viscosity(turbconv_params, tke, mixing_length)

Calculates the eddy viscosity (K_u) for momentum based on the turbulent
kinetic energy (TKE) and the mixing length.

Returns K_u in units of [m^2/s].
"""
function eddy_viscosity(turbconv_params, tke, mixing_length)
    FT = typeof(tke)
    c_m = CAP.tke_ed_coeff(turbconv_params)
    return c_m * mixing_length * sqrt(max(tke, FT(0)))
end

"""
    eddy_diffusivity(K_u, prandtl_nvec)

Calculates the eddy diffusivity (K_h) for scalars given the
eddy viscosity (K_u) and the turbulent Prandtl number.

Returns K_h in units of [m^2/s].
"""

function eddy_diffusivity(K_u, prandtl_nvec)
    return K_u / prandtl_nvec # prandtl_nvec is already bounded by eps_FT and Pr_max
end


"""
    eddy_diffusivity(turbconv_params, tke, mixing_length, prandtl_nvec)

Calculates the eddy diffusivity (K_h) for scalars given turbulent kinetic energy (TKE),
the mixing length, and the turbulent Prandtl number.

Returns K_h in units of [m^2/s].
"""
function eddy_diffusivity(turbconv_params, tke, mixing_length, prandtl_nvec)
    K_u = eddy_viscosity(turbconv_params, tke, mixing_length)
    return K_u / prandtl_nvec
end
