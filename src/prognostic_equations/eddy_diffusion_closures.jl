#####
##### TKE-based eddy diffusion closures
#####

import StaticArrays as SA
import Lux
import Thermodynamics.Parameters as TDP
import ClimaCore.Geometry as Geometry
import ClimaCore.Fields as Fields

const LINEAR_EQN1_FEATURE_ORDER = (
    :bgrad,
    :strain,
    :tke,
    :mix_len_pi1,
    :mix_len_pi2,
    :mix_len_pi3,
    :z_obu,
    :res_obu,
    :bgrad_slog1p,
    :strain_slog1p,
    :tke_slog1p,
    :mix_len_pi1_slog1p,
    :mix_len_pi2_slog1p,
    :mix_len_pi3_slog1p,
    :z_obu_slog1p,
    :res_obu_slog1p,
)

const LINEAR_EQN1_EXPECTED_PARAM_LENGTH =
    length(LINEAR_EQN1_FEATURE_ORDER) + 1
const SYM_EQN1_PARAM_COUNT = 5

const LINEAR_EQN1_MEANS = NamedTuple{LINEAR_EQN1_FEATURE_ORDER}((
    5.082637653686106e-05,
    2.2808653739048168e-05,
    0.24866291880607605,
    -1.0863544940948486,
    -14.831830978393555,
    1.0883108377456665,
    -39.594970703125,
    -4.873882293701172,
    5.082637524710884e-05,
    2.280865347833197e-05,
    0.222250648226703,
    -0.7352520975355456,
    -2.760693474229933,
    0.7372262688331963,
    -3.704753971779477,
    -1.7692407264060654,
))

const LINEAR_EQN1_STDS = NamedTuple{LINEAR_EQN1_FEATURE_ORDER}((
    0.0014844609540887177,
    0.0004074893149663694,
    2.8298795223236084,
    66.78991794586182,
    1325.919189453125,
    45.556983947753906,
    2584.10400390625,
    236.17761611938477,
    0.0014844609540887177,
    0.0004074893149663694,
    2.2657163633550213,
    32.02409003305618,
    83.7823805840645,
    21.81734874850377,
    63.66186165226027,
    40.21037228818965,
))

@inline symmetric_log1p(x) = sign(x) * log1p(abs(x))

@inline function convert_namedtuple(::Type{FT}, data::NamedTuple{N}) where {FT, N}
    return NamedTuple{N}(map(FT, Tuple(data)))
end

@inline function linear_eqn1_coefficients(::Type{FT}, vec) where {FT}
    n_features = length(LINEAR_EQN1_FEATURE_ORDER)
    expected_len = LINEAR_EQN1_EXPECTED_PARAM_LENGTH
    offset = if length(vec) == expected_len
        0
    elseif length(vec) >= SYM_EQN1_PARAM_COUNT + expected_len
        SYM_EQN1_PARAM_COUNT
    else
        error(
            "LinearEqn1MixingLengthType expects mixing_length_param_vec length " *
            "≥ $(expected_len) (or ≥ $(SYM_EQN1_PARAM_COUNT + expected_len) when sharing with SymEqn1); got $(length(vec)).",
        )
    end
    coeff_slice = @view vec[(offset + 1):(offset + n_features)]
    coeffs = NamedTuple{LINEAR_EQN1_FEATURE_ORDER}(ntuple(
        i -> FT(coeff_slice[i]),
        n_features,
    ))
    intercept = FT(vec[offset + expected_len])
    return coeffs, intercept
end


"""
    buoyancy_gradients(
        closure::AbstractEnvBuoyGradClosure,
        thermo_params,

        # Arguments for the first method (most commonly called):
        ts::TD.ThermodynamicState,
        ::Type{C3}, # Covariant3 vector type, for projecting gradients
        ∂qt∂z::AbstractField,   # Vertical gradient of total specific humidity
        ∂θli∂z::AbstractField,   # Vertical gradient of liquid-ice potential temperature
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
- `ts`: Center-level thermodynamic state of the environment.
- `C3`: The `ClimaCore.Geometry.Covariant3Vector` type, used for projecting input vertical gradients.
- `∂qt∂z`: Field of vertical gradients of total specific humidity.
- `∂θli∂z`: Field of vertical gradients of liquid-ice potential temperature.
- `local_geometry`: Field of local geometry at cell centers, used for gradient projection.
The second method takes a precomputed `EnvBuoyGradVars` object instead of `ts` and gradient fields.

Returns:
- `∂b∂z`: The mean vertical buoyancy gradient [s⁻²], as a field of scalars.
"""
function buoyancy_gradients end

function buoyancy_gradients(
    ebgc::AbstractEnvBuoyGradClosure,
    thermo_params,
    ts,
    cf,
    ::Type{C3},
    ∂qt∂z,
    ∂θli∂z,
    ᶜlg,
) where {C3}
    return buoyancy_gradients(
        ebgc,
        thermo_params,
        EnvBuoyGradVars(
            ts,
            cf,
            projected_vector_buoy_grad_vars(
                C3,
                ∂qt∂z,
                ∂θli∂z,
                ᶜlg,
            ),
        ),
    )
end

function buoyancy_gradients(
    ebgc::AbstractEnvBuoyGradClosure,
    thermo_params,
    bg_model::EnvBuoyGradVars,
)
    FT = eltype(bg_model)

    g = TDP.grav(thermo_params)
    Rv_over_Rd = TDP.Rv_over_Rd(thermo_params)
    R_v = TDP.R_v(thermo_params)

    ts = bg_model.ts
    ∂b∂θv = g / TD.virtual_pottemp(thermo_params, ts)

    T = TD.air_temperature(thermo_params, ts)
    λ = TD.liquid_fraction(thermo_params, ts)
    lh =
        λ * TD.latent_heat_vapor(thermo_params, T) +
        (1 - λ) * TD.latent_heat_sublim(thermo_params, T)
    cp_m = TD.cp_m(thermo_params, ts)
    q_sat = TD.q_vap_saturation(thermo_params, ts)
    q_tot = TD.total_specific_humidity(thermo_params, ts)
    θ = TD.dry_pottemp(thermo_params, ts)
    ∂b∂θli_unsat = ∂b∂θv * (1 + (Rv_over_Rd - 1) * q_tot)
    ∂b∂qt_unsat = ∂b∂θv * (Rv_over_Rd - 1) * θ
    ∂b∂θli_sat = (
        ∂b∂θv *
        (1 + Rv_over_Rd * (1 + lh / R_v / T) * q_sat - q_tot) /
        (1 + lh^2 / cp_m / R_v / T^2 * q_sat)
    )
    ∂b∂qt_sat =
        (lh / cp_m / T * ∂b∂θli_sat - ∂b∂θv) * θ

    ∂b∂z = buoyancy_gradient_chain_rule(
        ebgc,
        bg_model,
        thermo_params,
        ∂b∂θli_unsat,
        ∂b∂qt_unsat,
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
    mixing_length_lopez_gomez_2020(
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
- `scale_blending_method`: The method to use for blending physical scales.

Point-wise calculation of the turbulent mixing length, limited by physical constraints (wall distance,
TKE balance, stability) and grid resolution. Based on
Lopez‐Gomez, I., Cohen, Y., He, J., Jaruga, A., & Schneider, T. (2020).
A generalized mixing length closure for eddy‐diffusivity mass‐flux schemes of turbulence and convection.
Journal of Advances in Modeling Earth Systems, 12, e2020MS002161. https://doi.org/ 10.1029/2020MS002161

Returns a `MixingLength{FT}` struct containing the final blended mixing length (`master`)
and its constituent physical scales.
"""

function mixing_length_lopez_gomez_2020(
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
    scale_blending_method,
)

    FT = eltype(ᶜz)
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
    gryanik_a_m = CAP.coefficient_a_m_gryanik(params)  # Gryanik a_m
    gryanik_b_m = CAP.coefficient_b_m_gryanik(params)  # Gryanik b_m

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
        #     φ_m = 1 + a_m ζ / (1 + b_m ζ)^(2/3),
        # a nonlinear refinement to the Businger formulation.
        # Optimization: (1+b_m*ζ)^(2/3) -> cbrt((1+b_m*ζ)^2)
        # Guard against a negative base in the fractional power
        # (theoretically impossible for ζ ≥ 0 and b_m > 0, retained for robustness).
        phi_m_denom = max(cbrt((1 + gryanik_b_m * zeta)^2), eps_FT)

        phi_m = 1 + (gryanik_a_m * zeta) / phi_m_denom

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

    # Solve for l_TKE in
    #     a_pd · l_TKE − c_neg / l_TKE = 0
    #  ⇒  a_pd · l_TKE² − c_neg = 0
    # yielding
    #     l_TKE = √c_neg / a_pd.
    l_TKE = ifelse(a_pd > eps_FT, sqrt(c_neg / max(a_pd, eps_FT)), FT(0))

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

    return MixingLength(l_final, l_W, l_TKE, l_N, l_grid)
end

# GPU-safe field access using Val dispatch
@inline get_mixing_length_field(ml::MixingLength, ::Val{:master}) = ml.master
@inline get_mixing_length_field(ml::MixingLength, ::Val{:wall}) = ml.wall
@inline get_mixing_length_field(ml::MixingLength, ::Val{:tke}) = ml.tke
@inline get_mixing_length_field(ml::MixingLength, ::Val{:buoy}) = ml.buoy
@inline get_mixing_length_field(ml::MixingLength, ::Val{:l_grid}) = ml.l_grid




function ᶜmixing_length(Y, p, ::Val{:master})

    (; params) = p
    (; ustar, obukhov_length) = p.precomputed.sfc_conditions
    (; ᶜlinear_buoygrad, ᶜstrain_rate_norm) = p.precomputed
    ᶜz = Fields.coordinate_field(Y.c).z
    z_sfc = Fields.level(Fields.coordinate_field(Y.f).z, Fields.half)
    ᶜdz = Fields.Δz_field(axes(Y.c))
    turbconv_model = p.atmos.turbconv_model
    ᶜρa⁰ =
        turbconv_model isa PrognosticEDMFX ?
        (@. lazy(ρa⁰(Y.c.ρ, Y.c.sgsʲs, turbconv_model))) : Y.c.ρ
    ᶜtke⁰ = @. lazy(specific_tke(Y.c.ρ, Y.c.sgs⁰.ρatke, ᶜρa⁰, turbconv_model))
    sfc_tke = Fields.level(ᶜtke⁰, 1)
    ᶜprandtl_nvec = @. lazy(turbulent_prandtl_number(params, ᶜlinear_buoygrad, ᶜstrain_rate_norm))
    ᶜtke_exch = ᶜtke_exchange(Y, p)
    ᶜmixing_length_tuple = @. lazy(
        mixing_length_lopez_gomez_2020(
            params,
            ustar,
            ᶜz,
            z_sfc,
            ᶜdz,
            sfc_tke,
            ᶜlinear_buoygrad,
            ᶜtke⁰,
            obukhov_length,
            ᶜstrain_rate_norm,
            ᶜprandtl_nvec,
            ᶜtke_exch,
            p.atmos.edmfx_model.scale_blending_method,
        ),
    )
    return @. lazy(get_mixing_length_field(ᶜmixing_length_tuple, Val{:master}()))
end

function ᶜmixing_length(Y, p, ::Val{:sym_eqn1})
    # Symbolic Eqn1 closure (previous hack under master)
    (; params) = p
    (; ustar, obukhov_length) = p.precomputed.sfc_conditions
    (; ᶜlinear_buoygrad, ᶜstrain_rate_norm) = p.precomputed
    ᶜz = Fields.coordinate_field(Y.c).z
    z_sfc = Fields.level(Fields.coordinate_field(Y.f).z, Fields.half)
    ᶜdz = Fields.Δz_field(axes(Y.c))
    turbconv_model = p.atmos.turbconv_model
    ᶜρa⁰ =
        turbconv_model isa PrognosticEDMFX ?
        (@. lazy(ρa⁰(Y.c.ρ, Y.c.sgsʲs, turbconv_model))) : Y.c.ρ
    ᶜtke⁰ = @. lazy(specific_tke(Y.c.ρ, Y.c.sgs⁰.ρatke, ᶜρa⁰, turbconv_model))
    sfc_tke = Fields.level(ᶜtke⁰, 1)
    turbconv_params = CAP.turbconv_params(params)
    ml_param_vec = CAP.mixing_length_param_vec(turbconv_params)
    
    FT = eltype(ᶜz)
    eps_FT = eps(FT)
    l_z = max.(ᶜz .- z_sfc, FT(0))
    ᶜlg = Fields.local_geometry_field(Y.c)
    n_up = n_mass_flux_subdomains(turbconv_model)
    ᶜdelta_w_sq = nothing
    if turbconv_model isa PrognosticEDMFX || turbconv_model isa DiagnosticEDMFX
        (; ᶜu⁰, ᶜuʲs) = p.precomputed
        ᶜw_env = @. lazy(get_physical_w(ᶜu⁰, ᶜlg))
        if n_up > 0
            ᶜw_up = @. lazy(get_physical_w(ᶜuʲs.:1, ᶜlg))
        else
            ᶜw_up = ᶜw_env
        end
        ᶜdelta_w = @. lazy(ᶜw_up - ᶜw_env)
        ᶜdelta_w_sq = @. lazy(max(ᶜdelta_w * ᶜdelta_w, eps_FT))
    else
        ᶜdelta_w_sq = p.scratch.ᶜtemp_scalar_4
        @. ᶜdelta_w_sq = eps_FT
    end
    ᶜmix_len_pi3 = @. lazy(min(ᶜtke⁰ / ᶜdelta_w_sq, FT(100)))
    l0 = FT(ml_param_vec[1])
    a_lin = FT(ml_param_vec[2])
    b_lin = FT(ml_param_vec[3])
    k_tke_dw2 = FT(ml_param_vec[4])
    k_strain = FT(ml_param_vec[5])
    ᶜexp_arg = @. lazy(-k_tke_dw2 * ᶜmix_len_pi3 - k_strain * ᶜstrain_rate_norm)
    ᶜℓ_raw = @. lazy(l0 + (a_lin * ᶜtke⁰ + b_lin) * exp(ᶜexp_arg))
    ᶜℓ_pos = @. lazy(max(ᶜℓ_raw, FT(1)))
    ᶜℓ_clip = @. lazy(min(ᶜℓ_pos, l_z, ᶜdz))
    ᶜℓ = ᶜℓ_clip
    ᶜmixing_length_tuple = @. lazy(MixingLength(ᶜℓ, ᶜℓ, ᶜℓ, ᶜℓ, ᶜdz))
    return @. lazy(get_mixing_length_field(ᶜmixing_length_tuple, Val{:master}()))
end

function ᶜmixing_length(Y, p, ::Val{:linear_eqn1})
    (; params) = p
    (; obukhov_length) = p.precomputed.sfc_conditions
    (; ᶜlinear_buoygrad, ᶜstrain_rate_norm) = p.precomputed
    ᶜz = Fields.coordinate_field(Y.c).z
    z_sfc = Fields.level(Fields.coordinate_field(Y.f).z, Fields.half)
    ᶜdz = Fields.Δz_field(axes(Y.c))
    turbconv_model = p.atmos.turbconv_model
    ᶜρa⁰ =
        turbconv_model isa PrognosticEDMFX ?
        (@. lazy(ρa⁰(Y.c.ρ, Y.c.sgsʲs, turbconv_model))) : Y.c.ρ
    ᶜtke⁰ = @. lazy(specific_tke(Y.c.ρ, Y.c.sgs⁰.ρatke, ᶜρa⁰, turbconv_model))

    turbconv_params = CAP.turbconv_params(params)
    ml_param_vec = CAP.mixing_length_param_vec(turbconv_params)

    FT = eltype(ᶜz)
    eps_FT = eps(FT)

    coeffs, intercept = linear_eqn1_coefficients(FT, ml_param_vec)
    means = convert_namedtuple(FT, LINEAR_EQN1_MEANS)
    stds = convert_namedtuple(FT, LINEAR_EQN1_STDS)

    ᶜlg = Fields.local_geometry_field(Y.c)
    n_up = n_mass_flux_subdomains(turbconv_model)

    ᶜdelta_w_sq = nothing
    if turbconv_model isa PrognosticEDMFX || turbconv_model isa DiagnosticEDMFX
        (; ᶜu⁰, ᶜuʲs) = p.precomputed
        ᶜw_env = @. lazy(get_physical_w(ᶜu⁰, ᶜlg))
        if n_up > 0
            ᶜw_up = @. lazy(get_physical_w(ᶜuʲs.:1, ᶜlg))
        else
            ᶜw_up = ᶜw_env
        end
        ᶜdelta_w = @. lazy(ᶜw_up - ᶜw_env)
        ᶜdelta_w_sq = @. lazy(max(ᶜdelta_w * ᶜdelta_w, eps_FT))
    else
        ᶜdelta_w_sq = p.scratch.ᶜtemp_scalar_4
        @. ᶜdelta_w_sq = eps_FT
    end

    ᶜmix_len_pi1_raw =
        @. lazy(ᶜstrain_rate_norm / (ᶜlinear_buoygrad + eps_FT))
    ᶜmix_len_pi2_raw =
        @. lazy(ᶜtke⁰ / (ᶜlinear_buoygrad * ᶜz * ᶜz + eps_FT))
    ᶜmix_len_pi3_raw = @. lazy(ᶜtke⁰ / ᶜdelta_w_sq)

    ᶜmix_len_pi1 = @. lazy(clamp(ᶜmix_len_pi1_raw, -FT(100), FT(100)))
    ᶜmix_len_pi2 = @. lazy(clamp(ᶜmix_len_pi2_raw, -FT(1e4), FT(1e4)))
    ᶜmix_len_pi3 = @. lazy(clamp(ᶜmix_len_pi3_raw, -FT(100), FT(100)))
    ᶜbgrad = @. lazy(clamp(ᶜlinear_buoygrad, -FT(0.02), FT(0.02)))
    ᶜstrain = @. lazy(clamp(ᶜstrain_rate_norm, -FT(1e-3), FT(1e-3)))
    ᶜtke = ᶜtke⁰
    ᶜz_obu =
        @. lazy(clamp(ᶜz / (obukhov_length + eps_FT), -FT(3e4), FT(3e4)))
    ᶜres_obu =
        @. lazy(clamp(ᶜdz / (obukhov_length + eps_FT), -FT(2500), FT(2500)))

    ᶜbgrad_slog1p = @. lazy(symmetric_log1p(ᶜbgrad))
    ᶜstrain_slog1p = @. lazy(symmetric_log1p(ᶜstrain))
    ᶜtke_slog1p = @. lazy(symmetric_log1p(ᶜtke))
    ᶜmix_len_pi1_slog1p = @. lazy(symmetric_log1p(ᶜmix_len_pi1))
    ᶜmix_len_pi2_slog1p = @. lazy(symmetric_log1p(ᶜmix_len_pi2))
    ᶜmix_len_pi3_slog1p = @. lazy(symmetric_log1p(ᶜmix_len_pi3))
    ᶜz_obu_slog1p = @. lazy(symmetric_log1p(ᶜz_obu))
    ᶜres_obu_slog1p = @. lazy(symmetric_log1p(ᶜres_obu))

    ᶜz_bgrad = @. lazy((ᶜbgrad - means.bgrad) / stds.bgrad)
    ᶜz_strain = @. lazy((ᶜstrain - means.strain) / stds.strain)
    ᶜz_tke = @. lazy((ᶜtke - means.tke) / stds.tke)
    ᶜz_mix_len_pi1 =
        @. lazy((ᶜmix_len_pi1 - means.mix_len_pi1) / stds.mix_len_pi1)
    ᶜz_mix_len_pi2 =
        @. lazy((ᶜmix_len_pi2 - means.mix_len_pi2) / stds.mix_len_pi2)
    ᶜz_mix_len_pi3 =
        @. lazy((ᶜmix_len_pi3 - means.mix_len_pi3) / stds.mix_len_pi3)
    ᶜz_z_obu = @. lazy((ᶜz_obu - means.z_obu) / stds.z_obu)
    ᶜz_res_obu = @. lazy((ᶜres_obu - means.res_obu) / stds.res_obu)
    ᶜz_bgrad_slog1p =
        @. lazy((ᶜbgrad_slog1p - means.bgrad_slog1p) / stds.bgrad_slog1p)
    ᶜz_strain_slog1p =
        @. lazy((ᶜstrain_slog1p - means.strain_slog1p) / stds.strain_slog1p)
    ᶜz_tke_slog1p =
        @. lazy((ᶜtke_slog1p - means.tke_slog1p) / stds.tke_slog1p)
    ᶜz_mix_len_pi1_slog1p = @. lazy(
        (ᶜmix_len_pi1_slog1p - means.mix_len_pi1_slog1p) /
        stds.mix_len_pi1_slog1p,
    )
    ᶜz_mix_len_pi2_slog1p = @. lazy(
        (ᶜmix_len_pi2_slog1p - means.mix_len_pi2_slog1p) /
        stds.mix_len_pi2_slog1p,
    )
    ᶜz_mix_len_pi3_slog1p = @. lazy(
        (ᶜmix_len_pi3_slog1p - means.mix_len_pi3_slog1p) /
        stds.mix_len_pi3_slog1p,
    )
    ᶜz_z_obu_slog1p =
        @. lazy((ᶜz_obu_slog1p - means.z_obu_slog1p) / stds.z_obu_slog1p)
    ᶜz_res_obu_slog1p =
        @. lazy((ᶜres_obu_slog1p - means.res_obu_slog1p) / stds.res_obu_slog1p)

    ᶜlmix = @. lazy(
        intercept +
        coeffs.bgrad * ᶜz_bgrad +
        coeffs.strain * ᶜz_strain +
        coeffs.tke * ᶜz_tke +
        coeffs.mix_len_pi1 * ᶜz_mix_len_pi1 +
        coeffs.mix_len_pi2 * ᶜz_mix_len_pi2 +
        coeffs.mix_len_pi3 * ᶜz_mix_len_pi3 +
        coeffs.z_obu * ᶜz_z_obu +
        coeffs.res_obu * ᶜz_res_obu +
        coeffs.bgrad_slog1p * ᶜz_bgrad_slog1p +
        coeffs.strain_slog1p * ᶜz_strain_slog1p +
        coeffs.tke_slog1p * ᶜz_tke_slog1p +
        coeffs.mix_len_pi1_slog1p * ᶜz_mix_len_pi1_slog1p +
        coeffs.mix_len_pi2_slog1p * ᶜz_mix_len_pi2_slog1p +
        coeffs.mix_len_pi3_slog1p * ᶜz_mix_len_pi3_slog1p +
        coeffs.z_obu_slog1p * ᶜz_z_obu_slog1p +
        coeffs.res_obu_slog1p * ᶜz_res_obu_slog1p,
    )

    ᶜlmix_pos = @. lazy(max(ᶜlmix, FT(0)))
    l_z = @. lazy(max(ᶜz - z_sfc, FT(0)))
    ᶜℓ_clip = @. lazy(min(ᶜlmix_pos, l_z, ᶜdz))
    ᶜmixing_length_tuple = @. lazy(MixingLength(ᶜℓ_clip, ᶜℓ_clip, ᶜℓ_clip, ᶜℓ_clip, ᶜdz))
    return @. lazy(get_mixing_length_field(ᶜmixing_length_tuple, Val{:master}()))
end

function ᶜmixing_length(Y, p, ::Val{:nn})
    # Neural network closure (Lux-based)
    (; params, atmos) = p
    isnothing(atmos.turbconv.mixing_length_nn) && error("mixing_length_type=nn but no NN is initialized")
    nn = atmos.turbconv.mixing_length_nn
    (; ustar, obukhov_length) = p.precomputed.sfc_conditions
    (; ᶜlinear_buoygrad, ᶜstrain_rate_norm) = p.precomputed
    ᶜz = Fields.coordinate_field(Y.c).z
    z_sfc = Fields.level(Fields.coordinate_field(Y.f).z, Fields.half)
    ᶜdz = Fields.Δz_field(axes(Y.c))
    turbconv_model = p.atmos.turbconv_model
    ᶜρa⁰ =
        turbconv_model isa PrognosticEDMFX ?
        (@. lazy(ρa⁰(Y.c.ρ, Y.c.sgsʲs, turbconv_model))) : Y.c.ρ
    ᶜtke⁰ = @. lazy(specific_tke(Y.c.ρ, Y.c.sgs⁰.ρatke, ᶜρa⁰, turbconv_model))
    sfc_tke = Fields.level(ᶜtke⁰, 1)
    FT = eltype(ᶜz)
    eps_FT = eps(FT)
    ᶜlg = Fields.local_geometry_field(Y.c)
    n_up = n_mass_flux_subdomains(turbconv_model)
    ᶜw_env = @. lazy(get_physical_w(p.precomputed.ᶜu⁰, ᶜlg))
    if n_up > 0
        ᶜw_up = @. lazy(get_physical_w(ᶜuʲs.:1, ᶜlg))
    else
        ᶜw_up = ᶜw_env
    end
    l_z = @. lazy(max(ᶜz - z_sfc, FT(0)))
    ᶜdelta_w_sq = @. lazy(max((ᶜw_up - ᶜw_env) * (ᶜw_up - ᶜw_env), eps_FT))
    X_1 = @. lazy(ᶜstrain_rate_norm / (ᶜlinear_buoygrad + eps_FT))
    X_2 = @. lazy(ᶜtke⁰ / (ᶜlinear_buoygrad * ᶜz * ᶜz + eps_FT))
    X_3 = @. lazy(ᶜtke⁰ / (ᶜdelta_w_sq))
    X_4 = ᶜlinear_buoygrad
    X_5 = ᶜstrain_rate_norm
    X_6 = ᶜtke⁰
    X_7 = @. lazy(ᶜz / (obukhov_length + eps_FT))
    X_8 = @. lazy(ᶜdz / (obukhov_length + eps_FT))
    X_1 = @. lazy(clamp(X_1, -100.0, 100.0))
    X_2 = @. lazy(clamp(X_2, -1e4, 1e4))
    X_3 = @. lazy(clamp(X_3, -100.0, 100.0))
    X_4 = @. lazy(clamp(X_4, -0.02, 0.02))
    X_5 = @. lazy(clamp(X_5, -1e-3, 1e-3))
    X_7 = @. lazy(clamp(X_7, -3e4, 3e4))
    X_8 = @. lazy(clamp(X_8, -2500, 2500))
    means = (
        X1 = FT(0.08152589946985245),
        X2 = FT(-9.220643997192383),
        X3 = FT(1.456505298614502),
        X4 = FT(4.150567838223651e-05),
        X5 = FT(1.2203592632431537e-05),
        X6 = FT(0.1851629614830017),
        X7 = FT(-56.88902282714844),
        X8 = FT(-7.018703460693359),
    )
    stds = (
        X1 = FT(42.38632678985596),
        X2 = FT(1158.9794158935547),
        X3 = FT(55.166049003601074),
        X4 = FT(0.0010860399197554216),
        X5 = FT(0.0001870773485279642),
        X6 = FT(2.441168576478958),
        X7 = FT(4223.447265625),
        X8 = FT(414.67926025390625),
    )
    N1 = @. lazy((X_1 - means.X1) / stds.X1)
    N2 = @. lazy((X_2 - means.X2) / stds.X2)
    N3 = @. lazy((X_3 - means.X3) / stds.X3)
    N4 = @. lazy((X_4 - means.X4) / stds.X4)
    N5 = @. lazy((X_5 - means.X5) / stds.X5)
    N6 = @. lazy((X_6 - means.X6) / stds.X6)
    N7 = @. lazy((X_7 - means.X7) / stds.X7)
    N8 = @. lazy((X_8 - means.X8) / stds.X8)
    function eval_nn(n1, n2, n3, n4, n5, n6, n7, n8)
        x = SA.SVector(n1, n2, n3, n4, n5, n6, n7, n8)
        y, _ = Lux.apply(nn.model, x, nn.params, nn.state)
        return y[1]
    end
    l_norm = @. lazy(eval_nn(N1, N2, N3, N4, N5, N6, N7, N8))
    l_pred = @. lazy(max(FT(l_norm) * FT(510.1035690307617) + FT(36.83180618286133), FT(0)))
    N_eff = @. lazy(sqrt(max(ᶜlinear_buoygrad, 0)))
    ᶜprandtl_nvec = @. lazy(turbulent_prandtl_number(params, ᶜlinear_buoygrad, ᶜstrain_rate_norm))
    l_smag = @. lazy(smagorinsky_lilly_length(CAP.c_smag(params), N_eff, ᶜdz, ᶜprandtl_nvec, ᶜstrain_rate_norm))
    l_clip = @. lazy(max(l_smag, min(l_pred, l_z)))
    ℓ = l_clip
    ᶜmixing_length_tuple = @. lazy(MixingLength(ℓ, ℓ, ℓ, ℓ, ᶜdz))
    return @. lazy(get_mixing_length_field(ᶜmixing_length_tuple, Val{:master}()))
end

function ᶜmixing_length(Y, p, property::Val{P} = Val{:master}()) where {P}
    # Dispatcher mapping config type to mixing-length variant
    t = p.atmos.turbconv.mixing_length_type
    if t isa PhysicalMixingLengthType
        return ᶜmixing_length(Y, p, Val{:master}())
    elseif t isa SymEqn1MixingLengthType
        return ᶜmixing_length(Y, p, Val{:sym_eqn1}())
    elseif t isa LinearEqn1MixingLengthType
        return ᶜmixing_length(Y, p, Val{:linear_eqn1}())
    elseif t isa NeuralNetworkMixingLengthType
        return ᶜmixing_length(Y, p, Val{:nn}())
    else
        error("Unsupported mixing length type $(typeof(t))")
    end
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
