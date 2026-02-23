# ============================================================================
# SGS-Aware Saturation Adjustment
# ============================================================================
# Computes saturation-adjusted thermodynamic state (T, q_liq, q_ice) by
# integrating over the joint PDF of (T, q_tot), accounting for subgrid-scale
# fluctuations. Used with EquilMoistModel for physical
# consistency between cloud fraction and condensate.

import Thermodynamics as TD

# ============================================================================
# Saturation Adjustment Evaluator
# ============================================================================

"""
    SaturationAdjustmentEvaluator

GPU-safe functor for computing saturation-adjusted state at quadrature points.

At each quadrature point `(T_hat, q_hat)`:
1. Computes saturation specific humidity at `(T_hat, ρ)`
2. Determines condensate as saturation excess
3. Partitions condensate using liquid fraction `λ(T_hat)`

This is the equilibrium thermodynamics equivalent of the
`DiagnosticCloudEvaluator` used for cloud fraction, but returns
the full thermodynamic state.

# Fields
- `thermo_params`: Thermodynamics parameters
- `ρ`: Air density [kg/m³]
"""
struct SaturationAdjustmentEvaluator{TPS, T1}
    thermo_params::TPS
    ρ::T1
end

"""
    (eval::SaturationAdjustmentEvaluator)(T_hat, q_hat)

Compute saturation-adjusted state at a single quadrature point.

# Arguments
- `T_hat`: Temperature at quadrature point [K]
- `q_hat`: Total specific humidity at quadrature point [kg/kg]

# Returns
NamedTuple with:
- `T`: Temperature at quadrature point [K]
- `q_liq`: Liquid condensate [kg/kg]
- `q_ice`: Ice condensate [kg/kg]
"""
@inline function (eval::SaturationAdjustmentEvaluator)(T_hat, q_hat)
    FT = typeof(q_hat)
    thp = eval.thermo_params

    # Compute saturation specific humidity at (T_hat, ρ)
    # Note: Using dry saturation (q_liq=q_ice=0) since we're computing
    # the equilibrium condensate from scratch at this quadrature point
    q_sat = TD.q_vap_saturation(thp, T_hat, eval.ρ)

    # Condensate is the saturation excess (positive only)
    q_cond = max(FT(0), q_hat - q_sat)

    # Partition condensate using liquid fraction based on temperature ramp
    # liquid_fraction_ramp is appropriate for equilibrium thermodynamics
    λ = TD.liquid_fraction_ramp(thp, T_hat)
    q_liq = λ * q_cond
    q_ice = (FT(1) - λ) * q_cond

    return (; T = T_hat, q_liq, q_ice)
end

# ============================================================================
# SGS Saturation Adjustment Integration
# ============================================================================

"""
    compute_sgs_saturation_adjustment(
        thermo_params, SG_quad, ρ, T_mean, q_mean, T′T′, q′q′, corr_Tq
    )

Compute SGS-averaged saturation adjustment by integrating over the joint PDF
of `(T, q_tot)`.

This function replaces the grid-mean saturation adjustment when using
`QuadratureCloud` or `MLCloud` with `EquilMoistModel`, ensuring that cloud condensate
is computed consistently with cloud fraction.

# Arguments
- `thermo_params`: Thermodynamics parameters
- `SG_quad`: `SGSQuadrature` configuration
- `ρ`: Air density [kg/m³]
- `T_mean`: Grid-mean temperature [K]
- `q_mean`: Grid-mean total specific humidity [kg/kg]
- `T′T′`: Temperature variance [K²]
- `q′q′`: Moisture variance [(kg/kg)²]
- `corr_Tq`: Correlation coefficient ρ(T', q')

# Returns
NamedTuple with SGS-averaged:
- `T`: Grid-mean temperature [K]
- `q_liq`: Liquid condensate [kg/kg]
- `q_ice`: Ice condensate [kg/kg]

# Notes
The returned `T` is the grid-mean temperature. A more sophisticated
implementation could compute a perturbed T based on latent heat release,
but this requires iteration. For now, we only average the condensate.
"""
@inline function compute_sgs_saturation_adjustment(
    thermo_params,
    SG_quad::SGSQuadrature,
    ρ,
    T_mean,
    q_mean,
    T′T′,
    q′q′,
    corr_Tq,
)
    # Create GPU-safe functor (not a closure)
    evaluator = SaturationAdjustmentEvaluator(thermo_params, ρ)

    # Integrate over quadrature points
    result = integrate_over_sgs(evaluator, SG_quad, q_mean, T_mean, q′q′, T′T′, corr_Tq)

    # Return with grid-mean T (condensate is SGS-averaged)
    return (; T = T_mean, q_liq = result.q_liq, q_ice = result.q_ice)
end
