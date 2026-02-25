# ============================================================================
# SGS-Aware Saturation Adjustment
# ============================================================================
# Computes saturation-adjusted thermodynamic state (T, q_liq, q_ice) by
# integrating over the joint PDF of (T, q_tot), accounting for subgrid-scale
# fluctuations. Used with EquilibriumMicrophysics0M for physical
# consistency between cloud fraction and condensate.

import Thermodynamics as TD

# ============================================================================
# Quadrature Point Evaluator
# ============================================================================

"""
    SaturationAdjustmentEvaluator

GPU-safe functor for computing saturation-adjusted state at individual
quadrature points (T_hat, q_hat). Used by `integrate_over_sgs` for both
saturation adjustment and 0-moment microphysics.

Given (T_hat, q_hat) at a quadrature point, computes:
1. Saturation specific humidity q_sat from Clausius-Clapeyron
2. Condensate as saturation excess: q_cond = max(0, q_hat - q_sat)
3. Liquid/ice partition using temperature-based ramp function

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
- `q_tot_quad`: Total specific humidity at quadrature point [kg/kg]
"""
@noinline function (eval::SaturationAdjustmentEvaluator)(T_hat, q_hat)
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

    # Return q_tot_quad = q_hat so the caller can compute the effective
    # integrated mean of the (possibly truncated) q_tot distribution.
    return (; T = T_hat, q_liq, q_ice, q_tot_quad = q_hat)
end

# ============================================================================
# SGS Saturation Adjustment Integration
# ============================================================================

"""
    compute_sgs_saturation_adjustment(
        thermo_params, SG_quad, ρ, T_mean, q_mean, T′T′, q′q′, corr_Tq,
    )

Compute SGS-averaged saturation adjustment by integrating over the joint PDF
of (T, q_tot). At each quadrature point, condensate is diagnosed from
saturation excess, providing a sub-grid-aware estimate of cloud condensate.

This function replaces the grid-mean saturation adjustment when using
`QuadratureCloud` or `MLCloud` with `EquilibriumMicrophysics0M`, ensuring that cloud condensate
is computed consistently with cloud fraction.

# Weight Adjustment for Truncated Distribution

When quadrature points for `q_tot` are clamped to zero (because they sample
the negative tail of the distribution), the integrated mean `q̃_mean` exceeds
`q_mean`. To preserve `q_mean`, we conceptually adjust the weights of the
valid (non-truncated) quadrature points by `ratio = q_mean / q̃_mean`. Since
condensate is zero whenever `q_hat = 0`, this is equivalent to scaling the 
integrated condensate by `ratio`.

# Arguments
- `thermo_params`: Thermodynamics parameters
- `SG_quad`: `SGSQuadrature` configuration
- `ρ`: Air density [kg/m³]
- `T_mean`: Grid-mean temperature [K]
- `q_mean`: Grid-mean total specific humidity [kg/kg]
- `T′T′`: Temperature variance [K²]
- `q′q′`: Moisture variance [(kg/kg)²]
- `corr_Tq`: Correlation coefficient corr(T', q')

# Returns
NamedTuple with SGS-averaged:
- `T`: Grid-mean temperature [K] (unchanged from saturation adjustment)
- `q_liq`: Liquid condensate [kg/kg]
- `q_ice`: Ice condensate [kg/kg]
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
    FT = typeof(T_mean)
    # Create GPU-safe functor (not a closure)
    evaluator = SaturationAdjustmentEvaluator(thermo_params, ρ)

    # Integrate over quadrature points
    result =
        integrate_over_sgs(evaluator, SG_quad, q_mean, T_mean, q′q′, T′T′, corr_Tq)

    # Weight adjustment for truncated distribution (correct for any q distribution
    # but will be no-op for lognormal q distribution):
    # When q_hat is clamped to 0, the integrated q̃_mean exceeds q_mean.
    # Adjusting weights of valid points by ratio = q_mean / q̃_mean preserves
    # q_mean. Since q_cond = 0 whenever q_hat = 0, this is equivalent to
    # scaling the integrated condensate by ratio.
    # Clamp ratio ≤ 1: only correct downward (lower-bound truncation).
    ratio = min(one(FT), q_mean / max(result.q_tot_quad, ϵ_numerics(FT)))
    q_liq = result.q_liq * ratio
    q_ice = result.q_ice * ratio

    # Return the equilibrium-adjusted T_mean unchanged. Recomputing T from
    # (e_int, q_tot, q_liq_sgs, q_ice_sgs) would yield a temperature
    # inconsistent with saturation equilibrium (causing problems in Jacobian approximations).
    return (; T = T_mean, q_liq, q_ice)
end
