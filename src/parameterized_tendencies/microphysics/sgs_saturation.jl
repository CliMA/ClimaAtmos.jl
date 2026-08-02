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
    SaturationAdjustmentEvaluator{TPS, T1}

GPU-safe functor diagnosing the saturation-adjusted state at an SGS quadrature
point, for use with [`integrate_over_sgs`](@ref).

Used both by [`compute_sgs_saturation_adjustment`](@ref) and, as a component, by
[`Microphysics0MEvaluator`](@ref).

# Fields

  - `thermo_params`: Thermodynamics parameters.
  - `ρ`: Air density [kg/m³], the grid-cell value, held fixed across quadrature
    points.
  - `λ_mean`: Liquid fraction [-], evaluated once at the grid mean and held fixed
    across quadrature points.
"""
struct SaturationAdjustmentEvaluator{TPS, T1}
    thermo_params::TPS
    ρ::T1
    λ_mean::T1
end

"""
    (eval::SaturationAdjustmentEvaluator)(T_hat, q_hat)

Diagnose the saturation-adjusted state at a single quadrature point.

The condensate is the saturation excess `max(0, q_hat - q_sat(T_hat, ρ))`, split
into liquid and ice by the fixed `λ_mean`. The saturation humidity is the
condensate-free value, consistent with diagnosing equilibrium condensate from
scratch at each point.

# Arguments

  - `T_hat`: Temperature at the quadrature point [K].
  - `q_hat`: Total specific humidity at the quadrature point [kg/kg].

# Returns

NamedTuple with:

  - `T`: Temperature, passed through unchanged [K].
  - `q_liq`: Liquid condensate [kg/kg].
  - `q_ice`: Ice condensate [kg/kg].
  - `q_tot_quad`: `q_hat` itself, so the caller can integrate the mean of the
    possibly truncated `q_tot` distribution [kg/kg].
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

    # Partition condensate using a grid-mean liquid fraction held fixed across
    # quadrature points. The 0M / saturation-adjustment scheme has no
    # prognostic phase memory, so `λ_mean` is supplied by the caller using a
    # temperature-based ramp evaluated at the grid-mean temperature.
    q_liq = eval.λ_mean * q_cond
    q_ice = (FT(1) - eval.λ_mean) * q_cond

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

Compute SGS-averaged saturation adjustment by integrating over the joint PDF of
`(T, q_tot)`.

Condensate is diagnosed from the saturation excess at each quadrature point (see
[`SaturationAdjustmentEvaluator`](@ref)), giving a subgrid-aware cloud condensate.
Called from `set_precomputed_quantities!` as a second pass that overwrites the
grid-mean saturation-adjustment condensate `ᶜq_liq`, `ᶜq_ice` whenever an
equilibrium moisture model is run with an SGS quadrature configured, so that
condensate and cloud fraction follow the same distribution.

Quadrature points whose sampled `q_tot` falls in the negative tail are clamped to
zero, which makes the integrated mean `q̃_mean` exceed `q_mean`. Reweighting the
surviving points by `ratio = q_mean / q̃_mean` restores `q_mean`, and because
condensate vanishes wherever `q_hat = 0`, that reweighting is equivalent to scaling
the integrated condensate by `ratio`. The ratio is clamped at 1 so only downward
correction is applied, and it is a no-op for a lognormal `q` distribution.

The returned temperature is `T_mean` unchanged. Recomputing `T` from
`(e_int, q_tot, q_liq, q_ice)` with the SGS condensate would give a temperature
inconsistent with saturation equilibrium and degrade the Jacobian approximations.

# Arguments

  - `thermo_params`: Thermodynamics parameters.
  - `SG_quad`: `SGSQuadrature` configuration.
  - `ρ`: Air density [kg/m³].
  - `T_mean`: Grid-mean temperature [K].
  - `q_mean`: Grid-mean total specific humidity [kg/kg].
  - `T′T′`: Temperature variance [K²].
  - `q′q′`: Total-water variance [(kg/kg)²].
  - `corr_Tq`: Correlation coefficient corr(T′, q′) [-].

# Returns

NamedTuple with `T` [K] and the SGS-averaged condensate `q_liq` and `q_ice`
[kg/kg].
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
    q_min = TD.Parameters.q_min(thermo_params)
    # Grid-mean liquid fraction, held fixed across all quadrature points.
    λ_mean = TD.liquid_fraction_ramp(thermo_params, T_mean)
    # Create GPU-safe functor (not a closure)
    evaluator = SaturationAdjustmentEvaluator(thermo_params, ρ, λ_mean)

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
    ratio = min(one(FT), q_mean / max(result.q_tot_quad, q_min))
    q_liq = result.q_liq * ratio
    q_ice = result.q_ice * ratio

    # Return the equilibrium-adjusted T_mean unchanged. Recomputing T from
    # (e_int, q_tot, q_liq_sgs, q_ice_sgs) would yield a temperature
    # inconsistent with saturation equilibrium (causing problems in Jacobian approximations).
    return (; T = T_mean, q_liq, q_ice)
end
