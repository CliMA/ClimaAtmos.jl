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

Used as a component by `Microphysics0MEvaluator`; `compute_sgs_saturation_adjustment`
evaluates the analytic equilibrium expectation instead.

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
        lin = 0, curv = 0,
    )

Compute SGS-averaged saturation adjustment over the joint PDF of `(T, q_tot)`.

The mean saturation excess `μ_S = q_mean − q_sat(T_mean, ρ)` and variance
`σ_S² = E[(S − μ_S)²]` are accumulated over the quadrature points via
`_sgs_saturation_moments`, and the expected equilibrium condensate is evaluated
from the analytic truncated-Gaussian integral
`q_cond = μ_S Φ(μ_S / σ_S) + σ_S φ(μ_S / σ_S)` (`truncated_gaussian_condensate`),
or, with nonzero displacement coefficients `lin`, `curv`
(`displacement_curvature_coefficients`), from the curvature-aware expectation
`displacement_condensate(μ_S, σ_S, lin, curv)`. The continuous expectation
has none of the discrete 3-point Gauss–Hermite support cutoff at
`μ_S < −√3 σ_S` and keeps the equilibrium condensate smooth; with
`lin = curv = 0` it is the same truncated-Gaussian expectation the
cloud-fraction closure uses.
Called from `set_precomputed_quantities!` as a second pass that overwrites the
grid-mean saturation-adjustment condensate `ᶜq_liq`, `ᶜq_ice` whenever an
equilibrium moisture model is run with an SGS quadrature configured, so that
condensate and cloud fraction follow the same distribution.

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
  - `lin`, `curv`: Linear and curvature coefficients of the saturation excess
    under the vertical displacement [kg/kg]; zero for the truncated-Gaussian
    expectation.

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
    lin = zero(ρ),
    curv = zero(ρ),
)
    FT = typeof(T_mean)
    λ_mean = TD.liquid_fraction_ramp(thermo_params, T_mean)
    (; mu_S, sigma_S) = _sgs_saturation_moments(
        thermo_params, ρ, T_mean, q_mean, SG_quad, T′T′, q′q′, corr_Tq,
    )
    # The Gaussian tail of `q_tot` extends below zero, so the analytic
    # expectation can exceed the total water the cell holds; the condensate is
    # bounded by `q_mean` to keep the grid-mean vapor non-negative.
    q_cond = min(
        displacement_condensate(mu_S, sigma_S, lin, curv),
        max(q_mean, zero(FT)),
    )
    q_liq = λ_mean * q_cond
    q_ice = (one(FT) - λ_mean) * q_cond
    return (; T = T_mean, q_liq, q_ice)
end
