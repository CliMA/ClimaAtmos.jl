import Thermodynamics as TD
import CloudMicrophysics.Parameters as CMP
import CloudMicrophysics.BulkMicrophysicsTendencies as BMT
import CloudMicrophysics.AerosolModel as CMAM
import CloudMicrophysics.AerosolActivation as CMAA

# Import SGS quadrature utilities
using ..ClimaAtmos: integrate_over_sgs

###
### 0 Moment Microphysics
###

"""
    e_tot_0M_precipitation_sources_helper(thp, T, q_liq, q_ice, Φ)

Compute the specific energy carried away by precipitation in the 0-moment scheme.

The precipitating condensate carries internal energy (weighted by liquid fraction)
plus potential energy. This helper returns the energy per unit mass of precipitate.

# Arguments

  - `thp`: Thermodynamics parameters.
  - `T`: Air temperature [K].
  - `q_liq`: Cloud liquid specific humidity [kg/kg].
  - `q_ice`: Cloud ice specific humidity [kg/kg].
  - `Φ`: Geopotential [J/kg].

# Returns

Specific energy of the precipitating condensate [J/kg]:

```math
\\lambda I_l + (1 - \\lambda) I_i + \\Phi
```

where `λ` is the liquid fraction and `I_l`, `I_i` are liquid/ice internal energies.
"""
@inline function e_tot_0M_precipitation_sources_helper(thp, T, q_liq, q_ice, Φ)
    λ = TD.liquid_fraction(thp, T, q_liq, q_ice)
    Iₗ = TD.internal_energy_liquid(thp, T)
    Iᵢ = TD.internal_energy_ice(thp, T)

    return λ * Iₗ + (1 - λ) * Iᵢ + Φ
end

"""
    Microphysics0MEvaluator{CMP, SAE, FT}

GPU-safe functor evaluating 0-moment microphysics tendencies at SGS quadrature
points, for use with [`integrate_over_sgs`](@ref).

# Fields

  - `cm_params`: 0M microphysics parameters.
  - `sat_eval`: `SaturationAdjustmentEvaluator` used to diagnose the local
    condensate.
  - `Φ`: Geopotential [J/kg], constant within a grid cell.

# Constructor

    Microphysics0MEvaluator(cm_params, thermo_params, ρ, T_mean, Φ)

Build the evaluator from the grid-mean state. The liquid fraction passed to
`sat_eval` is the temperature ramp evaluated once at the grid-mean `T_mean` and
held fixed across quadrature points, since the 0M scheme has no prognostic phase
memory.
"""
struct Microphysics0MEvaluator{CMP, SAE, FT}
    cm_params::CMP
    sat_eval::SAE
    Φ::FT
end
function Microphysics0MEvaluator(cm_params, thermo_params, ρ, T_mean, Φ)
    # Grid-mean liquid fraction, held fixed across quadrature points.
    # The 0M scheme has no prognostic phase memory, so we use a
    # temperature-based ramp at the grid mean.
    λ_mean = TD.liquid_fraction_ramp(thermo_params, T_mean)
    sat_eval = SaturationAdjustmentEvaluator(thermo_params, ρ, λ_mean)
    return Microphysics0MEvaluator(cm_params, sat_eval, Φ)
end

"""
    (eval::Microphysics0MEvaluator)(T_hat, q_hat)

Evaluate the 0-moment tendency at one quadrature point `(T_hat, q_hat)` [K, kg/kg].

Diagnoses the local condensate by saturation adjustment, then calls
`BMT.bulk_microphysics_tendencies(BMT.Microphysics0Moment(), ...)`.

# Returns

NamedTuple with `dq_tot_dt` [kg/kg/s] and the energy-flux product
`dq_e = dq_tot_dt · e_tot_hlpr` [W/kg]. The product is formed per point so
that its SGS average is the true energy sink `E[dq·e]` (averaging `dq` and
`e` separately and multiplying the means would drop their covariance).
"""
@inline function (eval::Microphysics0MEvaluator)(T_hat, q_hat)
    # Diagnose condensate via saturation adjustment
    sa = eval.sat_eval(T_hat, q_hat)

    # Compute saturation specific humidity for supersaturation threshold
    q_vap_sat = TD.q_vap_saturation(
        eval.sat_eval.thermo_params, T_hat, eval.sat_eval.ρ,
    )

    # Compute 0M dq_tot_dt at this quadrature point
    dq_tot_dt = BMT.bulk_microphysics_tendencies(
        BMT.Microphysics0Moment(), eval.cm_params, eval.sat_eval.thermo_params,
        T_hat, sa.q_liq, sa.q_ice, q_vap_sat,
    )
    # Energy helper at this quadrature point using the locally-diagnosed
    # condensate; returned as the product with dq_tot_dt (see docstring).
    e_tot_hlpr = e_tot_0M_precipitation_sources_helper(
        eval.sat_eval.thermo_params, T_hat, sa.q_liq, sa.q_ice, eval.Φ,
    )
    return (; dq_tot_dt, dq_e = dq_tot_dt * e_tot_hlpr)
end

"""
    microphysics_tendencies_0m(SG_quad, cmp, thp, ρ, T, q_tot_nonneg, T′T′, q′q′, corr_Tq, Φ, dt)
    microphysics_tendencies_0m(cmp, thp, ρ, T, q_tot_nonneg, q_liq, q_ice, Φ, dt)

Compute 0-moment microphysics tendencies.

The quadrature form integrates over the joint SGS PDF of `(T, q_tot)`: at each
quadrature point, condensate is diagnosed from saturation excess (see
`Microphysics0MEvaluator`), then the 0M precipitation-removal tendency is
computed and SGS-averaged.

The form without `SG_quad` is used in EDMF updrafts, or to compute the grid-mean
tendency without accounting for fluctuations; it evaluates the 0M tendencies from
the provided point values of temperature and specific humidities.

In both forms, the total water sink is limited by the available `q_tot_nonneg`
via `apply_0m_tendency_limit`.

# Arguments

  - `SG_quad`: `SGSQuadrature` configuration.
  - `cmp`, `thp`: Cloud microphysics and thermodynamics parameters.
  - `ρ`, `T`: Air density [kg/m³] and temperature [K].
  - `q_tot_nonneg`, `q_liq`, `q_ice`: Total water, liquid, and ice specific
    humidities [kg/kg].
  - `T′T′`: Variance of temperature ``\\langle T'^2 \\rangle`` [K²].
  - `q′q′`: Variance of `q_tot` ``\\langle q'^2 \\rangle`` [(kg/kg)²].
  - `corr_Tq`: Correlation coefficient corr(T′, q′) [-].
  - `Φ`: Geopotential energy [J/kg].
  - `dt`: Model timestep [s].

# Returns

NamedTuple with `dq_tot_dt` [kg/kg/s] and `e_tot_hlpr` [J/kg]. In the
quadrature form, `e_tot_hlpr` is the flux-weighted helper
`E[dq·e] / E[dq]`, so downstream products `dq_tot_dt · e_tot_hlpr`
reconstruct the true SGS-averaged energy sink `E[dq·e]` — including after
the limiter, which scales mass and energy by the same factor. The
flux-weighted helper is a `dq`-weighted average of the per-point helper
values (all `dq` share one sign), so it lies within their range. It is
zero where nothing precipitates, which carries no energy because
`dq_tot_dt` is zero there too.
"""
@inline function microphysics_tendencies_0m(
    SG_quad, cmp, thp, ρ, T, q_tot_nonneg, T′T′, q′q′, corr_Tq, Φ, dt,
)
    FT = typeof(ρ)
    # Create GPU-safe functor (Φ is constant within a grid cell)
    # The evaluator does saturation adjustment, computes saturation vapor pressure
    # and computes the total water sink and energy-flux product from 0M microphysics
    evaluator = Microphysics0MEvaluator(cmp, thp, ρ, T, Φ)
    # Integrate over quadrature points; dq_tot_dt and the product dq·e are
    # averaged over the SGS distribution.
    (; dq_tot_dt, dq_e) = integrate_over_sgs(
        evaluator, SG_quad, q_tot_nonneg, T, q′q′, T′T′, corr_Tq,
    )
    # Flux-weighted energy helper: E[dq·e] / E[dq]. The ratio is stable for
    # any strictly negative mean sink because numerator and denominator share
    # the dq scale. The 0M sink is nonpositive at every quadrature point and
    # the weights are positive, so `E[dq] = 0` means no point precipitates and
    # the energy change is zero too.
    e_tot_hlpr = ifelse(dq_tot_dt < zero(FT), dq_e / dq_tot_dt, zero(FT))
    # Apply limiter
    dq_tot_dt = apply_0m_tendency_limit(dq_tot_dt, q_tot_nonneg, dt)

    return (; dq_tot_dt, e_tot_hlpr)
end
@inline function microphysics_tendencies_0m(
    cmp, thp, ρ, T, q_tot_nonneg, q_liq, q_ice, Φ, dt,
)
    # Computes saturation vapor pressure, total water sink and energy helper
    # based on provided mean temperature, total water, liquid and ice specific humidities.
    # Does not take into account SGS fluctuations.
    q_vap_sat = TD.q_vap_saturation(thp, T, ρ)
    dq_tot_dt = BMT.bulk_microphysics_tendencies(
        BMT.Microphysics0Moment(), cmp, thp, T, q_liq, q_ice, q_vap_sat,
    )
    e_tot_hlpr = e_tot_0M_precipitation_sources_helper(thp, T, q_liq, q_ice, Φ)

    # Apply limiter
    dq_tot_dt = apply_0m_tendency_limit(dq_tot_dt, q_tot_nonneg, dt)

    return (; dq_tot_dt, e_tot_hlpr)
end

###
### 1 Moment Microphysics
###

"""
    Microphysics1MEvaluator{S, MP, TPS, FT, Args}

GPU-safe functor evaluating 1-moment microphysics tendencies at SGS quadrature
points, for use with [`integrate_over_sgs`](@ref).

The local condensate at each point follows the truncated-Gaussian
Lagrange-multiplier closure described in `microphysics_tendencies_1m`.
Precipitation (`q_rai`, `q_sno`), the liquid fraction `λ`, and the closure
quantities (`λ_lagrange`, `mu_S`, `α`) are grid-cell constants held fixed across
quadrature points.

# Fields

  - `scheme`: CloudMicrophysics scheme tag (e.g. `BMT.Microphysics1Moment()`).
  - `mp`, `tps`: Microphysics and thermodynamics parameters.
  - `ρ`: Air density [kg/m³].
  - `w`: Physical vertical velocity [m/s], used for velocity-dependent
    rain autoconversion.
  - `q_rai`, `q_sno`: Rain and snow specific humidity [kg/kg], clamped
    non-negative by the caller.
  - `λ`: Thermodynamic liquid fraction [-].
  - `ξ_liq`, `ξ_ice`: Uniform fractions of cloud liquid and cloud ice over the
    quadrature [-].
  - `q_lcl`, `q_icl`: Subdomain-mean cloud liquid and cloud ice [kg/kg].
  - `λ_lagrange`: Lagrange multiplier enforcing
    `E[max(0, λ_lagrange + α·S′)] = q_c` under the quadrature
    measure (fitted in `_compute_sgs_moments`) [kg/kg].
  - `mu_S`: Linearized SGS mean saturation excess `q_tot − q_sat(T, ρ)` [kg/kg].
  - `α`: Variance fidelity parameter [-].
  - `dt`: Timestep used for the time-averaged process rates [s].
  - `nsubs`: Number of substeps in the tendency averaging.
  - `args`: Extra trailing arguments forwarded to the CloudMicrophysics call.
  - `T_ramp_lo`, `T_ramp_hi`: Temperature ramp of `ξ_ice` at the nodes
    (`sgs_ice_uniform_fraction_ramped`); disabled when `T_ramp_hi ≤ T_ramp_lo`.
  - `β_precip`, `cf_precip`: Precipitation-fraction placement of the cell-mean
    rain and snow onto the moist half of the PDF (`SGSMoistHalfFlag`) and that
    half's weight; `β_precip = 0` disables it.
  - `β_snow`: snow share of the placement (equal to `β_precip` unless set).
  - `S_star`, `ε_S`: Threshold and width of the shaft weight on the centred
    saturation excess (`sgs_precip_shaft_weight`); `(0, 0)` is the moist-half
    mode, the overlap fraction sets them through `sgs_precip_shaft_threshold`.
"""
struct Microphysics1MEvaluator{S, MP, TPS, FT, Args <: Tuple}
    scheme::S
    mp::MP
    tps::TPS
    ρ::FT
    w::FT              # physical vertical velocity [m/s]
    # Precipitation (held fixed across quadrature points)
    q_rai::FT
    q_sno::FT
    λ::FT              # liquid fraction (from thermodynamics, held fixed)
    # Uniform fractions of each species over the quadrature and the subdomain
    # means they draw on
    ξ_liq::FT
    ξ_ice::FT
    q_lcl::FT
    q_icl::FT
    # Truncated-Gaussian Lagrange multiplier, μ_S, and liquid fraction
    λ_lagrange::FT # Lagrange multiplier for centred S′ (discrete fit)
    mu_S::FT       # linearized SGS mean μ_S = q_tot_mean − q_sat(T_mean, ρ)
    α::FT          # variance fidelity parameter (from sgs_variance_fidelity)
    # Numerical parameters
    dt::FT
    nsubs::Int
    args::Args
    # Temperature ramp of ξ_ice (`sgs_ice_uniform_fraction_ramped`): ξ_ice at
    # T ≥ T_ramp_hi, 1 at T ≤ T_ramp_lo; disabled when T_ramp_hi ≤ T_ramp_lo.
    T_ramp_lo::FT
    T_ramp_hi::FT
    # Precipitation-fraction placement (`sgs_placement_factor` applied to
    # rain and snow): the fraction `β_precip` of the cell-mean rain and snow is
    # confined to the moist half of the PDF (nodes with centred saturation
    # excess S′ ≥ 0, quadrature weight `cf_precip`, precomputed by the caller
    # with `SGSMoistHalfFlag`), so below-cloud evaporation and sublimation are
    # evaluated with the humidity of the air the precipitation falls through.
    # `β_precip = 0` disables the placement.
    β_precip::FT
    cf_precip::FT
    # Snow share of the precipitation-fraction placement (`β_snow`); the
    # 22-argument constructor sets it equal to `β_precip` (rain and snow placed
    # alike), a separate value lets rain and snow be placed independently.
    β_snow::FT
    # Shaft weight on the centred saturation excess (`sgs_precip_shaft_weight`):
    # nodes with S′ ≥ S_star carry the placed precipitation, smoothed over the
    # width ε_S. (0, 0) is the hard moist-half flag (S′ ≥ 0); the overlap
    # precipitation fraction sets S_star so that the shaft covers the moistest
    # `a_p` of the PDF (`sgs_precip_shaft_threshold`).
    S_star::FT
    ε_S::FT
    # Sub-population placement (`sgs_precip_shaft_random`): the shaft holds all
    # cloudy nodes (smooth cloudy weight of width ε_w, the one `CF_d` was
    # accumulated with) and a random share `p_clear` of the clear nodes, at the
    # in-shaft concentration `conc = 1/A`, `A = CF_d + (1 − CF_d) p_clear`; the
    # node tendency is the in-shaft/out-of-shaft mixture. `p_clear < 0`
    # disables this mode (the rank/moist-half placement above applies).
    p_clear::FT
    conc::FT
    ε_w::FT
end
# Ramp-free construction (ramp and placement disabled), the pre-existing
# positional signature.
Microphysics1MEvaluator(
    scheme, mp, tps, ρ, w, q_rai, q_sno, λ, ξ_liq, ξ_ice, q_lcl, q_icl,
    λ_lagrange, mu_S, α, dt, nsubs, args,
) = Microphysics1MEvaluator(
    scheme, mp, tps, ρ, w, q_rai, q_sno, λ, ξ_liq, ξ_ice, q_lcl, q_icl,
    λ_lagrange, mu_S, α, dt, nsubs, args, zero(ρ), zero(ρ),
    zero(ρ), one(ρ), zero(ρ), zero(ρ), zero(ρ), -one(ρ), one(ρ), zero(ρ),
)
# Ramp only (placement disabled).
Microphysics1MEvaluator(
    scheme, mp, tps, ρ, w, q_rai, q_sno, λ, ξ_liq, ξ_ice, q_lcl, q_icl,
    λ_lagrange, mu_S, α, dt, nsubs, args, T_ramp_lo, T_ramp_hi,
) = Microphysics1MEvaluator(
    scheme, mp, tps, ρ, w, q_rai, q_sno, λ, ξ_liq, ξ_ice, q_lcl, q_icl,
    λ_lagrange, mu_S, α, dt, nsubs, args, T_ramp_lo, T_ramp_hi,
    zero(ρ), one(ρ), zero(ρ), zero(ρ), zero(ρ), -one(ρ), one(ρ), zero(ρ),
)
# Rain and snow placed alike (β_snow = β_precip).
Microphysics1MEvaluator(
    scheme, mp, tps, ρ, w, q_rai, q_sno, λ, ξ_liq, ξ_ice, q_lcl, q_icl,
    λ_lagrange, mu_S, α, dt, nsubs, args, T_ramp_lo, T_ramp_hi, β_precip, cf_precip,
) = Microphysics1MEvaluator(
    scheme, mp, tps, ρ, w, q_rai, q_sno, λ, ξ_liq, ξ_ice, q_lcl, q_icl,
    λ_lagrange, mu_S, α, dt, nsubs, args, T_ramp_lo, T_ramp_hi,
    β_precip, cf_precip, β_precip, zero(ρ), zero(ρ), -one(ρ), one(ρ), zero(ρ),
)
# Moist-half placement with a separate snow share (S_star = ε_S = 0).
Microphysics1MEvaluator(
    scheme, mp, tps, ρ, w, q_rai, q_sno, λ, ξ_liq, ξ_ice, q_lcl, q_icl,
    λ_lagrange, mu_S, α, dt, nsubs, args, T_ramp_lo, T_ramp_hi, β_precip, cf_precip,
    β_snow,
) = Microphysics1MEvaluator(
    scheme, mp, tps, ρ, w, q_rai, q_sno, λ, ξ_liq, ξ_ice, q_lcl, q_icl,
    λ_lagrange, mu_S, α, dt, nsubs, args, T_ramp_lo, T_ramp_hi,
    β_precip, cf_precip, β_snow, zero(ρ), zero(ρ), -one(ρ), one(ρ), zero(ρ),
)
# Rank placement with an explicit threshold and width (sub-population off).
Microphysics1MEvaluator(
    scheme, mp, tps, ρ, w, q_rai, q_sno, λ, ξ_liq, ξ_ice, q_lcl, q_icl,
    λ_lagrange, mu_S, α, dt, nsubs, args, T_ramp_lo, T_ramp_hi, β_precip, cf_precip,
    β_snow, S_star, ε_S,
) = Microphysics1MEvaluator(
    scheme, mp, tps, ρ, w, q_rai, q_sno, λ, ξ_liq, ξ_ice, q_lcl, q_icl,
    λ_lagrange, mu_S, α, dt, nsubs, args, T_ramp_lo, T_ramp_hi,
    β_precip, cf_precip, β_snow, S_star, ε_S, -one(ρ), one(ρ), zero(ρ),
)

"""
    sgs_precip_subpopulation(a_p, CF_d)

Per-cell constants of the sub-population placement: the random share
`p_clear = (a_p − CF_d) / (1 − CF_d)` of the clear nodes inside the shaft and the
in-shaft concentration factor `conc = 1 / A` with `A = CF_d + (1 − CF_d) p_clear`
the shaft's weight under the discrete measure. Every cloudy node is in the shaft
(maximum overlap: the precipitation falls through the cloud below it) and a clear
node is in it with probability `p_clear`, independent of its humidity, so
evaporation and sublimation are evaluated at the clear-sky humidity of the cell
rather than at its moist tail; the in-shaft concentration `q / A` carries the
sub-linear dependence of the rates on the precipitation content. The quadrature
mean of the node precipitation `P·q/A` is `q` exactly whenever `CF_d` was
accumulated with the same cloudy weight. `a_p` is floored at
`sgs_precip_fraction_min`; `a_p ≥ 1` or `CF_d ≥ 1` give `(1, 1)`, the uniform
limit.
"""
@inline function sgs_precip_subpopulation(a_p, CF_d)
    FT = typeof(CF_d)
    a = clamp(FT(a_p), sgs_precip_fraction_min(FT), one(FT))
    c = clamp(CF_d, zero(FT), one(FT))
    p_clear = clamp((a - c) / max(one(FT) - c, eps(FT)), zero(FT), one(FT))
    A = c + (one(FT) - c) * p_clear
    return (p_clear, one(FT) / max(A, sgs_precip_weight_min(FT)))
end

"""
    sgs_ice_uniform_fraction_ramped(ξ_ice, T, T_lo, T_hi)

Uniform-ice fraction at a quadrature node of temperature `T` [K]: `ξ_ice` at
`T ≥ T_hi`, `1` (ice uniform over the nodes) at `T ≤ T_lo`, linear in between.
`T_hi ≤ T_lo` disables the ramp and returns `ξ_ice` exactly. Physical reading:
in mixed-phase clouds the ice follows the condensate distribution (excess
reconstruction, so the thick nodes carry the ice that converts their liquid),
while cold cirrus ice is detrained and long-lived, hence uniform.
"""
@inline function sgs_ice_uniform_fraction_ramped(ξ_ice, T, T_lo, T_hi)
    FT = typeof(ξ_ice)
    width = max(T_hi - T_lo, eps(FT))
    r = ifelse(
        T_hi > T_lo,
        clamp((T_hi - T) / width, zero(FT), one(FT)),
        zero(FT),
    )
    return ξ_ice + (one(FT) - ξ_ice) * r
end

"""
    sgs_precip_shaft_weight(S′, S_star, ε_S)

Weight in `[0, 1]` with which a quadrature node of centred saturation excess
`S′` belongs to the precipitation shaft: the nodes with `S′ ≥ S_star`, smoothed
over the width `ε_S`,

    s = sigmoid((S′ − S_star) / ε_S)      (ε_S > 0)
    s = 1[S′ ≥ S_star]                     (ε_S = 0, the hard flag)

`(S_star, ε_S) = (0, 0)` is the moist-half flag (nodes at least as close to
saturation as the cell mean). Its quadrature mean `cf_precip` is the weight of
the shaft under the discrete measure, which normalizes the placement
(`sgs_placement_factor`), so the cell-mean precipitation is conserved for any
threshold and width.
"""
@inline function sgs_precip_shaft_weight(S′, S_star, ε_S)
    FT = typeof(S′)
    return ifelse(
        ε_S > zero(FT),
        (one(FT) + tanh((S′ - S_star) / (2 * max(ε_S, eps(FT))))) / 2,
        ifelse(S′ >= S_star, one(FT), zero(FT)),
    )
end

"""
    sgs_precip_shaft_width_coeff(FT)

Width of the shaft weight (`sgs_precip_shaft_weight`) in units of the sampled
PDF width `σ_S` under the overlap precipitation fraction: `0.25` keeps the
transition narrow compared with the Gauss–Hermite node spacing (≈ 1.7 σ_S at
order 3) while smoothing the node assignment as `a_p` changes between steps.
"""
@inline sgs_precip_shaft_width_coeff(::Type{FT}) where {FT} = FT(0.25)

"""
    sgs_precip_fraction_min(FT)

Floor on the overlap precipitation fraction `a_p` seen by the quadrature
(`sgs_precip_shaft_threshold`): a shaft is never taken narrower than this
fraction of the cell. With `0.1` the moistest Gauss–Hermite node (weight 1/36
at order 3) is always resolved as the shaft, so the in-shaft precipitation
`q_precip / cf_precip` stays within the range the 1M closures are evaluated in.
"""
@inline sgs_precip_fraction_min(::Type{FT}) where {FT} = FT(0.1)

"""
    sgs_precip_weight_min(FT)

Smallest discrete shaft weight `cf_precip` the placement resolves; below it the
quadrature cannot represent the shaft and the precipitation is left uniform
(`sgs_placement_factor` with `cf = 0`).
"""
@inline sgs_precip_weight_min(::Type{FT}) where {FT} = FT(0.02)

"""
    sgs_precip_shaft_threshold(a_p, sigma_S)

Threshold `S_star` on the centred saturation excess such that the shaft covers
the moistest fraction `a_p` of the sampled PDF, taken Gaussian with standard
deviation `sigma_S` (the quadrature's own `Σᵢ wᵢ S′ᵢ²`):

    S_star = σ_S · Φ⁻¹(1 − a_p)

`a_p` is floored at `sgs_precip_fraction_min`; `a_p = 1` places the threshold
far below every node (uniform precipitation). The discrete weight of the nodes
above the threshold, not `a_p` itself, normalizes the placement, so the
Gaussian rank is only used to order the nodes.
"""
@inline function sgs_precip_shaft_threshold(a_p, sigma_S)
    FT = typeof(sigma_S)
    a = clamp(FT(a_p), sgs_precip_fraction_min(FT), one(FT))
    return sigma_S * normal_cdf_inv(one(FT) - a)
end

"""
    SGSPrecipShaftFlag(tps, ρ, mu_S, S_star, ε_S)

Point-wise functor for the SGS quadrature returning the shaft weight
(`sgs_precip_shaft_weight`) of a node from its centred saturation excess
`S′ = q_tot_hat − q_sat(T_hat, ρ) − mu_S`. Its quadrature mean `cf_precip` is
the weight of the part of the cell the precipitation is taken to fall through
under the precipitation-fraction placement (`Microphysics1MEvaluator`).
`SGSMoistHalfFlag(tps, ρ, mu_S)` is the hard moist-half instance
`(S_star, ε_S) = (0, 0)`.
"""
struct SGSPrecipShaftFlag{TPS, FT}
    tps::TPS
    ρ::FT
    mu_S::FT
    S_star::FT
    ε_S::FT
end
SGSMoistHalfFlag(tps, ρ, mu_S) = SGSPrecipShaftFlag(tps, ρ, mu_S, zero(ρ), zero(ρ))
@inline function (f::SGSPrecipShaftFlag)(T_hat, q_tot_hat)
    FT = typeof(f.ρ)
    S′ = max(zero(FT), q_tot_hat) - TD.q_vap_saturation(f.tps, T_hat, f.ρ) - f.mu_S
    return sgs_precip_shaft_weight(S′, f.S_star, f.ε_S)
end

"""
    sgs_placement_factor(β, flag, cf)

Factor `φ` on a cell-mean quantity at a quadrature node when a fraction `β` of it
is confined to the flagged nodes (`flag = 1`, total quadrature weight `cf`) and
the rest stays uniform,

    φ = (1 − β) + β · flag / cf,

so that the quadrature mean of `φ` is 1 and the cell mean is conserved. `β = 0`
returns exactly 1; `cf = 0` (no flagged node) falls back to uniform. Used by the
precipitation-fraction placement of rain and snow (`Microphysics1MEvaluator`).
"""
@inline function sgs_placement_factor(β, flag, cf)
    FT = typeof(β)
    φ_in = ifelse(cf > zero(FT), flag / max(cf, eps(FT)), one(FT))
    return (one(FT) - β) + β * φ_in
end

"""
    sgs_local_condensate(λ, shifted_excess, ξ_liq, ξ_ice, q_lcl, q_icl)

Local cloud liquid and cloud ice `(q_lcl_hat, q_icl_hat)` [kg/kg] at one
quadrature node from the reconstructed `shifted_excess = max(0, λ_lagrange + α S′)`,
the liquid fraction `λ`, the subdomain means `q_lcl`, `q_icl` and the uniform
fractions `ξ_liq`, `ξ_ice`:

    q_lcl_hat = (1 − ξ_liq) · λ · shifted_excess       + ξ_liq · q_lcl
    q_icl_hat = (1 − ξ_ice) · (1 − λ) · shifted_excess + ξ_ice · q_icl

`ξ = 0` is the excess reconstruction (the species sits only where the node is
saturated, in proportion to its excess); `ξ = 1` is the uniform distribution (the
species is the subdomain mean at every node). Either end, and any blend, conserves
the quadrature mean of the species in cells with condensate, since the excess
share has mean `λ q_c = q_lcl` (resp. `(1 − λ) q_c = q_icl`) by construction of
`λ_lagrange`.
"""
@inline function sgs_local_condensate(λ, shifted_excess, ξ_liq, ξ_ice, q_lcl, q_icl)
    q_lcl_hat = (1 - ξ_liq) * (λ * shifted_excess) + ξ_liq * q_lcl
    q_icl_hat = (1 - ξ_ice) * ((1 - λ) * shifted_excess) + ξ_ice * q_icl
    return (q_lcl_hat, q_icl_hat)
end

# `@noinline` here is the SGS quadrature function barrier. The functor body
# below (saturation, shape-function partition, plus the heavy
# `BMT.average_bulk_microphysics_tendencies` call with its `nsubs`
# substep loop and 4×4 linearized operator) gets invoked N² times from
# `sum_over_quadrature_points`. Without the barrier those N² copies inline
# into one giant GPU broadcast kernel, pushing register pressure past the
# 255-reg hard cap and pinning occupancy at 12.5%. Marking the functor
# itself (vs a trivial forwarding wrapper) is the strongest signal we can
# give LLVM/NVPTX not to re-inline this — the body is multi-statement and
# meaningfully sized, so the late inliner won't undo it.
"""
    (eval::Microphysics1MEvaluator)(T_hat, q_tot_hat)

Evaluate the 1-moment tendencies at one quadrature point `(T_hat, q_tot_hat)`
[K, kg/kg].

The local cloud condensate is obtained from the centred saturation excess
`S′_hat = (q_tot_hat − q_sat(T_hat, ρ)) − mu_S`:

    shifted_excess = max(0, λ_lagrange + α · S′_hat)
    q_lcl_hat      = (1 − ξ_liq) · λ · shifted_excess       + ξ_liq · q_lcl
    q_icl_hat      = (1 − ξ_ice) · (1 − λ) · shifted_excess + ξ_ice · q_icl

The Lagrange multiplier `λ_lagrange` is fitted (in `_compute_sgs_moments`) so
that `E[shifted_excess] = q_c`, where `q_c = q_lcl + q_icl` is the grid-mean
*cloud* condensate, excluding precipitation. The reconstruction therefore
partitions `shifted_excess` into local cloud liquid and ice by the liquid
fraction; the uniform fractions `ξ_liq`, `ξ_ice` replace part of a
species' share by its subdomain mean at every node (see
`sgs_local_condensate`). Precipitation is held constant across
quadrature points and is accounted for downstream, where CloudMicrophysics
subtracts it from `q_tot` to diagnose the local vapor.

`q_tot_hat` is clamped non-negative first. Subsaturated points contribute zero
condensate but still drive rain evaporation and snow sublimation against the
local vapor.

# Returns

NamedTuple from `BMT.bulk_microphysics_tendencies(BMT.LinearizedAverage(), ...)`
with `dq_lcl_dt`, `dq_icl_dt`, `dq_rai_dt`, `dq_sno_dt` [kg/kg/s].
"""
@noinline function (eval::Microphysics1MEvaluator)(T_hat, q_tot_hat)
    FT = typeof(eval.ρ)
    q_tot_hat = max(FT(0), q_tot_hat)

    # Local cloud condensate from the Lagrange-multiplier closure.
    # The mass conservation equation is E[max(0, λ + α·S′)] = q_c, so the
    # local shifted excess at each quadrature point is λ + α·S′_hat where
    # S′_hat = (q_tot_hat − q_sat_hat) − μ_S is the centred saturation excess.
    # Precipitation in q_tot needs no special handling here: its mean level
    # cancels in the centred S′ (the level is re-anchored by λ_lagrange, fitted
    # to cloud-only q_c), and CloudMicrophysics subtracts q_rai/q_sno from
    # q_tot_hat when it diagnoses the local vapor. Subtracting them from the
    # cloud condensate as well would double-count them and break ⟨q_c^local⟩ = q_c.
    q_sat_hat = TD.q_vap_saturation(eval.tps, T_hat, eval.ρ)
    S′_hat = q_tot_hat - q_sat_hat - eval.mu_S
    shifted_excess_signed = eval.λ_lagrange + eval.α * S′_hat
    shifted_excess = max(FT(0), shifted_excess_signed)
    ξ_ice = sgs_ice_uniform_fraction_ramped(
        eval.ξ_ice, T_hat, eval.T_ramp_lo, eval.T_ramp_hi,
    )
    q_lcl_hat, q_icl_hat = sgs_local_condensate(
        eval.λ, shifted_excess, eval.ξ_liq, ξ_ice, eval.q_lcl, eval.q_icl,
    )
    # Sub-population placement: the node is in the shaft with probability P
    # (1 if cloudy, `p_clear` if clear); its tendency is the mixture of the
    # in-shaft state (precipitation at `conc` times the cell mean) and the
    # precipitation-free state, both with the node total water shifted so that
    # the node vapour is the same in either state.
    if eval.p_clear >= zero(FT)
        s_c = discrete_cloudy_weight(shifted_excess_signed, eval.ε_w)
        P = s_c + (one(FT) - s_c) * eval.p_clear
        q_p = eval.q_rai + eval.q_sno
        q_tot_in = max(FT(0), q_tot_hat + (eval.conc - one(FT)) * q_p)
        q_tot_out = max(FT(0), q_tot_hat - q_p)
        if P >= one(FT) - eps(FT)
            return BMT.bulk_microphysics_tendencies(
                BMT.LinearizedAverage(),
                eval.scheme, eval.mp, eval.tps, eval.ρ, T_hat, eval.w,
                q_tot_in, q_lcl_hat, q_icl_hat,
                eval.q_rai * eval.conc, eval.q_sno * eval.conc,
                eval.dt, eval.nsubs, eval.args...,
            )
        elseif P <= eps(FT)
            return BMT.bulk_microphysics_tendencies(
                BMT.LinearizedAverage(),
                eval.scheme, eval.mp, eval.tps, eval.ρ, T_hat, eval.w,
                q_tot_out, q_lcl_hat, q_icl_hat, zero(FT), zero(FT),
                eval.dt, eval.nsubs, eval.args...,
            )
        else
            t_in = BMT.bulk_microphysics_tendencies(
                BMT.LinearizedAverage(),
                eval.scheme, eval.mp, eval.tps, eval.ρ, T_hat, eval.w,
                q_tot_in, q_lcl_hat, q_icl_hat,
                eval.q_rai * eval.conc, eval.q_sno * eval.conc,
                eval.dt, eval.nsubs, eval.args...,
            )
            t_out = BMT.bulk_microphysics_tendencies(
                BMT.LinearizedAverage(),
                eval.scheme, eval.mp, eval.tps, eval.ρ, T_hat, eval.w,
                q_tot_out, q_lcl_hat, q_icl_hat, zero(FT), zero(FT),
                eval.dt, eval.nsubs, eval.args...,
            )
            return map((x, y) -> P * x + (one(FT) - P) * y, t_in, t_out)
        end
    end
    # Precipitation-fraction placement: the rain and snow this node sees are
    # the cell means scaled by φ_p (1 when disabled); the node total water is
    # shifted by the same amount so that the node vapour, and hence the
    # condensate reconstruction, is unchanged by the placement.
    q_rai_node, q_sno_node, q_tot_node =
        if (eval.β_precip > zero(FT)) | (eval.β_snow > zero(FT))
            flag_p = sgs_precip_shaft_weight(S′_hat, eval.S_star, eval.ε_S)
            φ_r = sgs_placement_factor(eval.β_precip, flag_p, eval.cf_precip)
            φ_s = sgs_placement_factor(eval.β_snow, flag_p, eval.cf_precip)
            shift = (φ_r - one(FT)) * eval.q_rai + (φ_s - one(FT)) * eval.q_sno
            (eval.q_rai * φ_r, eval.q_sno * φ_s, max(FT(0), q_tot_hat + shift))
        else
            (eval.q_rai, eval.q_sno, q_tot_hat)
        end

    return BMT.bulk_microphysics_tendencies(
        BMT.LinearizedAverage(),
        eval.scheme, eval.mp, eval.tps, eval.ρ, T_hat, eval.w,
        q_tot_node, q_lcl_hat, q_icl_hat, q_rai_node, q_sno_node,
        eval.dt, eval.nsubs, eval.args...,
    )
end

"""
    microphysics_tendencies_1m(
        ρ, q_tot_nonneg, q_lcl, q_icl, q_rai, q_sno, T, w, cmp, thp, dt, nsubs,
    )
    microphysics_tendencies_1m(
        scheme, sgs_quad, cmp, thp, ρ, T, w, q_tot_nonneg,
        q_lcl, q_icl, q_rai, q_sno, T′T′, q′q′, corr_Tq,
        λ_lagrange, α, ξ_liq, ξ_ice, dt, nsubs, λ = ..., mu_S = ...,
        ice_ramp_T_low = 0, ice_ramp_T_high = 0, precip_incloud_fraction = 0,
        snow_incloud_fraction = -1, precip_overlap_decay = -1, precip_frac = 1,
        sigma_S = 0, args...,
    )

Compute time-averaged 1-moment microphysics tendencies.

The 11-argument (no `sgs_quad`) form takes the condensate inputs as-is and is used
in EDMF updrafts, or wherever a grid-mean state is to be evaluated directly: a
single CloudMicrophysics call with no SGS averaging.

The quadrature form integrates over the SGS PDF using the truncated-Gaussian
Lagrange-multiplier closure; see `Microphysics1MEvaluator` for the
point-wise condensate diagnosis. Rain and snow are clamped non-negative before the
integration. Subsaturated quadrature points contribute below-cloud rain
evaporation and snow sublimation; saturated points drive autoconversion and
accretion.

# Arguments

  - `scheme`: CloudMicrophysics scheme tag (from `BulkMicrophysicsTendencies`).
  - `sgs_quad`: `SGSQuadrature` configuration.
  - `cmp`, `thp`: Microphysics and thermodynamics parameters.
  - `ρ`, `T`: Air density [kg/m³] and temperature [K].
  - `w`: Physical vertical velocity [m/s], used for velocity-dependent
    rain autoconversion.
  - `q_tot_nonneg`: Total water specific humidity, clamped non-negative [kg/kg].
  - `q_lcl`, `q_icl`: Cloud liquid and cloud ice specific humidity [kg/kg].
  - `q_rai`, `q_sno`: Rain and snow specific humidity [kg/kg].
  - `T′T′`: Temperature variance ``\\langle T'^2 \\rangle`` [K²].
  - `q′q′`: Total-water variance ``\\langle q'^2 \\rangle`` [(kg/kg)²].
  - `corr_Tq`: Correlation coefficient corr(T′, q′) [-], the per-cell
    `p.precomputed.ᶜcorr_Tq` set by `set_tq_correlation!`.
  - `λ_lagrange`: Lagrange multiplier from `ᶜsgs_moments`, precomputed to
    enforce `E[max(0, λ_lagrange + α·S′)] = q_c` exactly under the
    quadrature measure [kg/kg].
  - `α`: Variance fidelity parameter from `sgs_variance_fidelity` [-].
  - `ξ_liq`, `ξ_ice`: Uniform fractions of cloud liquid and cloud ice over the
    quadrature (`sgs_liquid_uniform_fraction`, `sgs_ice_uniform_fraction`, in
    `[0, 1]`); `0` is the excess reconstruction, see `sgs_local_condensate`.
  - `dt`: Timestep [s].
  - `nsubs`: Number of substeps for tendency averaging.
  - `λ`: Liquid fraction [-]; defaults to `TD.liquid_fraction` at the mean state.
  - `mu_S`: Linearized SGS mean saturation excess [kg/kg]; defaults to
    `q_tot_nonneg − q_sat(T, ρ)`. Both are quadrature invariants and may be
    precomputed by the caller to avoid recomputing them at every point.
  - `ice_ramp_T_low`, `ice_ramp_T_high`: Temperature ramp of `ξ_ice` at the nodes
    (`sgs_ice_uniform_fraction_ramped`); the defaults `0, 0` disable it [K].
  - `precip_incloud_fraction`: Fraction `β_p` of the cell-mean rain and snow confined
    to the moist half of the PDF (`SGSMoistHalfFlag`, `sgs_precip_incloud_fraction`);
    the default `0` disables the placement and its extra quadrature pass [-].
  - `snow_incloud_fraction`: Snow share of that placement (`sgs_snow_incloud_fraction`);
    a negative value (the default) means "same as `precip_incloud_fraction`" [-].
  - `precip_overlap_decay`: `sgs_precip_overlap_decay`; non-negative selects the
    overlap precipitation fraction, under which the placed precipitation is
    confined to the moistest `precip_frac` of the PDF
    (`sgs_precip_shaft_threshold`) instead of its moist half; negative (the
    default) keeps the moist-half mode [-].
  - `precip_frac`: Overlap precipitation fraction `a_p` of the cell
    (`ᶜprecip_frac`, `set_precip_fraction!`) [-].
  - `sigma_S`: SGS saturation-excess standard deviation of the sampled PDF
    (`ᶜsgs_moments.sigma_S`), which scales the shaft threshold and width [kg/kg].
  - `CF_d`: Discrete cloudy mass of the sampled PDF (`ᶜsgs_moments.CF_d`) [-].
  - `precip_shaft_random`: `sgs_precip_shaft_random`; positive (with the overlap
    mode on) selects the sub-population placement (`sgs_precip_subpopulation`):
    all cloudy nodes plus a random share of the clear nodes carry the shaft at
    the in-shaft concentration, instead of the moistest `precip_frac` [-].
  - `precip_frac_floor`: `sgs_precip_fraction_floor`; the shaft is never taken
    narrower than this fraction of the cell (a wider shaft than the overlap of
    the cover gives: fall-streak spreading and shear); defaults to
    `sgs_precip_fraction_min` [-].
  - `args...`: Extra trailing arguments forwarded to CloudMicrophysics.

# Returns

NamedTuple with `dq_lcl_dt`, `dq_icl_dt`, `dq_rai_dt`, `dq_sno_dt` [kg/kg/s],
positive when a source of the corresponding tracer.
"""
@inline function microphysics_tendencies_1m( #compute_1m_precipitation_tendencies!(
    ρ, q_tot_nonneg, q_lcl, q_icl, q_rai, q_sno, T, w, cmp, thp, dt, nsubs,
)
    local_tendency = BMT.bulk_microphysics_tendencies(
        BMT.LinearizedAverage(),
        BMT.Microphysics1Moment(), cmp, thp, ρ, T, w,
        q_tot_nonneg, q_lcl, q_icl, q_rai, q_sno, dt, nsubs,
    )
    return local_tendency
end
@inline function microphysics_tendencies_1m( #microphysics_tendencies_quadrature_1m
    scheme, sgs_quad, cmp, thp, ρ, T, w, q_tot_nonneg,
    q_lcl, q_icl, q_rai, q_sno, T′T′, q′q′, corr_Tq,
    λ_lagrange, α, ξ_liq, ξ_ice, dt, nsubs,
    # `λ` (liquid fraction) and `mu_S` (linearized SGS saturation-excess mean) are
    # invariant across the quadrature. They default to being computed here from the
    # mean state; a caller evaluating this broadcast over many quadrature points can
    # precompute them once and pass them in to avoid recomputing them per point.
    λ = TD.liquid_fraction(thp, T, max(zero(ρ), q_lcl), max(zero(ρ), q_icl)),
    mu_S = q_tot_nonneg - TD.q_vap_saturation(thp, T, ρ),
    ice_ramp_T_low = zero(ρ),
    ice_ramp_T_high = zero(ρ),
    precip_incloud_fraction = zero(ρ),
    snow_incloud_fraction = -one(ρ),
    precip_overlap_decay = -one(ρ),
    precip_frac = one(ρ),
    sigma_S = zero(ρ),
    CF_d = zero(ρ),
    precip_shaft_random = zero(ρ),
    precip_frac_floor = sgs_precip_fraction_min(typeof(ρ)),
    args...,
)
    FT = typeof(ρ)
    # Clamp specific humidities to non-negative.
    q_rai_nonneg = max(FT(0), q_rai)
    q_sno_nonneg = max(FT(0), q_sno)

    # Same transform `integrate_over_sgs` builds; shared by the (optional)
    # precipitation-fraction pass and the tendency pass.
    transform =
        build_physical_transform(sgs_quad, q_tot_nonneg, T, q′q′, T′T′, corr_Tq)
    # Shaft weight on the nodes: the moist half of the PDF (S′ ≥ 0), or under
    # the overlap precipitation fraction the moistest `a_p` of it, smoothed
    # over `c_w σ_S`. Its quadrature weight `cf_precip` normalizes the placement.
    β_snow = ifelse(
        snow_incloud_fraction < zero(FT), FT(precip_incloud_fraction),
        FT(snow_incloud_fraction),
    )
    overlap_on = precip_overlap_decay >= zero(FT)
    # Sub-population mode: cloudy nodes plus a random share of the clear nodes
    # (`sgs_precip_subpopulation`); no node flag pass is needed, `CF_d` is the
    # shaft's cloudy weight.
    random_on =
        overlap_on & (precip_shaft_random > zero(FT)) &
        (precip_incloud_fraction > zero(FT))
    # The shaft is never taken narrower than `precip_frac_floor` of the cell
    # (`sgs_precip_fraction_floor`; at least `sgs_precip_fraction_min`).
    a_p = max(FT(precip_frac), FT(precip_frac_floor))
    p_clear, conc = sgs_precip_subpopulation(a_p, FT(CF_d))
    p_clear = ifelse(random_on, p_clear, -one(FT))
    ε_w = discrete_cloudy_weight_width(α, FT(sigma_S))
    S_star = ifelse(
        overlap_on, sgs_precip_shaft_threshold(a_p, FT(sigma_S)), zero(FT),
    )
    ε_S = ifelse(overlap_on, sgs_precip_shaft_width_coeff(FT) * FT(sigma_S), zero(FT))
    cf_precip = if ((precip_incloud_fraction > zero(FT)) | (β_snow > zero(FT))) &
       !random_on
        cf = sum_over_quadrature_points(
            SGSPrecipShaftFlag(thp, ρ, mu_S, S_star, ε_S), transform, sgs_quad,
        )
        # A shaft the quadrature cannot resolve is left uniform.
        ifelse(cf < sgs_precip_weight_min(FT), zero(FT), cf)
    else
        one(FT)
    end
    evaluator = Microphysics1MEvaluator(
        scheme, cmp, thp, ρ, w,
        q_rai_nonneg, q_sno_nonneg, λ,
        FT(ξ_liq), FT(ξ_ice), max(zero(ρ), q_lcl), max(zero(ρ), q_icl),
        λ_lagrange, mu_S, α, dt, nsubs, args,
        FT(ice_ramp_T_low), FT(ice_ramp_T_high),
        FT(precip_incloud_fraction), FT(cf_precip), β_snow, S_star, ε_S,
        p_clear, conc, ε_w,
    )
    return sum_over_quadrature_points(evaluator, transform, sgs_quad)
end

"""
    SGSMicrophysicsOptions{FT}

Scalar options of the 1M quadrature microphysics, built by
`sgs_microphysics_options` and broadcast as a scalar (see that function).
"""
Base.@kwdef struct SGSMicrophysicsOptions{FT}
    α::FT
    ξ_liq::FT
    ξ_ice::FT
    ice_ramp_T_low::FT
    ice_ramp_T_high::FT
    precip_incloud_fraction::FT
    snow_incloud_fraction::FT
    precip_overlap_decay::FT
    precip_shaft_random::FT
    precip_frac_floor::FT
end
Base.broadcastable(o::SGSMicrophysicsOptions) = tuple(o)

"""
    sgs_microphysics_options(params)

The per-run scalar options of the 1M quadrature microphysics as one
`SGSMicrophysicsOptions` (`α`, `ξ_liq`, `ξ_ice`, the ξ_ice ramp, and the
precipitation-placement keys), so the tendency broadcast passes one argument
instead of ten: a broadcast with more than ~30 arguments falls off Julia's
specialized `Broadcast._getindex` path and allocates inside the GPU kernel.
"""
function sgs_microphysics_options(params)
    return SGSMicrophysicsOptions(;
        α = sgs_variance_fidelity(CAP.cloud_fraction_steepness_scale(params)),
        ξ_liq = CAP.sgs_liquid_uniform_fraction(params),
        ξ_ice = CAP.sgs_ice_uniform_fraction(params),
        ice_ramp_T_low = CAP.sgs_ice_uniform_ramp_T_low(params),
        ice_ramp_T_high = CAP.sgs_ice_uniform_ramp_T_high(params),
        precip_incloud_fraction = CAP.sgs_precip_incloud_fraction(params),
        snow_incloud_fraction = CAP.sgs_snow_incloud_fraction(params),
        precip_overlap_decay = CAP.sgs_precip_overlap_decay(params),
        precip_shaft_random = CAP.sgs_precip_shaft_random(params),
        precip_frac_floor = CAP.sgs_precip_fraction_floor(params),
    )
end

"""
    microphysics_tendencies_1m(
        scheme, sgs_quad, cmp, thp, ρ, T, w, q_tot_nonneg, q_lcl, q_icl, q_rai, q_sno,
        T′T′, q′q′, corr_Tq, λ_lagrange, dt, nsubs, λ, mu_S, precip_frac, sigma_S, CF_d,
        opts::SGSMicrophysicsOptions,
    )

Packed form of the quadrature driver: the scalar options come as the
`SGSMicrophysicsOptions` of `sgs_microphysics_options`, the per-cell fields (`precip_frac`, `sigma_S`,
`CF_d`) and the precomputed `λ`, `mu_S` positionally. Forwards to the positional
form, so the two are identical.
"""
@inline function microphysics_tendencies_1m(
    scheme, sgs_quad, cmp, thp, ρ, T, w, q_tot_nonneg,
    q_lcl, q_icl, q_rai, q_sno, T′T′, q′q′, corr_Tq,
    λ_lagrange, dt, nsubs, λ, mu_S, precip_frac, sigma_S, CF_d,
    opts::SGSMicrophysicsOptions,
)
    return microphysics_tendencies_1m(
        scheme, sgs_quad, cmp, thp, ρ, T, w, q_tot_nonneg,
        q_lcl, q_icl, q_rai, q_sno, T′T′, q′q′, corr_Tq,
        λ_lagrange, opts.α, opts.ξ_liq, opts.ξ_ice, dt, nsubs, λ, mu_S,
        opts.ice_ramp_T_low, opts.ice_ramp_T_high,
        opts.precip_incloud_fraction, opts.snow_incloud_fraction,
        opts.precip_overlap_decay, precip_frac, sigma_S, CF_d,
        opts.precip_shaft_random, opts.precip_frac_floor,
    )
end

###
### 2 Moment Microphysics
###

"""
    compute_prescribed_aerosol_properties!(
        seasalt_num, seasalt_mean_radius, sulfate_num,
        prescribed_aerosol_field, aerosol_params,
    )

Compute prescribed sea salt and sulfate aerosol number concentrations and the sea
salt geometric mean radius, overwriting the first three arguments.

Aerosol mass mixing ratios are converted to number concentrations using the
per-mode particle radii and densities in `aerosol_params`. Sea salt aggregates all
available `:SSLT0X` modes; its geometric mean radius is the number-weighted mean of
`log(radius)`, exponentiated.

# Arguments

  - `seasalt_num`: Overwritten with the total sea salt number concentration [kg⁻¹].
  - `seasalt_mean_radius`: Overwritten with the sea salt geometric mean radius [m],
    set to zero where no sea salt is present.
  - `sulfate_num`: Overwritten with the total sulfate number concentration [kg⁻¹].
  - `prescribed_aerosol_field`: Container of aerosol mass mixing ratios (e.g.
    `:SSLT01`, `:SO4`) [kg/kg].
  - `aerosol_params`: Aerosol properties (density, mode radius, geometric standard
    deviation, hygroscopicity).

The return value is unused; the results are the mutated arguments.
"""
function compute_prescribed_aerosol_properties!(
    seasalt_num, seasalt_mean_radius, sulfate_num,
    prescribed_aerosol_field, aerosol_params,
)

    FT = eltype(aerosol_params)
    @. seasalt_num = 0
    @. seasalt_mean_radius = 0
    @. sulfate_num = 0

    # Get aerosol concentrations if available
    seasalt_names = (:SSLT01, :SSLT02, :SSLT03, :SSLT04, :SSLT05)
    seasalt_radius_props =
        (:SSLT01_radius, :SSLT02_radius, :SSLT03_radius, :SSLT04_radius, :SSLT05_radius)
    sulfate_names = (:SO4,)
    for aerosol_name in propertynames(prescribed_aerosol_field)
        if aerosol_name in seasalt_names
            # Find the index of the sea salt mode to get the corresponding radius property
            idx = findfirst(isequal(aerosol_name), seasalt_names)
            seasalt_particle_radius = getproperty(aerosol_params, seasalt_radius_props[idx])
            seasalt_particle_mass =
                FT(4 / 3 * pi) *
                seasalt_particle_radius^3 *
                aerosol_params.seasalt_density
            seasalt_mass = getproperty(prescribed_aerosol_field, aerosol_name)
            @. seasalt_num += seasalt_mass / seasalt_particle_mass
            @. seasalt_mean_radius +=
                seasalt_mass / seasalt_particle_mass *
                log(seasalt_particle_radius)
        elseif aerosol_name in sulfate_names
            sulfate_particle_mass =
                FT(4 / 3 * pi) *
                aerosol_params.sulfate_radius^3 *
                aerosol_params.sulfate_density
            sulfate_mass = getproperty(prescribed_aerosol_field, aerosol_name)
            @. sulfate_num += sulfate_mass / sulfate_particle_mass
        end
    end
    # Compute geometric mean radius of the log-normal distribution:
    # exp(weighted average of log(radius))
    @. seasalt_mean_radius =
        ifelse(seasalt_num == 0, 0, exp(seasalt_mean_radius / seasalt_num))
end

"""
    aerosol_activation_sources(
        act_params, seasalt_num, seasalt_mean_radius, sulfate_num,
        qₜ, qₗ, qᵢ, nₗ, ρ, w, cmp, thermo_params, T, p, dt, aerosol_params,
    )

Compute the cloud droplet number source from aerosol activation, following the
Abdul-Razzak and Ghan (2000) parameterization.

Activation of a bimodal (sea salt plus sulfate) aerosol distribution is evaluated
at the local supersaturation and vertical velocity, and the activated number
`n_act` is relaxed onto the existing droplet number over one timestep:
`(n_act - nₗ) / dt`.

Three guards keep the result physical and keep CloudMicrophysics from throwing:

  - Early return of zero if the environment cannot activate aerosol, namely for
    subsaturated air (`S < 0`), negligible total aerosol number, non-positive
    vertical velocity, non-positive mode radii, or non-finite `S`, `T`, or `p`.
  - Zero if the diagnosed maximum supersaturation `S_max` is below the ambient
    supersaturation `S`, or if `n_act` is not finite.
  - Zero if `n_act < nₗ`, so activation never removes existing droplets; the
    tendency is one-sided by construction.

# Arguments

  - `act_params`: Aerosol activation parameters (`AerosolActivationParameters`).
  - `seasalt_num`: Sea salt number concentration per mass of air [kg⁻¹].
  - `seasalt_mean_radius`: Geometric mean dry radius of the sea salt mode [m].
  - `sulfate_num`: Sulfate number concentration per mass of air [kg⁻¹].
  - `qₜ`: Total water specific humidity [kg/kg].
  - `qₗ`: Liquid water (cloud plus rain) specific humidity [kg/kg].
  - `qᵢ`: Ice water (cloud ice plus snow) specific humidity [kg/kg].
  - `nₗ`: Cloud droplet number concentration per mass of air [kg⁻¹].
  - `ρ`: Air density [kg/m³].
  - `w`: Vertical velocity [m/s].
  - `cmp`: `CMP.Microphysics2MParams` parameters.
  - `thermo_params`: Thermodynamics parameters.
  - `T`: Air temperature [K].
  - `p`: Air pressure [Pa].
  - `dt`: Model timestep [s].
  - `aerosol_params`: Prescribed aerosol properties (sea salt and sulfate widths,
    radii, hygroscopicities).

# Returns

Tendency of cloud droplet number concentration per mass of air [kg⁻¹/s], zero or
positive.
"""
function aerosol_activation_sources(
    act_params, seasalt_num, seasalt_mean_radius, sulfate_num,
    qₜ, qₗ, qᵢ, nₗ, ρ, w, cmp, thermo_params, T, p, dt, aerosol_params,
)
    FT = eltype(nₗ)
    air_params = cmp.warm_rain.air_properties
    q_vap = qₜ - qₗ - qᵢ
    S = TD.supersaturation(thermo_params, q_vap, ρ, T, TD.Liquid())
    n_aer = seasalt_num + sulfate_num

    # Extract aerosol properties
    seasalt_std = aerosol_params.seasalt_std
    seasalt_kappa = aerosol_params.seasalt_kappa
    sulfate_radius = aerosol_params.sulfate_radius
    sulfate_std = aerosol_params.sulfate_std
    sulfate_kappa = aerosol_params.sulfate_kappa

    # Early exit for invalid inputs (negative supersaturation, no aerosols, or
    # non-physical values that would cause DomainError in CMP)
    invalid_inputs =
        (S < FT(0)) || (n_aer < ϵ_numerics(FT)) || (w <= FT(0)) ||
        (seasalt_mean_radius <= FT(0)) || (sulfate_radius <= FT(0)) ||
        !isfinite(S) || !isfinite(T) || !isfinite(p)

    # Short-circuit to avoid expensive CMAA calls that may throw DomainError
    if invalid_inputs
        return FT(0)
    end

    # Mode_κ constructor: (r_dry, stdev, N, vol_mix_ratio, mass_mix_ratio, molar_mass, kappa)
    # For single-component aerosols, vol_mix_ratio and mass_mix_ratio are (1,).
    # NOTE: molar_mass is set to (0,) because it is NOT USED by the functions we call
    # (max_supersaturation, N_activated_per_mode, total_N_activated). These only use
    # vol_mix_ratio and kappa for Mode_κ hygroscopicity calculations. However, if
    # M_activated_per_mode were ever called, it would incorrectly return 0 due to this.
    # TODO: Add proper molar masses (seasalt ~58.44 g/mol NaCl, sulfate ~132.14 g/mol (NH4)2SO4)
    # to the prescribed_aerosol_params if M_activated is needed in the future.
    seasalt_mode = CMAM.Mode_κ(
        seasalt_mean_radius,                 # r_dry: geometric mean dry radius [m]
        seasalt_std,                         # stdev: geometric standard deviation
        max(FT(0), seasalt_num) * ρ,         # N: number concentration [#/m³]
        (FT(1),),                            # vol_mix_ratio: volume mixing ratio (pure component)
        (FT(1),),                            # mass_mix_ratio: mass mixing ratio (pure component)
        (FT(0),),                            # molar_mass: [kg/mol] (unused, see note above)
        (seasalt_kappa,),                    # kappa: hygroscopicity parameter
    )
    sulfate_mode = CMAM.Mode_κ(
        sulfate_radius,                      # r_dry: geometric mean dry radius [m]
        sulfate_std,                         # stdev: geometric standard deviation
        max(FT(0), sulfate_num) * ρ,         # N: number concentration [#/m³]
        (FT(1),),                            # vol_mix_ratio: volume mixing ratio (pure component)
        (FT(1),),                            # mass_mix_ratio: mass mixing ratio (pure component)
        (FT(0),),                            # molar_mass: [kg/mol] (unused, see note above)
        (sulfate_kappa,),                    # kappa: hygroscopicity parameter
    )
    distribution = CMAM.AerosolDistribution((seasalt_mode, sulfate_mode))
    args = (
        act_params, distribution, air_params, thermo_params,
        T, p, w, qₜ, qₗ, qᵢ, nₗ * ρ, FT(0),
    )

    # Compute maximum supersaturation and activated aerosol number
    S_max = CMAA.max_supersaturation(args...)
    n_act = CMAA.total_N_activated(args...) / ρ

    # Determine tendency: zero if supersaturation too low,
    # NaN result, or activation would decrease droplet count
    return ifelse(
        S_max < S || !isfinite(n_act) || n_act < nₗ,
        FT(0),
        (n_act - nₗ) / dt,
    )
end

"""
    compute_2m_precipitation_tendencies!(
        mp_tendency, ρ, qₜ, qₗ, nₗ, qᵣ, nᵣ, T, dt, mp, thp, timestepping,
    )

Fill the 2-moment microphysics tendency field and apply the explicit-stepping
limiters.

Evaluates `BMT.bulk_microphysics_tendencies(BMT.Microphysics2Moment(), ...)` over
the field (condensation/evaporation, autoconversion, accretion, rain evaporation,
and self-collection), then calls `apply_2m_tendency_limits!`, which is a no-op for
`Implicit` timestepping and scales coupled mass/number sinks for `Explicit`.

# Arguments

  - `mp_tendency`: Field of tendency NamedTuples, overwritten in place.
  - `ρ`: Air density [kg/m³].
  - `qₜ`: Total water specific humidity [kg/kg].
  - `qₗ`: Cloud liquid specific humidity [kg/kg].
  - `nₗ`: Cloud droplet number concentration per mass [kg⁻¹].
  - `qᵣ`: Rain specific humidity [kg/kg].
  - `nᵣ`: Rain drop number concentration per mass [kg⁻¹].
  - `T`: Air temperature [K].
  - `dt`: Model timestep [s], used by the limiter.
  - `mp`: Microphysics parameters (`CMP.Microphysics2MParams`).
  - `thp`: Thermodynamics parameters.
  - `timestepping`: `Implicit`, `Explicit`, or `nothing`; selects the limiter.

Mutates `mp_tendency`; the return value is unused.
"""
function compute_2m_precipitation_tendencies!(
    mp_tendency, ρ, qₜ, qₗ, nₗ, qᵣ, nᵣ, T, dt, mp, thp, timestepping,
)
    @. mp_tendency = BMT.bulk_microphysics_tendencies(
        BMT.Microphysics2Moment(), mp, thp, ρ, T, qₜ, qₗ, nₗ, qᵣ, nᵣ,
    )
    apply_2m_tendency_limits!(mp_tendency, timestepping, qₗ, nₗ, qᵣ, nᵣ, dt)
end

"""
    microphysics_tendencies_quadrature_2m(
        ::GridMeanSGS, cmp, tps, ρ, T, q_tot, q_liq, n_liq, q_rai, n_rai,
    )
    microphysics_tendencies_quadrature_2m(
        SG_quad::SGSQuadrature, cmp, tps, ρ, T, q_tot, q_liq, n_liq, q_rai, n_rai,
    )

Evaluate 2-moment microphysics tendencies on the SGS-quadrature interface.

!!! warning "Limited SGS support"

    Only `GridMeanSGS` is implemented; it evaluates CloudMicrophysics once at the
    grid mean. The `SGSQuadrature` method throws an error. Full quadrature
    integration for 2M would need an evaluator that also perturbs the number
    concentrations.

# Arguments

  - `SG_quad`: SGS distribution; only `GridMeanSGS` is supported.
  - `cmp`: 2M microphysics parameters.
  - `tps`: Thermodynamics parameters.
  - `ρ`: Air density [kg/m³].
  - `T`: Air temperature [K].
  - `q_tot`: Total water specific humidity [kg/kg].
  - `q_liq`: Cloud liquid specific humidity [kg/kg].
  - `n_liq`: Cloud droplet number concentration per mass [kg⁻¹].
  - `q_rai`: Rain specific humidity [kg/kg].
  - `n_rai`: Rain drop number concentration per mass [kg⁻¹].

# Returns

The CloudMicrophysics 2M tendency NamedTuple: mass and number tendencies
`dq_lcl_dt`, `dn_lcl_dt`, `dq_rai_dt`, `dn_rai_dt` [kg/kg/s and kg⁻¹/s], plus the
ice-phase entries `dq_ice_dt`, `dq_rim_dt`, `db_rim_dt`, which are identically
zero for warm-rain-only parameters.
"""
@inline function microphysics_tendencies_quadrature_2m(
    ::GridMeanSGS, cmp, tps, ρ, T, q_tot, q_liq, n_liq, q_rai, n_rai,
)
    # Direct GridMeanSGS dispatch for 2M: evaluates BMT at grid mean.
    return BMT.bulk_microphysics_tendencies(
        BMT.Microphysics2Moment(), cmp, tps, ρ, T,
        q_tot, q_liq, n_liq, q_rai, n_rai,
    )
end
@inline function microphysics_tendencies_quadrature_2m(
    SG_quad::SGSQuadrature, cmp, tps, ρ, T,
    q_tot, q_liq, n_liq, q_rai, n_rai,
)
    error("Not implemented yet")
    return nothing
end
