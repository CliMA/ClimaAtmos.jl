import ClimaComms
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

# `mp` and `tps` may arrive `Val`-wrapped. A `Val{p}()` is zero-size and carries
# its value in its type, so the field costs nothing and the parameters fold to
# literals instead of being read back from the struct -- worth 162 registers and
# a doubling of occupancy in the SGS quadrature kernel, which is latency-bound
# at the 255-register cap. `_unwrapped` is the identity for bare parameters.
@inline _unwrapped(p) = p
@inline _unwrapped(::Val{P}) where {P} = P

# Only on the device. A struct type parameter is boxed, so on the host each use
# copies it back out and allocates per point; the device compiler has no heap
# and folds it to a literal instead.
@inline _fold(::ClimaComms.AbstractCPUDevice, p) = p
@inline _fold(::ClimaComms.AbstractDevice, p) = Val(p)

"""
    Microphysics1MEvaluator{S, MP, TPS, FT, Args, PL}

GPU-safe functor evaluating 1-moment microphysics tendencies at the nodes of
the conditional-moment rule (`sum_over_conditional_nodes`).

The local condensate at each point follows the truncated-Gaussian
Lagrange-multiplier closure described in `microphysics_tendencies_1m`.
Precipitation (`q_rai`, `q_sno`), the liquid fraction `λ`, and the closure
quantities (`λ_lagrange`, `λ_eq`, `mu_S`, `α`) are grid-cell constants held fixed across
the nodes.

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
    nodes [-].
  - `q_lcl`, `q_icl`: Subdomain-mean cloud liquid and cloud ice [kg/kg].
  - `λ_lagrange`: Lagrange multiplier enforcing
    `E[max(0, λ_lagrange + α·S′)] = q_c` (`_compute_sgs_moments`) [kg/kg].
  - `λ_eq`: Equilibrium Lagrange multiplier enforcing
    `E[max(0, λ_eq + S′)] = q_c_eq` (with `α = 1`), from which the
    node saturation excess of cloudy equilibrium nodes is reconstructed [kg/kg].
  - `mu_S`: Linearized SGS mean saturation excess `q_tot − q_sat(T, ρ)` [kg/kg].
  - `α`: Variance fidelity parameter [-].
  - `β_precip`, `shaft_frac`: Precipitation-fraction placement of the cell-mean
    rain and snow onto the shaft nodes and the node weight of the shaft
    (`conditional_rule_shaft_fraction`); `β_precip = 0` disables it.
  - `dt`: Timestep used for the time-averaged process rates [s].
  - `nsubs`: Number of substeps in the tendency averaging.
  - `args`: Extra trailing arguments forwarded to the CloudMicrophysics call.
  - `placement`: `ShaftPrecipPlacement()` (the default; the placement is the
    shaft of `β_precip`), or a normalized `LogLinearNodePlacement`, which
    replaces it.
"""
struct Microphysics1MEvaluator{S, MP, TPS, FT, Args <: Tuple, PL}
    scheme::S
    mp::MP
    tps::TPS
    ρ::FT
    w::FT              # physical vertical velocity [m/s]
    # Precipitation
    q_rai::FT
    q_sno::FT
    λ::FT              # liquid fraction (from thermodynamics, held fixed)
    # Uniform fractions of each species over the nodes and the subdomain
    # means they draw on
    ξ_liq::FT
    ξ_ice::FT
    q_lcl::FT
    q_icl::FT
    # Truncated-Gaussian Lagrange multipliers, μ_S, and liquid fraction
    λ_lagrange::FT # Lagrange multiplier for actual condensate q_c
    λ_eq::FT       # equilibrium multiplier for q_c_eq (α = 1)
    mu_S::FT       # linearized SGS mean μ_S = q_tot_mean − q_sat(T_mean, ρ)
    α::FT          # variance fidelity parameter (from sgs_variance_fidelity)
    # Precipitation-fraction placement
    β_precip::FT
    shaft_frac::FT
    # Numerical parameters
    dt::FT
    nsubs::Int
    args::Args
    # Rain and snow placement other than the shaft (zero-size by default)
    placement::PL
end

# Without `placement`: the shaft placement of `β_precip`.
@inline Microphysics1MEvaluator(
    scheme, mp, tps, ρ, w,
    q_rai, q_sno, λ,
    ξ_liq, ξ_ice, q_lcl, q_icl,
    λ_lagrange, λ_eq, mu_S, α,
    β_precip, shaft_frac,
    dt, nsubs, args,
) = Microphysics1MEvaluator(
    scheme, mp, tps, ρ, w,
    q_rai, q_sno, λ,
    ξ_liq, ξ_ice, q_lcl, q_icl,
    λ_lagrange, λ_eq, mu_S, α,
    β_precip, shaft_frac,
    dt, nsubs, args, ShaftPrecipPlacement(),
)

# Without `λ_eq`: the grid-mean equilibrium multiplier `max(0, μ_S − q_rai − q_sno)`,
# as in `microphysics_tendencies_1m`.
@inline Microphysics1MEvaluator(
    scheme, mp, tps, ρ, w,
    q_rai, q_sno, λ,
    ξ_liq, ξ_ice, q_lcl, q_icl,
    λ_lagrange, mu_S, α,
    β_precip, shaft_frac,
    dt, nsubs, args,
) = Microphysics1MEvaluator(
    scheme, mp, tps, ρ, w,
    q_rai, q_sno, λ,
    ξ_liq, ξ_ice, q_lcl, q_icl,
    λ_lagrange, max(zero(ρ), mu_S - (q_rai + q_sno)), mu_S, α,
    β_precip, shaft_frac,
    dt, nsubs, args,
)

"""
    sgs_precip_shaft_threshold(a_p, sigma_S)

Threshold `S_star` on the centred saturation excess such that the shaft covers
the moistest fraction `a_p` of the Gaussian PDF of standard deviation `sigma_S`:

    S_star = σ_S · Φ⁻¹(1 − a_p)

`a_p` is clamped to `[eps, 1]`: `a_p = 1` places the threshold below the whole
PDF (uniform precipitation). The threshold is an edge of the conditional-moment
rule, so the node weight of the shaft (`conditional_rule_shaft_fraction`) is
`a_p` for any `a_p`, and it normalizes the placement.
"""
@inline function sgs_precip_shaft_threshold(a_p, sigma_S)
    FT = typeof(sigma_S)
    a = clamp(FT(a_p), eps(FT), one(FT))
    return sigma_S * normal_cdf_inv(one(FT) - a)
end

"""
    sgs_placement_factor(β, flag, shaft_frac)

Factor `φ` on a cell-mean quantity at a node when a fraction `β` of it is
confined to the flagged nodes (`flag = 1`, total node weight `shaft_frac`) and
the rest stays uniform,

    φ = (1 − β) + β · flag / shaft_frac,

so that the node mean of `φ` is 1 and the cell mean is conserved. `β = 0`
returns exactly 1; `shaft_frac = 0` falls back to uniform.
"""
@inline function sgs_placement_factor(β, flag, shaft_frac)
    FT = typeof(β)
    φ_in = ifelse(
        shaft_frac > zero(FT), flag / max(shaft_frac, eps(FT)), one(FT),
    )
    return (one(FT) - β) + β * φ_in
end

"""
    sgs_local_condensate(λ, shifted_excess, ξ_liq, ξ_ice, q_lcl, q_icl)

Local cloud liquid and cloud ice `(q_lcl_hat, q_icl_hat)` [kg/kg] at one
node from the reconstructed `shifted_excess = max(0, λ_lagrange + α S′)`,
the liquid fraction `λ`, the subdomain means `q_lcl`, `q_icl` and the uniform
fractions `ξ_liq`, `ξ_ice`:

    q_lcl_hat = (1 − ξ_liq) · λ · shifted_excess       + ξ_liq · q_lcl
    q_icl_hat = (1 − ξ_ice) · (1 − λ) · shifted_excess + ξ_ice · q_icl

`ξ = 0` is the excess reconstruction (the species sits only where the node is
saturated, in proportion to its excess); `ξ = 1` is the uniform distribution (the
species is the subdomain mean at every node). Either end, and any blend, conserves
the node mean of the species in cells with condensate, since the excess
share has mean `λ q_c = q_lcl` (resp. `(1 − λ) q_c = q_icl`) by construction of
`λ_lagrange`.
"""
@inline function sgs_local_condensate(λ, shifted_excess, ξ_liq, ξ_ice, q_lcl, q_icl)
    q_lcl_hat = (1 - ξ_liq) * (λ * shifted_excess) + ξ_liq * q_lcl
    q_icl_hat = (1 - ξ_ice) * ((1 - λ) * shifted_excess) + ξ_ice * q_icl
    return (q_lcl_hat, q_icl_hat)
end

"""
    LogLinearNodePlacement{FT}

Log-linear placement of the cell-mean rain and snow over the points of the
conditional-moment rule (`sgs_precip_placement: loglinear`). With
`z = S′/σ_z` the standardized saturation excess the evaluator recovers at a
point, the rain and snow there are the cell means times

    φ_r = exp(κ_r z) / N_r,
    φ_s = exp(κ_s z) ψ(T) / N_s,    ψ(T) = exp(−max(T − T_freeze, 0) / ΔT_m),

with `N_r`, `N_s` the sums of the numerators over all points of the rule,
temperature points included, so that the cell means are conserved exactly
(`normalized_placement`). A fraction `f` of each point precipitates, with
`φ q / f`; the rest has no rain or snow (`_placed_precip_tendencies`).
`κ = 0` (both species), `ΔT_m = ∞` and `f = 1` give uniform rain and snow.

Fields:

  - `κ_r`, `κ_s`: rain and snow tilts [-].
  - `inv_ΔT_m`: `1 / ΔT_m` [1/K]; `0` disables the warm-side depletion `ψ`.
  - `T_freeze`: freezing temperature [K].
  - `T_ref`: `max(T, T_freeze)` of the cell mean, where `ψ` is taken as `1`
    (the constant cancels in the normalization and keeps the exponent small).
  - `σ_z`: width standardizing `S′`, `σ_S` bounded below by the rounding of the
    recovered `S′` [kg/kg].
  - `f`: precipitating fraction of each point, in `(0, 1]` [-].
  - `inv_N_r`, `inv_N_s`: inverse normalizations, `1` until
    `normalized_placement` sets them.

The exponents are bounded at `±20`, which only binds far outside the
calibrated tilts and keeps the factors finite in `Float32`.
"""
struct LogLinearNodePlacement{FT}
    κ_r::FT
    κ_s::FT
    inv_ΔT_m::FT
    T_freeze::FT
    T_ref::FT
    σ_z::FT
    f::FT
    inv_N_r::FT
    inv_N_s::FT
end

"""
    loglinear_placement_weights(pl::LogLinearNodePlacement, S′, T)

Unnormalized rain and snow weights `(exp(κ_r z), exp(κ_s z) ψ(T))` at a point
with centred saturation excess `S′` [kg/kg] and temperature `T` [K], with
`z = S′/σ_z` and `ψ` taken relative to the cell mean (`LogLinearNodePlacement`).
"""
@inline function loglinear_placement_weights(pl::LogLinearNodePlacement, S′, T)
    FT = typeof(S′)
    E = FT(20)
    z = S′ / pl.σ_z
    # (T − T_freeze)⁺ − (T_mean − T_freeze)⁺, in units of ΔT_m
    warmth = (max(T, pl.T_freeze) - pl.T_ref) * pl.inv_ΔT_m
    g_r = exp(clamp(pl.κ_r * z, -E, E))
    g_s = exp(clamp(pl.κ_s * z - warmth, -E, E))
    return (g_r, g_s)
end

"""
    loglinear_node_precip(pl::LogLinearNodePlacement, q_rai, q_sno, S′, T)

Rain and snow `(q_rai φ_r / f, q_sno φ_s / f)` [kg/kg] of the precipitating
part of a point (`LogLinearNodePlacement`), from the cell means `q_rai`,
`q_sno`. Their rule mean times `f` is the cell mean.
"""
@inline function loglinear_node_precip(pl::LogLinearNodePlacement, q_rai, q_sno, S′, T)
    (g_r, g_s) = loglinear_placement_weights(pl, S′, T)
    return (
        q_rai * (g_r * pl.inv_N_r) / pl.f,
        q_sno * (g_s * pl.inv_N_s) / pl.f,
    )
end

"""
    LogLinearPlacementWeights(thp, ρ, mu_S, placement)

Functor `(T, q, flag) ↦ (; r, s)`, the unnormalized weights
`loglinear_placement_weights` at a point of the rule, with `S′` recovered from
`(T, q)` exactly as `Microphysics1MEvaluator` does. Its sum over the rule
(`sum_over_conditional_nodes`) is the normalization of the placement.
"""
struct LogLinearPlacementWeights{TPS, FT}
    thp::TPS
    ρ::FT
    mu_S::FT
    placement::LogLinearNodePlacement{FT}
end
@inline function (g::LogLinearPlacementWeights)(T_hat, q_tot_hat, flag)
    FT = typeof(g.ρ)
    q_tot_hat = max(FT(0), q_tot_hat)
    q_sat_hat = TD.q_vap_saturation(_unwrapped(g.thp), T_hat, g.ρ)
    S′_hat = q_tot_hat - q_sat_hat - g.mu_S
    (g_r, g_s) = loglinear_placement_weights(g.placement, S′_hat, T_hat)
    return (; r = g_r, s = g_s)
end

"""
    normalized_placement(placement, thp, ρ, mu_S, rule, spread)

The placement with its normalization over the points of `rule` and `spread`
(the same points `sum_over_conditional_nodes` evaluates): a
`LogLinearNodePlacement` gets `inv_N_r`, `inv_N_s` from the sum of
`LogLinearPlacementWeights`; `ShaftPrecipPlacement()` is returned as is.
"""
@inline normalized_placement(pl::ShaftPrecipPlacement, thp, ρ, mu_S, rule, spread) = pl
@inline function normalized_placement(
    pl::LogLinearNodePlacement, thp, ρ, mu_S, rule, spread,
)
    FT = typeof(ρ)
    N = sum_over_conditional_nodes(
        LogLinearPlacementWeights(thp, ρ, FT(mu_S), pl), rule, spread,
    )
    return LogLinearNodePlacement(
        pl.κ_r, pl.κ_s, pl.inv_ΔT_m, pl.T_freeze, pl.T_ref, pl.σ_z, pl.f,
        one(FT) / N.r, one(FT) / N.s,
    )
end

"""
    precip_placement_edge(placement, β_precip, S_star)

The precipitation edge of the conditional-moment rule: the shaft edge `S_star`
where the shaft is on (`β_precip > 0`), and, for the log-linear placement,
wherever a tilt is nonzero, so that the rule resolves the moist tail the
placement weights; `+Inf` (no edge) otherwise, which leaves the rule of the
uniform placement.
"""
@inline precip_placement_edge(::ShaftPrecipPlacement, β_precip, S_star) =
    ifelse(β_precip > zero(β_precip), S_star, oftype(S_star, Inf))
@inline precip_placement_edge(pl::LogLinearNodePlacement, β_precip, S_star) =
    ifelse(
        (pl.κ_r != zero(pl.κ_r)) | (pl.κ_s != zero(pl.κ_s)),
        S_star,
        oftype(S_star, Inf),
    )

# `@noinline` here is the SGS node function barrier. The functor body
# below (saturation, shape-function partition, plus the heavy
# `BMT.average_bulk_microphysics_tendencies` call with its `nsubs`
# substep loop and 4×4 linearized operator) gets invoked at every node and
# temperature point of `sum_over_conditional_nodes`. Without the barrier those copies inline
# into one giant GPU broadcast kernel, pushing register pressure past the
# 255-reg hard cap and pinning occupancy at 12.5%. Marking the functor
# itself (vs a trivial forwarding wrapper) is the strongest signal we can
# give LLVM/NVPTX not to re-inline this — the body is multi-statement and
# meaningfully sized, so the late inliner won't undo it.
"""
    (eval::Microphysics1MEvaluator)(T_hat, q_tot_hat, flag_p)

Evaluate the 1-moment tendencies at one node `(T_hat, q_tot_hat)` [K, kg/kg]
with the node's shaft flag `flag_p` (`1` inside the precipitation shaft, `0`
outside; the intervals of the conditional-moment rule lie entirely on one
side of the shaft edge).

The local cloud condensate is obtained from the centred saturation excess
`S′_hat = (q_tot_hat − q_sat(T_hat, ρ)) − mu_S`:

    shifted_excess = max(0, λ_lagrange + α · S′_hat)
    q_lcl_hat      = (1 − ξ_liq) · λ · shifted_excess       + ξ_liq · q_lcl
    q_icl_hat      = (1 − ξ_ice) · (1 − λ) · shifted_excess + ξ_ice · q_icl

The Lagrange multiplier `λ_lagrange` is set (in `_compute_sgs_moments`) so
that `E[shifted_excess] = q_c`, where `q_c = q_lcl + q_icl` is the grid-mean
*cloud* condensate, excluding precipitation. The reconstruction therefore
partitions `shifted_excess` into local cloud liquid and ice by the liquid
fraction; the uniform fractions `ξ_liq`, `ξ_ice` replace part of a
species' share by its subdomain mean at every node (see
`sgs_local_condensate`). Precipitation is held constant across
the nodes (up to the placement) and is accounted for downstream, where CloudMicrophysics
subtracts it from `q_tot` to diagnose the local vapor.

`q_tot_hat` is clamped non-negative first. Subsaturated nodes contribute zero
condensate but still drive rain evaporation and snow sublimation against the
local vapor.

# Returns

NamedTuple from `BMT.bulk_microphysics_tendencies(BMT.LinearizedAverage(), ...)`
with `dq_lcl_dt`, `dq_icl_dt`, `dq_rai_dt`, `dq_sno_dt` [kg/kg/s].
"""
@noinline function (eval::Microphysics1MEvaluator)(T_hat, q_tot_hat, flag_p)
    FT = typeof(eval.ρ)
    q_tot_hat = max(FT(0), q_tot_hat)

    # Local cloud condensate from the Lagrange-multiplier closure.
    # The mass conservation equation is E[max(0, λ + α·S′)] = q_c, so the
    # local shifted excess at each node is λ + α·S′_hat where
    # S′_hat = (q_tot_hat − q_sat_hat) − μ_S is the centred saturation excess.
    # Precipitation in q_tot needs no special handling here: its mean level
    # cancels in the centred S′ (the level is re-anchored by λ_lagrange, set
    # by cloud-only q_c), and CloudMicrophysics subtracts q_rai/q_sno from
    # q_tot_hat when it diagnoses the local vapor. Subtracting them from the
    # cloud condensate as well would double-count them and break ⟨q_c^local⟩ = q_c.
    q_sat_hat = TD.q_vap_saturation(_unwrapped(eval.tps), T_hat, eval.ρ)
    S′_hat = q_tot_hat - q_sat_hat - eval.mu_S
    αS′_hat = eval.α * S′_hat
    shifted_excess = max(FT(0), eval.λ_lagrange + αS′_hat)
    q_lcl_hat, q_icl_hat = sgs_local_condensate(
        eval.λ, shifted_excess, eval.ξ_liq, eval.ξ_ice, eval.q_lcl, eval.q_icl,
    )
    # Reconstruct the node saturation excess on cloudy equilibrium nodes from
    # `λ_eq` so that condensation and evaporation relax toward the continuous
    # truncated-Gaussian equilibrium `q_c_eq`, while preserving raw
    # subsaturation on clear nodes (`q_c_eq_hat == 0`) for below-cloud rain
    # evaporation and snow sublimation. The equilibrium node uses the
    # unattenuated fluctuation `S′_hat`: `α` attenuates only the actual
    # condensate, which lags equilibrium, and `λ_eq` has `α = 1`.
    q_c_eq_hat = max(FT(0), eval.λ_eq + S′_hat)
    q_precip = eval.q_rai + eval.q_sno
    S_raw_cloud = q_tot_hat - q_precip - q_sat_hat
    S_eff_cloud = ifelse(
        q_c_eq_hat > zero(FT),
        q_c_eq_hat,
        min(zero(FT), S_raw_cloud),
    )
    q_tot_eff = max(FT(0), q_sat_hat + S_eff_cloud + q_precip)

    return _placed_precip_tendencies(
        eval.placement, eval, T_hat, S′_hat, q_tot_eff, q_lcl_hat, q_icl_hat,
        q_precip, flag_p,
    )
end

# The CloudMicrophysics call of `Microphysics1MEvaluator` at one (sub-)point.
@inline _node_bmt_1m(eval, T_hat, q_tot, q_lcl, q_icl, q_rai, q_sno) =
    BMT.bulk_microphysics_tendencies(
        BMT.LinearizedAverage(),
        eval.scheme, _unwrapped(eval.mp), _unwrapped(eval.tps),
        eval.ρ, T_hat, eval.w,
        q_tot, q_lcl, q_icl, q_rai, q_sno,
        eval.dt, eval.nsubs, eval.args...,
    )

"""
    _placed_precip_tendencies(placement, eval, T_hat, S′_hat, q_tot_eff, q_lcl_hat, q_icl_hat, q_precip, flag_p)

The tendencies of `Microphysics1MEvaluator` at the node point once its
condensate (`q_lcl_hat`, `q_icl_hat`) and its total water with the cell-mean
precipitation (`q_tot_eff`, `q_precip = q_rai + q_sno`) are set, with the rain
and snow placed by `placement`. Either placement shifts the node total water by
the change of the precipitation, so that the node vapour, and hence the
condensate reconstruction, is unchanged by it.
"""
@inline function _placed_precip_tendencies(
    ::ShaftPrecipPlacement, eval, T_hat, S′_hat, q_tot_eff, q_lcl_hat, q_icl_hat,
    q_precip, flag_p,
)
    FT = typeof(eval.ρ)
    # Precipitation-fraction placement: the rain and snow this node sees are
    # the cell means scaled by φ_p (1 when disabled); the node total water is
    # shifted by the same amount so that the node vapour, and hence the
    # condensate reconstruction, is unchanged by the placement.
    q_rai_node, q_sno_node, q_tot_node =
        if eval.β_precip > zero(FT)
            φ_p = sgs_placement_factor(eval.β_precip, flag_p, eval.shaft_frac)
            shift = (φ_p - one(FT)) * q_precip
            (eval.q_rai * φ_p, eval.q_sno * φ_p, max(FT(0), q_tot_eff + shift))
        else
            (eval.q_rai, eval.q_sno, q_tot_eff)
        end

    return _node_bmt_1m(
        eval, T_hat, q_tot_node, q_lcl_hat, q_icl_hat, q_rai_node, q_sno_node,
    )
end
@inline function _placed_precip_tendencies(
    pl::LogLinearNodePlacement, eval, T_hat, S′_hat, q_tot_eff, q_lcl_hat,
    q_icl_hat, q_precip, flag_p,
)
    FT = typeof(eval.ρ)
    f = pl.f
    # The precipitating sub-point (`i = 1`): the node's rain and snow
    # concentrated into the fraction `f`. The other sub-point (`i = 2`) has
    # none, and is skipped when the whole node precipitates or nothing does.
    # The total water carries each sub-point's precipitation in place of the
    # cell mean. The sub-points share one call site, so the inlined
    # CloudMicrophysics body is not duplicated in the node function (see the
    # `@noinline` note above).
    (q_rai_p, q_sno_p) =
        loglinear_node_precip(pl, eval.q_rai, eval.q_sno, S′_hat, T_hat)
    q_tot_cloud = q_tot_eff - q_precip
    n = ifelse((f >= one(FT)) | !(q_precip > zero(FT)), 1, 2)
    R = (;
        dq_lcl_dt = zero(FT), dq_icl_dt = zero(FT),
        dq_rai_dt = zero(FT), dq_sno_dt = zero(FT),
    )
    for i in 1:n
        q_r = ifelse(i == 1, q_rai_p, zero(FT))
        q_s = ifelse(i == 1, q_sno_p, zero(FT))
        w = ifelse(n == 1, one(FT), ifelse(i == 1, f, one(FT) - f))
        R_i = _node_bmt_1m(
            eval, T_hat, max(FT(0), q_tot_cloud + (q_r + q_s)),
            q_lcl_hat, q_icl_hat, q_r, q_s,
        )
        # (the closure captures values only: a captured type would be boxed)
        R = map((r, r_i) -> r + w * r_i, R, R_i)
    end
    return R
end

"""
    microphysics_tendencies_1m(
        ρ, q_tot_nonneg, q_lcl, q_icl, q_rai, q_sno, T, w, cmp, thp, dt, nsubs,
    )
    microphysics_tendencies_1m(
        scheme, sgs_quad, cmp, thp, ρ, T, w, q_tot_nonneg,
        q_lcl, q_icl, q_rai, q_sno, T′T′, q′q′, corr_Tq,
        λ_lagrange, α, ξ_liq, ξ_ice, dt, nsubs, λ = ..., mu_S = ...,
        precip_incloud_fraction = 0, precip_frac = 1, sigma_S = (linearized),
        precip_frac_floor = 0, λ_eq = max(0, mu_S − q_rai − q_sno), args...,
    )

Compute time-averaged 1-moment microphysics tendencies.

The 11-argument (no `sgs_quad`) form takes the condensate inputs as-is and is used
in EDMF updrafts, or wherever a grid-mean state is to be evaluated directly: a
single CloudMicrophysics call with no SGS averaging.

The quadrature form integrates over the Gaussian SGS PDF with the
conditional-moment node rule and the truncated-Gaussian Lagrange-multiplier
closure; see `_sum_1m_over_sgs` for the rule and `Microphysics1MEvaluator` for
the node-wise condensate diagnosis. Rain and snow are clamped non-negative before
the integration. Subsaturated nodes contribute below-cloud rain evaporation and
snow sublimation; saturated nodes drive autoconversion and accretion. Without SGS
sampling (`GridMeanSGS`) it is the grid-mean form.

# Arguments

  - `scheme`: CloudMicrophysics scheme tag (from `BulkMicrophysicsTendencies`).
  - `sgs_quad`: `SGSQuadrature` configuration (its `T_min`, `q_max` bound the nodes).
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
    enforce `E[max(0, λ_lagrange + α·S′)] = q_c` [kg/kg].
  - `α`: Variance fidelity parameter from `sgs_variance_fidelity` [-].
  - `ξ_liq`, `ξ_ice`: Uniform fractions of cloud liquid and cloud ice over the
    nodes (`sgs_liquid_uniform_fraction`, `sgs_ice_uniform_fraction`, in
    `[0, 1]`); `0` is the excess reconstruction, see `sgs_local_condensate`.
  - `dt`: Timestep [s].
  - `nsubs`: Number of substeps for tendency averaging.
  - `λ`: Liquid fraction [-]; defaults to `TD.liquid_fraction` at the mean state.
  - `mu_S`: Linearized SGS mean saturation excess [kg/kg]; defaults to
    `q_tot_nonneg − q_sat(T, ρ)`. Both are cell constants and may be
    precomputed by the caller.
  - `precip_incloud_fraction`: Fraction `β_p` of the cell-mean rain and snow confined
    to the precipitation shaft of the PDF; `0` disables the placement [-].
  - `precip_frac`: Overlap precipitation fraction `a_p` of the cell [-].
  - `sigma_S`: SGS saturation-excess standard deviation (`ᶜsgs_moments.sigma_S`),
    which sets the nodes and the shaft threshold; positive (`_compute_sgs_moments`
    and the default, the linearized width `_sgs_linear_sigma_S` of the given
    (co)variances, are floored at `ϵ_numerics`) [kg/kg].
  - `precip_frac_floor`: the shaft is never taken narrower than this fraction of
    the cell, which bounds the factor on the shaft nodes' rain and snow,
    `φ_p ≤ (1 − β_p) + β_p / precip_frac_floor`; it must be positive when
    `precip_incloud_fraction > 0` (`sgs_microphysics_options` checks the model
    parameters). Defaults to `0`, with the placement off [-].
  - `λ_eq`: Equilibrium Lagrange multiplier from `ᶜsgs_moments`, with `α = 1`,
    so that `E[max(0, λ_eq + S′)] = q_c_eq` [kg/kg].
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
        BMT.Microphysics1Moment(), _unwrapped(cmp), _unwrapped(thp), ρ, T, w,
        q_tot_nonneg, q_lcl, q_icl, q_rai, q_sno, dt, nsubs,
    )
    return local_tendency
end
@inline function microphysics_tendencies_1m( #microphysics_tendencies_quadrature_1m
    scheme, sgs_quad, cmp, thp, ρ, T, w, q_tot_nonneg,
    q_lcl, q_icl, q_rai, q_sno, T′T′, q′q′, corr_Tq,
    λ_lagrange, α, ξ_liq, ξ_ice, dt, nsubs,
    # `λ` (liquid fraction) and `mu_S` (linearized SGS saturation-excess mean) are
    # cell constants. They default to being computed here from the mean state; a
    # caller may precompute them once and pass them in.
    λ = TD.liquid_fraction(
        _unwrapped(thp), T, max(zero(ρ), q_lcl), max(zero(ρ), q_icl),
    ),
    mu_S = q_tot_nonneg - TD.q_vap_saturation(_unwrapped(thp), T, ρ),
    precip_incloud_fraction = zero(ρ),
    precip_frac = one(ρ),
    sigma_S = _sgs_linear_sigma_S(thp, ρ, T, T′T′, q′q′, corr_Tq),
    precip_frac_floor = zero(ρ),
    λ_eq = max(zero(ρ), mu_S - (max(zero(ρ), q_rai) + max(zero(ρ), q_sno))),
    args...,
)
    FT = typeof(ρ)
    not_quadrature(sgs_quad) && return microphysics_tendencies_1m(
        ρ, q_tot_nonneg, q_lcl, q_icl, q_rai, q_sno, T, w, cmp, thp, dt, nsubs,
    )
    # Shaft threshold on the centred excess: the moistest `a_p` of the PDF. The
    # shaft is never taken narrower than `precip_frac_floor` of the cell
    # (`sgs_precip_fraction_floor`).
    a_p = max(FT(precip_frac), FT(precip_frac_floor))
    σ_S = FT(sigma_S)
    S_star = sgs_precip_shaft_threshold(a_p, σ_S)
    return _sum_1m_over_sgs(
        scheme, sgs_quad, cmp, thp, ρ, T, w,
        max(zero(ρ), q_lcl), max(zero(ρ), q_icl),
        max(zero(ρ), q_rai), max(zero(ρ), q_sno),
        T′T′, q′q′, corr_Tq, λ_lagrange, α, FT(ξ_liq), FT(ξ_ice), dt, nsubs, λ,
        mu_S, FT(precip_incloud_fraction), σ_S, S_star, FT(λ_eq), args,
    )
end

"""
    _sum_1m_over_sgs(scheme, sgs_quad, cmp, thp, ρ, T, w,
                     q_lcl, q_icl, q_rai, q_sno, T′T′, q′q′, corr_Tq,
                     λ_lagrange, α, ξ_liq, ξ_ice, dt, nsubs, λ, mu_S,
                     β_precip, σ_S, S_star, λ_eq, args,
                     placement = ShaftPrecipPlacement())

Integrate the `Microphysics1MEvaluator` over the Gaussian SGS PDF with the
conditional-moment node rule (the arguments are those of
`microphysics_tendencies_1m`, with the condensate and precipitation clamped
non-negative and the shaft threshold `S_star` set).

The nodes are placed by `conditional_rule` at the conditional moments of `S′`
on the intervals between the cloud edges `−λ_lagrange/α` and `−λ_eq` and the
shaft edge `S_star`, with the temperature regressed on `S′`
(`sgs_temperature_regression`) and split at the freezing point where its
spread at fixed `S′` straddles it (`sum_over_conditional_nodes`). The rule
integrates the piecewise-linear condensates exactly; the multiplier and slope
of the actual condensate are scaled by the common factor that makes the node
mean of `max(0, λ + α S′)` equal `q_c` to rounding, which leaves the cloud
edge `−λ/α` and the cloud fraction unchanged. The shaft fraction is the node
weight of the shaft from the nodes' own flags
(`conditional_rule_shaft_fraction`), so the placement conserves the cell-mean
precipitation exactly for any `a_p` (the shaft is bounded below by
`precip_frac_floor`). A `LogLinearNodePlacement` `placement` replaces the
shaft (`β_precip` is then `0`); its edge is `S_star` where a tilt is nonzero
(`precip_placement_edge`), and, where the cell holds rain or snow, it is
normalized over the same points (`normalized_placement`).
"""
@inline function _sum_1m_over_sgs(
    scheme, sgs_quad, cmp, thp, ρ, T, w,
    q_lcl, q_icl, q_rai, q_sno, T′T′, q′q′, corr_Tq,
    λ_lagrange, α, ξ_liq, ξ_ice, dt, nsubs, λ, mu_S,
    β_precip, σ_S, S_star, λ_eq, args, placement = ShaftPrecipPlacement(),
)
    (evaluator, rule, spread) = _sgs_1m_rule_and_evaluator(
        scheme, sgs_quad, cmp, thp, ρ, T, w,
        q_lcl, q_icl, q_rai, q_sno, T′T′, q′q′, corr_Tq,
        λ_lagrange, α, ξ_liq, ξ_ice, dt, nsubs, λ, mu_S,
        β_precip, σ_S, S_star, λ_eq, args, placement,
    )
    return sum_over_conditional_nodes(evaluator, rule, spread)
end

"""
    _sgs_1m_rule_and_evaluator(arguments of `_sum_1m_over_sgs`)

The `Microphysics1MEvaluator`, `ConditionalRule` and `TemperatureSpread` that
`_sum_1m_over_sgs` sums over.
"""
@inline function _sgs_1m_rule_and_evaluator(
    scheme, sgs_quad, cmp, thp, ρ, T, w,
    q_lcl, q_icl, q_rai, q_sno, T′T′, q′q′, corr_Tq,
    λ_lagrange, α, ξ_liq, ξ_ice, dt, nsubs, λ, mu_S,
    β_precip, σ_S, S_star, λ_eq, args, placement,
)
    FT = typeof(ρ)
    q_c = q_lcl + q_icl
    α_safe = max(FT(α), ϵ_numerics(FT))
    t_a = -FT(λ_lagrange) / α_safe
    t_e = -FT(λ_eq)
    t_p = precip_placement_edge(placement, β_precip, S_star)
    β_T = sgs_temperature_regression(thp, T, ρ, σ_S, T′T′, q′q′, corr_Tq)
    rule = conditional_rule(
        thp, ρ, T, mu_S, σ_S, β_T, t_a, t_e, t_p, sgs_quad.T_min, sgs_quad.q_max,
    )
    # Scale (λ, α) together so the node mean of the actual condensate is q_c
    # to rounding; the cloud edge −λ/α is unchanged by a common factor.
    g = conditional_rule_mean(NodeCondensate(FT(λ_lagrange), α_safe), rule)
    s = ifelse(
        q_c > zero(FT),
        ifelse(g > zero(FT), q_c / max(g, floatmin(FT)), one(FT)),
        zero(FT),
    )
    shaft_frac = ifelse(
        β_precip > zero(FT),
        conditional_rule_shaft_fraction(rule),
        one(FT),
    )
    spread = TemperatureSpread(
        thp, ρ, mu_S, sgs_temperature_spread_at_fixed_excess(T′T′, β_T, σ_S),
        FT(TD.Parameters.T_freeze(_unwrapped(thp))), sgs_quad.T_min, sgs_quad.q_max,
    )
    # Without rain and snow the placement scales nothing: skip its normalization
    placement_n =
        q_rai + q_sno > zero(FT) ?
        normalized_placement(placement, thp, ρ, mu_S, rule, spread) : placement
    evaluator = Microphysics1MEvaluator(
        scheme, cmp, thp, ρ, w,
        q_rai, q_sno, λ,
        ξ_liq, ξ_ice, q_lcl, q_icl,
        s * FT(λ_lagrange), λ_eq, mu_S, s * α_safe,
        β_precip, shaft_frac, dt, nsubs, args, placement_n,
    )
    return (evaluator, rule, spread)
end

"""
    SGSMicrophysicsOptions{FT}

Scalar options of the 1M quadrature microphysics, built by
`sgs_microphysics_options` and broadcast as a scalar (see that function).
"""
Base.@kwdef struct SGSMicrophysicsOptions{FT}
    ξ_liq::FT
    ξ_ice::FT
    precip_incloud_fraction::FT
    precip_frac_floor::FT
end
Base.broadcastable(o::SGSMicrophysicsOptions) = tuple(o)

"""
    sgs_microphysics_options(params)

The per-run scalar options of the 1M quadrature microphysics as one
`SGSMicrophysicsOptions` (`ξ_liq`, `ξ_ice`, and the
precipitation-placement keys), so the tendency broadcast passes one argument
instead of four: a broadcast with more than ~30 arguments falls off Julia's
specialized `Broadcast._getindex` path and allocates inside the GPU kernel.
Errors if the placement is on (`sgs_precip_incloud_fraction > 0`) with a
non-positive `sgs_precip_fraction_floor`, which would leave the concentration
of the precipitation on the shaft nodes unbounded.
"""
function sgs_microphysics_options(params)
    β_p = CAP.sgs_precip_incloud_fraction(params)
    floor = CAP.sgs_precip_fraction_floor(params)
    # The floor bounds the factor on the shaft nodes' precipitation,
    # φ_p ≤ (1 − β_p) + β_p / floor (`sgs_placement_factor`).
    β_p > 0 && !(floor > 0) &&
        error(
            "sgs_precip_fraction_floor must be positive when \
            sgs_precip_incloud_fraction > 0 (got $(floor)): it bounds the \
            concentration of the precipitation on the shaft nodes.",
        )
    return SGSMicrophysicsOptions(;
        ξ_liq = CAP.sgs_liquid_uniform_fraction(params),
        ξ_ice = CAP.sgs_ice_uniform_fraction(params),
        precip_incloud_fraction = β_p,
        precip_frac_floor = floor,
    )
end

"""
    microphysics_tendencies_1m(
        scheme, sgs_quad, cmp, thp, ρ, T, w, q_tot_nonneg, q_lcl, q_icl, q_rai, q_sno,
        T′T′, q′q′, corr_Tq, sgs_moments, α, dt, nsubs, λ, mu_S, precip_frac,
        opts::SGSMicrophysicsOptions,
    )

Packed form of the quadrature driver: the scalar options come as the
`SGSMicrophysicsOptions` of `sgs_microphysics_options`, the per-cell fields
(`sgs_moments`, `α`, `precip_frac`) and the precomputed `λ`, `mu_S` positionally. Forwards to the positional
form, so the two are identical.
"""
@inline function microphysics_tendencies_1m(
    scheme, sgs_quad, cmp, thp, ρ, T, w, q_tot_nonneg,
    q_lcl, q_icl, q_rai, q_sno, T′T′, q′q′, corr_Tq,
    sgs_moments, α, dt, nsubs, λ, mu_S, precip_frac,
    opts::SGSMicrophysicsOptions,
)
    return microphysics_tendencies_1m(
        scheme, sgs_quad, cmp, thp, ρ, T, w, q_tot_nonneg,
        q_lcl, q_icl, q_rai, q_sno, T′T′, q′q′, corr_Tq,
        sgs_moments.λ_lagrange, α, opts.ξ_liq, opts.ξ_ice, dt, nsubs, λ, mu_S,
        opts.precip_incloud_fraction, precip_frac, sgs_moments.sigma_S,
        opts.precip_frac_floor, sgs_moments.λ_eq,
    )
end

"""
    LogLinearPrecipOptions{FT}

Scalar parameters of the log-linear rain and snow placement
(`sgs_precip_placement: loglinear`), built by `loglinear_precip_options` and
broadcast as a scalar: the rain tilt `rain_tilt` (`κ_r0`), its height slope
and offset (`loglinear_rain_tilt`), the snow tilt `snow_tilt` (`κ_s`), the
intermittency coefficient and minimum (`f = min(1, c max(a_p, a_min))`), and
the snow melt depletion width `ΔT_m` [K].
"""
Base.@kwdef struct LogLinearPrecipOptions{FT}
    rain_tilt::FT
    rain_tilt_height_slope::FT
    rain_tilt_height_offset::FT
    snow_tilt::FT
    intermittency_coeff::FT
    intermittency_min::FT
    melt_depletion_width::FT
end
Base.broadcastable(o::LogLinearPrecipOptions) = tuple(o)

"""
    loglinear_precip_options(params)

The `LogLinearPrecipOptions` of the run's parameters. Errors unless the
intermittency coefficient and minimum and the melt depletion width are
positive (the width may be `Inf`).
"""
function loglinear_precip_options(params)
    o = LogLinearPrecipOptions(;
        rain_tilt = CAP.sgs_precip_rain_tilt(params),
        rain_tilt_height_slope = CAP.sgs_precip_rain_tilt_height_slope(params),
        rain_tilt_height_offset = CAP.sgs_precip_rain_tilt_height_offset(params),
        snow_tilt = CAP.sgs_precip_snow_tilt(params),
        intermittency_coeff = CAP.sgs_precip_intermittency_coeff(params),
        intermittency_min = CAP.sgs_precip_intermittency_min(params),
        melt_depletion_width = CAP.sgs_snow_melt_depletion_width(params),
    )
    (o.intermittency_coeff > 0 && o.intermittency_min > 0) || error(
        "sgs_precip_intermittency_coeff and sgs_precip_intermittency_min must \
        be positive (got $(o.intermittency_coeff), $(o.intermittency_min)).",
    )
    o.melt_depletion_width > 0 || error(
        "sgs_snow_melt_depletion_width must be positive or Inf (got \
        $(o.melt_depletion_width)).",
    )
    return o
end

"""
    loglinear_rain_tilt(o::LogLinearPrecipOptions, z_agl)

Rain tilt `κ_r = κ_r0 + (dκ/dz) max(0, z_agl − z_1)` at the height `z_agl` [m]
above the surface.
"""
@inline loglinear_rain_tilt(o::LogLinearPrecipOptions, z_agl) =
    o.rain_tilt +
    o.rain_tilt_height_slope * max(zero(z_agl), z_agl - o.rain_tilt_height_offset)

"""
    loglinear_node_placement(thp, T, q_tot_nonneg, mu_S, σ_S, precip_frac, κ_r, ll)

The (unnormalized) `LogLinearNodePlacement` of a cell with mean temperature
`T` [K], total water `q_tot_nonneg` and mean excess `mu_S` [kg/kg], width
`σ_S` [kg/kg], precipitation fraction `a_p = precip_frac` and rain tilt `κ_r`,
and the bounded fraction `max(a_p, a_min)`, with the scalars of the
`LogLinearPrecipOptions` `ll`.
"""
@inline function loglinear_node_placement(
    thp, T, q_tot_nonneg, mu_S, σ_S, precip_frac, κ_r, ll,
)
    FT = typeof(σ_S)
    a_p = max(FT(precip_frac), FT(ll.intermittency_min))
    T_freeze = FT(TD.Parameters.T_freeze(_unwrapped(thp)))
    # Below about 256 ulps of the water contents, the S′ that the evaluator
    # recovers from a point's state is rounding, not displacement.
    σ_z = max(σ_S, 256 * eps(FT) * (abs(FT(q_tot_nonneg)) + abs(FT(mu_S))))
    placement = LogLinearNodePlacement(
        FT(κ_r), FT(ll.snow_tilt), one(FT) / FT(ll.melt_depletion_width),
        T_freeze, max(FT(T), T_freeze), σ_z,
        min(one(FT), FT(ll.intermittency_coeff) * a_p), one(FT), one(FT),
    )
    return (placement, a_p)
end

"""
    microphysics_tendencies_1m(
        scheme, sgs_quad, cmp, thp, ρ, T, w, q_tot_nonneg, q_lcl, q_icl, q_rai, q_sno,
        T′T′, q′q′, corr_Tq, sgs_moments, α, dt, nsubs, λ, mu_S, precip_frac, κ_r,
        opts::SGSMicrophysicsOptions, ll::LogLinearPrecipOptions,
    )

Packed form with the log-linear rain and snow placement
(`LogLinearNodePlacement`) in place of the shaft: the cell's rain tilt `κ_r`
(`loglinear_rain_tilt`) and the scalars of `ll`; `opts.ξ_liq` and
`opts.ξ_ice` as in the other packed form, whose shaft keys are not used. The
precipitating fraction is `f = min(1, c max(a_p, a_min))` with
`a_p = precip_frac`, and the rule gets the edge `S_star` of `max(a_p, a_min)`
(`precip_placement_edge`) where the cell holds rain or snow; without them the
result is that of the uniform placement.
"""
@inline function microphysics_tendencies_1m(
    scheme, sgs_quad, cmp, thp, ρ, T, w, q_tot_nonneg,
    q_lcl, q_icl, q_rai, q_sno, T′T′, q′q′, corr_Tq,
    sgs_moments, α, dt, nsubs, λ, mu_S, precip_frac, κ_r,
    opts::SGSMicrophysicsOptions, ll::LogLinearPrecipOptions,
)
    FT = typeof(ρ)
    not_quadrature(sgs_quad) && return microphysics_tendencies_1m(
        ρ, q_tot_nonneg, q_lcl, q_icl, q_rai, q_sno, T, w, cmp, thp, dt, nsubs,
    )
    σ_S = FT(sgs_moments.sigma_S)
    (placement, a_p) = loglinear_node_placement(
        thp, T, q_tot_nonneg, mu_S, σ_S, precip_frac, κ_r, ll,
    )
    q_rai_c = max(zero(ρ), q_rai)
    q_sno_c = max(zero(ρ), q_sno)
    # The edge resolves the rain and snow placement: none without them, so a
    # cell without rain and snow gets the rule of the uniform placement.
    S_star = ifelse(
        q_rai_c + q_sno_c > zero(FT),
        sgs_precip_shaft_threshold(a_p, σ_S),
        FT(Inf),
    )
    return _sum_1m_over_sgs(
        scheme, sgs_quad, cmp, thp, ρ, T, w,
        max(zero(ρ), q_lcl), max(zero(ρ), q_icl), q_rai_c, q_sno_c,
        T′T′, q′q′, corr_Tq, sgs_moments.λ_lagrange, α,
        FT(opts.ξ_liq), FT(opts.ξ_ice), dt, nsubs, λ, mu_S, zero(FT), σ_S,
        S_star, FT(sgs_moments.λ_eq), (), placement,
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
