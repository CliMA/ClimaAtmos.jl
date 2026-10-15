import ClimaComms
import Thermodynamics as TD
import CloudMicrophysics.Parameters as CMP
import CloudMicrophysics.BulkMicrophysicsTendencies as BMT
import CloudMicrophysics.Microphysics0M as CM0
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

"""
    precipitation_source_liquid_fraction(thp, ρ_dq_tot_dt, ρ_de_tot_dt, T, Φ)

Liquid fraction `λ` of the 0-moment precipitation produced in a cell, recovered
from the density-weighted sinks through
`ρ_de_tot_dt / ρ_dq_tot_dt = λ I_liq(T) + (1 - λ) I_ice(T) + Φ`
(see `e_tot_0M_precipitation_sources_helper`). Exact for a single subdomain;
for EDMF or SGS-quadrature sums it is the mass-weighted liquid fraction, up to
the spread of subdomain temperatures about `T`. Clamped to `[0, 1]`; `1` where
nothing is produced.
"""
@inline function precipitation_source_liquid_fraction(
    thp, ρ_dq_tot_dt, ρ_de_tot_dt, T, Φ,
)
    FT = typeof(T)
    Iₗ = TD.internal_energy_liquid(thp, T)
    Iᵢ = TD.internal_energy_ice(thp, T)
    has_source = !iszero(ρ_dq_tot_dt)
    e = ρ_de_tot_dt / ifelse(has_source, ρ_dq_tot_dt, one(FT))
    λ = clamp((e - Φ - Iᵢ) / (Iₗ - Iᵢ), zero(FT), one(FT))
    return ifelse(has_source, λ, one(FT))
end

"""
    wet_bulb_vapor_deficit(thp, RH_c, q_sat, q_vap, L, T)

Vapor [kg/kg] that evaporation into air at temperature `T` can add before the
latent cooling (latent heat `L`) brings the air to `RH_c` times saturation,
where `q_sat` is the saturation specific humidity over the evaporating phase:
`max(RH_c q_sat - q_vap, 0) / (1 + RH_c L² q_sat / (c_p R_v T²))`.
"""
@inline function wet_bulb_vapor_deficit(thp, RH_c, q_sat, q_vap, L, T)
    cp_d = TD.Parameters.cp_d(thp)
    R_v = TD.Parameters.R_v(thp)
    return max(RH_c * q_sat - q_vap, zero(q_sat)) /
           (1 + RH_c * L^2 * q_sat / (cp_d * R_v * T^2))
end

"""
    precipitation_evaporation_step(evaporation, thp, dt, state, input)

One level of the top-down scan (`Operators.column_accumulate!` with
`reverse = true`) of the 0-moment precipitation flux with below-cloud
evaporation of rain and sublimation of snow.

`state = (F_r, F_s, F_rl, F_sl, ρE_r, ρE_s, ρe_r, ρe_s)` holds the rain and
snow fluxes [kg m⁻² s⁻¹] entering the cell from above (on output: leaving it at
the bottom), the parts `F_rl ≤ F_r`, `F_sl ≤ F_s` of them that were removed
from the air as liquid, and the evaporation rates [kg m⁻³ s⁻¹] and their
energies [W m⁻³] of the cell above. `input` is the tuple
`(S_r, S_s, λ_src, Δz, ρ_env, ρa_env, T, p_over_ps, q_vap, Φ)`: the rain and
snow produced in the cell [kg m⁻³ s⁻¹], their liquid fraction (see
`precipitation_source_liquid_fraction`), the column-integration layer depth,
and the density, mass per unit volume of the grid box, temperature, vapor
specific humidity and geopotential of the air into which precipitation
evaporates.

The incoming fluxes evaporate at `ρa_env` times
`CloudMicrophysics.Microphysics0M.evaporate_precipitation`, rain with the
deficit over liquid and snow with the deficit over ice (over liquid above
freezing). Each rate is limited by the flux falling into the cell and by the
wet-bulb vapor deficit of its phase (`wet_bulb_vapor_deficit`, over liquid with
`L_v` for rain and over ice with `L_s` for snow below freezing); their sum is
limited by the larger of the two wet-bulb deficits.

The evaporated water carries the specific energy of the phase it was removed
from the air with, `χ I_liq(T) + (1 - χ) I_ice(T) + Φ` with `χ = F_rl / F_r` for
rain and `F_sl / F_s` for snow.
"""
@inline function precipitation_evaporation_step(
    evaporation, thp, dt, state, input,
)
    (F_r, F_s, F_rl, F_sl) = state
    (S_r, S_s, λ_src, Δz, ρ_env, ρa_env, T, p_over_ps, q_vap, Φ) = input
    (; RH_c) = evaporation
    FT = typeof(F_r)
    ρa = max(ρa_env, zero(FT))
    q_sat_liq = TD.q_vap_saturation(thp, T, ρ_env, TD.Liquid())
    # Snow uses ice saturation below freezing and liquid saturation above it
    # (where the extrapolated ice curve exceeds the liquid one).
    q_sat_ice = TD.q_vap_saturation(thp, T, ρ_env, TD.Ice())
    snow_over_ice = q_sat_ice < q_sat_liq
    q_sat_sno = ifelse(snow_over_ice, q_sat_ice, q_sat_liq)
    L_sno = ifelse(
        snow_over_ice, TD.latent_heat_sublim(thp, T), TD.latent_heat_vapor(thp, T),
    )
    E_r = CM0.evaporate_precipitation(
        evaporation, RH_c * q_sat_liq - q_vap, p_over_ps, F_r,
    )
    E_s = CM0.evaporate_precipitation(
        evaporation, RH_c * q_sat_sno - q_vap, p_over_ps, F_s,
    )
    Δq_wb_liq = wet_bulb_vapor_deficit(
        thp, RH_c, q_sat_liq, q_vap, TD.latent_heat_vapor(thp, T), T,
    )
    Δq_wb_sno = wet_bulb_vapor_deficit(thp, RH_c, q_sat_sno, q_vap, L_sno, T)
    ρE_r = min(ρa * E_r, F_r / Δz, ρa * Δq_wb_liq / dt)
    ρE_s = min(ρa * E_s, F_s / Δz, ρa * Δq_wb_sno / dt)
    ρE_max = ρa * max(Δq_wb_liq, Δq_wb_sno) / dt
    ρE = ρE_r + ρE_s
    scale = ifelse(ρE > ρE_max, ρE_max / ρE, one(FT))
    ρE_r *= scale
    ρE_s *= scale
    # Liquid fractions of the incoming fluxes (irrelevant where F = 0).
    χ_r = clamp(F_rl / max(F_r, floatmin(FT)), zero(FT), one(FT))
    χ_s = clamp(F_sl / max(F_s, floatmin(FT)), zero(FT), one(FT))
    # Fluxes leaving the cell at its bottom face, kept non-negative against
    # roundoff in `F - (F / Δz) Δz`.
    F_r_out = max(F_r + (S_r - ρE_r) * Δz, zero(FT))
    F_s_out = max(F_s + (S_s - ρE_s) * Δz, zero(FT))
    F_rl_out = clamp(F_rl + (λ_src * S_r - χ_r * ρE_r) * Δz, zero(FT), F_r_out)
    F_sl_out = clamp(F_sl + (λ_src * S_s - χ_s * ρE_s) * Δz, zero(FT), F_s_out)
    Iₗ = TD.internal_energy_liquid(thp, T)
    Iᵢ = TD.internal_energy_ice(thp, T)
    ρe_r = ρE_r * (χ_r * Iₗ + (1 - χ_r) * Iᵢ + Φ)
    ρe_s = ρE_s * (χ_s * Iₗ + (1 - χ_s) * Iᵢ + Φ)
    return (F_r_out, F_s_out, F_rl_out, F_sl_out, ρE_r, ρE_s, ρe_r, ρe_s)
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
    Microphysics1MEvaluator{S, MP, TPS, FT, Args}

GPU-safe functor evaluating 1-moment microphysics tendencies at SGS quadrature
points, for use with [`integrate_over_sgs`](@ref).

The local condensate at each point follows the truncated-Gaussian
Lagrange-multiplier closure described in `microphysics_tendencies_1m`.
Precipitation (`q_rai`, `q_sno`), the liquid fraction `λ`, and the closure
quantities (`λ_lagrange`, `λ_eq`, `mu_S`, `α`) are grid-cell constants held fixed across
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
  - `λ_eq`: Equilibrium Lagrange multiplier enforcing
    `E[max(0, λ_eq + S′)] = q_c_eq` (fitted with `α = 1`), from which the
    node saturation excess of cloudy equilibrium nodes is reconstructed [kg/kg].
  - `mu_S`: Linearized SGS mean saturation excess `q_tot − q_sat(T, ρ)` [kg/kg].
  - `α`: Variance fidelity parameter [-].
  - `β_precip`, `shaft_frac`: Precipitation-fraction placement of the cell-mean
    rain and snow onto the shaft nodes (`SGSPrecipShaftFlag`) and the discrete
    fraction of the quadrature measure in the shaft (the quadrature's own
    `a_p`); `β_precip = 0` disables it.
  - `S_star`, `ε_S`: Threshold and width of the shaft weight on the centred
    saturation excess (`sgs_precip_shaft_weight`), set by the overlap
    precipitation fraction through `sgs_precip_shaft_threshold`.
  - `dt`: Timestep used for the time-averaged process rates [s].
  - `nsubs`: Number of substeps in the tendency averaging.
  - `args`: Extra trailing arguments forwarded to the CloudMicrophysics call.
"""
struct Microphysics1MEvaluator{S, MP, TPS, FT, Args <: Tuple}
    scheme::S
    mp::MP
    tps::TPS
    ρ::FT
    w::FT              # physical vertical velocity [m/s]
    # Precipitation
    q_rai::FT
    q_sno::FT
    λ::FT              # liquid fraction (from thermodynamics, held fixed)
    # Uniform fractions of each species over the quadrature and the subdomain
    # means they draw on
    ξ_liq::FT
    ξ_ice::FT
    q_lcl::FT
    q_icl::FT
    # Truncated-Gaussian Lagrange multipliers, μ_S, and liquid fraction
    λ_lagrange::FT # Lagrange multiplier for actual condensate q_c (discrete fit)
    λ_eq::FT       # equilibrium multiplier for q_c_eq (discrete fit, α = 1)
    mu_S::FT       # linearized SGS mean μ_S = q_tot_mean − q_sat(T_mean, ρ)
    α::FT          # variance fidelity parameter (from sgs_variance_fidelity)
    # Precipitation-fraction placement
    β_precip::FT
    shaft_frac::FT
    S_star::FT
    ε_S::FT
    # Numerical parameters
    dt::FT
    nsubs::Int
    args::Args
end

@inline Microphysics1MEvaluator(
    scheme, mp, tps, ρ, w,
    q_rai, q_sno, λ,
    ξ_liq, ξ_ice, q_lcl, q_icl,
    λ_lagrange, mu_S, α,
    β_precip, shaft_frac, S_star, ε_S,
    dt, nsubs, args,
) = Microphysics1MEvaluator(
    scheme, mp, tps, ρ, w,
    q_rai, q_sno, λ,
    ξ_liq, ξ_ice, q_lcl, q_icl,
    λ_lagrange, max(zero(ρ), mu_S - (q_rai + q_sno)), mu_S, α,
    β_precip, shaft_frac, S_star, ε_S,
    dt, nsubs, args,
)

"""
    sgs_precip_shaft_weight(S′, S_star, ε_S)

Weight in `[0, 1]` with which a quadrature node of centred saturation excess
`S′` belongs to the precipitation shaft: the nodes with `S′ ≥ S_star`, smoothed
over the width `ε_S`,

    s = sigmoid((S′ − S_star) / ε_S)      for ε_S > 0
    s = 1 if S′ ≥ S_star, else 0           for ε_S = 0 (the hard flag)

Its quadrature mean `shaft_frac` is the fraction of
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

Width of the shaft weight in units of the sampled PDF width `σ_S` under the
overlap precipitation fraction: `0.25` keeps the transition narrow compared with
the Gauss–Hermite node spacing (≈ 1.7 σ_S at order 3) while smoothing the node
assignment as `a_p` changes between steps.
"""
@inline sgs_precip_shaft_width_coeff(::Type{FT}) where {FT} = FT(0.25)

"""
    sgs_precip_shaft_frac_min(FT)

Smallest discrete shaft fraction the placement resolves; below it the
quadrature cannot represent the shaft and the precipitation is left uniform.
"""
@inline sgs_precip_shaft_frac_min(::Type{FT}) where {FT} = FT(0.02)

"""
    sgs_precip_shaft_threshold(a_p, sigma_S)

Threshold `S_star` on the centred saturation excess such that the shaft covers
the moistest fraction `a_p` of the sampled PDF, taken Gaussian with standard
deviation `sigma_S` (the quadrature's own `Σᵢ wᵢ S′ᵢ²`):

    S_star = σ_S · Φ⁻¹(1 − a_p)

`a_p` is clamped to `[eps, 1]`: `a_p = 1` places the threshold far below every
node (uniform precipitation), `a_p → 0` far above every node (no node in the
shaft; the driver then leaves the precipitation uniform, see
`sgs_precip_shaft_frac_min`). The discrete weight of the nodes above the
threshold normalizes the placement.
"""
@inline function sgs_precip_shaft_threshold(a_p, sigma_S)
    FT = typeof(sigma_S)
    a = clamp(FT(a_p), eps(FT), one(FT))
    return sigma_S * normal_cdf_inv(one(FT) - a)
end

"""
    SGSPrecipShaftFlag(tps, ρ, mu_S, S_star, ε_S)

Point-wise functor for the SGS quadrature returning the shaft weight
(`sgs_precip_shaft_weight`) of a node from its centred saturation excess
`S′ = q_tot_hat − q_sat(T_hat, ρ) − mu_S`. Its quadrature mean `shaft_frac` is
the weight of the part of the cell the precipitation is taken to fall through
under the precipitation-fraction placement (`Microphysics1MEvaluator`).
"""
struct SGSPrecipShaftFlag{TPS, FT}
    tps::TPS
    ρ::FT
    mu_S::FT
    S_star::FT
    ε_S::FT
end
@inline function (f::SGSPrecipShaftFlag)(T_hat, q_tot_hat)
    FT = typeof(f.ρ)
    S′ =
        max(zero(FT), q_tot_hat) -
        TD.q_vap_saturation(_unwrapped(f.tps), T_hat, f.ρ) - f.mu_S
    return sgs_precip_shaft_weight(S′, f.S_star, f.ε_S)
end

"""
    sgs_placement_factor(β, flag, shaft_frac)

Factor `φ` on a cell-mean quantity at a quadrature node when a fraction `β` of it
is confined to the flagged nodes (`flag = 1`, total quadrature weight
`shaft_frac`) and the rest stays uniform,

    φ = (1 − β) + β · flag / shaft_frac,

so that the quadrature mean of `φ` is 1 and the cell mean is conserved. `β = 0`
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
    q_sat_hat = TD.q_vap_saturation(_unwrapped(eval.tps), T_hat, eval.ρ)
    S′_hat = q_tot_hat - q_sat_hat - eval.mu_S
    αS′_hat = eval.α * S′_hat
    shifted_excess = max(FT(0), eval.λ_lagrange + αS′_hat)
    q_lcl_hat, q_icl_hat = sgs_local_condensate(
        eval.λ, shifted_excess, eval.ξ_liq, eval.ξ_ice, eval.q_lcl, eval.q_icl,
    )
    # Reconstruct the node saturation excess on cloudy equilibrium nodes from
    # `λ_eq` so that condensation and evaporation relax toward the continuous
    # truncated-Gaussian equilibrium `q_c_eq` without the discrete 3-point
    # Gauss–Hermite support cutoff at `μ_S < −√3 σ_S`, while preserving raw
    # subsaturation on clear nodes (`q_c_eq_hat == 0`) for below-cloud rain
    # evaporation and snow sublimation. The equilibrium node uses the
    # unattenuated fluctuation `S′_hat`: `α` attenuates only the actual
    # condensate, which lags equilibrium, and `λ_eq` is fitted with `α = 1`.
    q_c_eq_hat = max(FT(0), eval.λ_eq + S′_hat)
    q_precip = eval.q_rai + eval.q_sno
    S_raw_cloud = q_tot_hat - q_precip - q_sat_hat
    S_eff_cloud = ifelse(
        q_c_eq_hat > zero(FT),
        q_c_eq_hat,
        min(zero(FT), S_raw_cloud),
    )
    q_tot_eff = max(FT(0), q_sat_hat + S_eff_cloud + q_precip)

    # Precipitation-fraction placement: the rain and snow this node sees are
    # the cell means scaled by φ_p (1 when disabled); the node total water is
    # shifted by the same amount so that the node vapour, and hence the
    # condensate reconstruction, is unchanged by the placement.
    q_rai_node, q_sno_node, q_tot_node =
        if eval.β_precip > zero(FT)
            flag_p = sgs_precip_shaft_weight(S′_hat, eval.S_star, eval.ε_S)
            φ_p = sgs_placement_factor(eval.β_precip, flag_p, eval.shaft_frac)
            shift = (φ_p - one(FT)) * q_precip
            (eval.q_rai * φ_p, eval.q_sno * φ_p, max(FT(0), q_tot_eff + shift))
        else
            (eval.q_rai, eval.q_sno, q_tot_eff)
        end

    return BMT.bulk_microphysics_tendencies(
        BMT.LinearizedAverage(),
        eval.scheme, _unwrapped(eval.mp), _unwrapped(eval.tps),
        eval.ρ, T_hat, eval.w,
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
        precip_incloud_fraction = 0, precip_frac = 1, sigma_S = 0,
        precip_frac_floor = 0, λ_eq = max(0, mu_S − q_rai − q_sno), args...,
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
  - `precip_incloud_fraction`: Fraction `β_p` of the cell-mean rain and snow confined
    to the precipitation shaft of the PDF; `0` disables the placement [-].
  - `precip_frac`: Overlap precipitation fraction `a_p` of the cell [-].
  - `sigma_S`: SGS saturation-excess standard deviation of the sampled PDF
    (`ᶜsgs_moments.sigma_S`), which scales the shaft threshold and width [kg/kg].
  - `precip_frac_floor`: the shaft is never taken narrower than this fraction of
    the cell; defaults to `0` [-].
  - `λ_eq`: Equilibrium Lagrange multiplier from `ᶜsgs_moments`, fitted with
    `α = 1` so that `E[max(0, λ_eq + S′)] = q_c_eq` under the quadrature
    measure [kg/kg].
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
    # invariant across the quadrature. They default to being computed here from the
    # mean state; a caller evaluating this broadcast over many quadrature points can
    # precompute them once and pass them in to avoid recomputing them per point.
    λ = TD.liquid_fraction(
        _unwrapped(thp), T, max(zero(ρ), q_lcl), max(zero(ρ), q_icl),
    ),
    mu_S = q_tot_nonneg - TD.q_vap_saturation(_unwrapped(thp), T, ρ),
    precip_incloud_fraction = zero(ρ),
    precip_frac = one(ρ),
    sigma_S = zero(ρ),
    precip_frac_floor = zero(ρ),
    λ_eq = max(zero(ρ), mu_S - (max(zero(ρ), q_rai) + max(zero(ρ), q_sno))),
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
    # Shaft weight on the nodes: the moistest `a_p` of the PDF, smoothed over
    # `c_w σ_S`. Its quadrature fraction `shaft_frac` normalizes the placement.
    # The shaft is never taken narrower than `precip_frac_floor` of the cell
    # (`sgs_precip_fraction_floor`).
    a_p = max(FT(precip_frac), FT(precip_frac_floor))
    # A caller that passes no `sigma_S` gets the narrowest resolvable PDF.
    σ_S = max(FT(sigma_S), ϵ_numerics(FT))
    S_star = sgs_precip_shaft_threshold(a_p, σ_S)
    ε_S = sgs_precip_shaft_width_coeff(FT) * σ_S
    shaft_frac = if precip_incloud_fraction > zero(FT)
        frac = sum_over_quadrature_points(
            SGSPrecipShaftFlag(thp, ρ, mu_S, S_star, ε_S), transform, sgs_quad,
        )
        # A shaft the quadrature cannot resolve is left uniform.
        ifelse(frac < sgs_precip_shaft_frac_min(FT), zero(FT), frac)
    else
        one(FT)
    end
    evaluator = Microphysics1MEvaluator(
        scheme, cmp, thp, ρ, w,
        q_rai_nonneg, q_sno_nonneg, λ,
        FT(ξ_liq), FT(ξ_ice), max(zero(ρ), q_lcl), max(zero(ρ), q_icl),
        λ_lagrange, FT(λ_eq), mu_S, α,
        FT(precip_incloud_fraction), FT(shaft_frac), S_star, ε_S, dt, nsubs, args,
    )
    return sum_over_quadrature_points(evaluator, transform, sgs_quad)
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
"""
function sgs_microphysics_options(params)
    return SGSMicrophysicsOptions(;
        ξ_liq = CAP.sgs_liquid_uniform_fraction(params),
        ξ_ice = CAP.sgs_ice_uniform_fraction(params),
        precip_incloud_fraction = CAP.sgs_precip_incloud_fraction(params),
        precip_frac_floor = CAP.sgs_precip_fraction_floor(params),
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
