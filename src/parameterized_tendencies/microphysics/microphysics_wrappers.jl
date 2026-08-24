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
  - `q_rai`, `q_sno`: Rain and snow specific humidity [kg/kg], clamped
    non-negative by the caller.
  - `λ`: Thermodynamic liquid fraction [-].
  - `λ_lagrange`: Lagrange multiplier enforcing
    `E[max(0, λ_lagrange + α·S′)] = q_c` under the quadrature
    measure (fitted in `_compute_sgs_moments`) [kg/kg].
  - `mu_S`: Linearized SGS mean saturation excess `q_tot − q_sat(T, ρ)` [kg/kg].
  - `α`: Variance fidelity parameter [-].
  - `dt`: Timestep used for the time-averaged process rates [s].
  - `nsubs`: Number of substeps in the tendency averaging.
  - `args`: Extra trailing arguments forwarded to the CloudMicrophysics call.
"""
struct Microphysics1MEvaluator{S, MP, TPS, FT, Args <: Tuple}
    scheme::S
    mp::MP
    tps::TPS
    ρ::FT
    # Precipitation (held fixed across quadrature points)
    q_rai::FT
    q_sno::FT
    # Truncated-Gaussian Lagrange multiplier, μ_S, and liquid fraction
    λ::FT              # liquid fraction (from thermodynamics, held fixed)
    λ_lagrange::FT # Lagrange multiplier for centred S′ (discrete fit)
    mu_S::FT       # linearized SGS mean μ_S = q_tot_mean − q_sat(T_mean, ρ)
    α::FT          # variance fidelity parameter (from sgs_variance_fidelity)
    # Numerical parameters
    dt::FT
    nsubs::Int
    args::Args
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
    q_lcl_hat      = λ · shifted_excess
    q_icl_hat      = (1 − λ) · shifted_excess

The Lagrange multiplier `λ_lagrange` is fitted (in `_compute_sgs_moments`) so
that `E[shifted_excess] = q_c`, where `q_c = q_lcl + q_icl` is the grid-mean
*cloud* condensate, excluding precipitation. The reconstruction therefore
partitions `shifted_excess` into local cloud liquid and ice by the liquid
fraction. Precipitation is held constant across quadrature points and is
accounted for downstream, where CloudMicrophysics subtracts it from `q_tot`
to diagnose the local vapor.

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
    shifted_excess = max(FT(0), eval.λ_lagrange + eval.α * S′_hat)
    q_lcl_hat = eval.λ * shifted_excess
    q_icl_hat = (FT(1) - eval.λ) * shifted_excess

    # Nothing to do at this quadrature point: subsaturated, no condensate, no
    # precipitation. Then every 1-moment process is identically zero --
    # autoconversion and accretion need condensate, evaporation needs rain,
    # melting and sublimation need snow, and condensation needs supersaturation.
    #
    # The subsaturation test is load-bearing and was missing from the first
    # version of this guard. Supersaturated air with no condensate still forms
    # cloud, so without it this skips real condensation: a sweep found 616 of
    # 2592 states returning nonzero, every one of them supersaturated. With it,
    # 2272 subsaturated states all return exactly zero, so the skip is
    # bit-for-bit rather than an approximation.
    #
    # Worth the branch because the call being skipped is essentially the entire
    # cost of this kernel, and it is evaluated at N^2 quadrature points in every
    # cell -- while most of an AMIP column is clear and precipitation-free. The
    # saving is per warp rather than per lane, but clear air comes in
    # contiguous blocks, so whole warps skip together.
    #
    # The returned NamedTuple must keep the same fields and types as the call
    # below, or the quadrature accumulator becomes type-unstable.
    if q_tot_hat <= q_sat_hat &&
       iszero(q_lcl_hat) &&
       iszero(q_icl_hat) &&
       iszero(eval.q_rai) &&
       iszero(eval.q_sno)
        z = zero(FT)
        return (; dq_lcl_dt = z, dq_icl_dt = z, dq_rai_dt = z, dq_sno_dt = z)
    end

    return BMT.bulk_microphysics_tendencies(
        BMT.LinearizedAverage(),
        eval.scheme, eval.mp, eval.tps, eval.ρ, T_hat, q_tot_hat,
        q_lcl_hat, q_icl_hat, eval.q_rai, eval.q_sno,
        eval.dt, eval.nsubs, eval.args...,
    )
end

"""
    microphysics_tendencies_1m(
        ρ, q_tot_nonneg, q_lcl, q_icl, q_rai, q_sno, T, cmp, thp, dt, nsubs,
    )
    microphysics_tendencies_1m(
        scheme, sgs_quad, cmp, thp, ρ, T, q_tot_nonneg,
        q_lcl, q_icl, q_rai, q_sno, T′T′, q′q′, corr_Tq,
        λ_lagrange, α, dt, nsubs, λ = ..., mu_S = ..., args...,
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
  - `q_tot_nonneg`: Total water specific humidity, clamped non-negative [kg/kg].
  - `q_lcl`, `q_icl`: Cloud liquid and cloud ice specific humidity [kg/kg].
  - `q_rai`, `q_sno`: Rain and snow specific humidity [kg/kg].
  - `T′T′`: Temperature variance ``\\langle T'^2 \\rangle`` [K²].
  - `q′q′`: Total-water variance ``\\langle q'^2 \\rangle`` [(kg/kg)²].
  - `corr_Tq`: Correlation coefficient corr(T′, q′) from `correlation_Tq(params)` [-].
  - `λ_lagrange`: Lagrange multiplier from `ᶜsgs_moments`, precomputed to
    enforce `E[max(0, λ_lagrange + α·S′)] = q_c` exactly under the
    quadrature measure [kg/kg].
  - `α`: Variance fidelity parameter from `sgs_variance_fidelity` [-].
  - `dt`: Timestep [s].
  - `nsubs`: Number of substeps for tendency averaging.
  - `λ`: Liquid fraction [-]; defaults to `TD.liquid_fraction` at the mean state.
  - `mu_S`: Linearized SGS mean saturation excess [kg/kg]; defaults to
    `q_tot_nonneg − q_sat(T, ρ)`. Both are quadrature invariants and may be
    precomputed by the caller to avoid recomputing them at every point.
  - `args...`: Extra trailing arguments forwarded to CloudMicrophysics.

# Returns

NamedTuple with `dq_lcl_dt`, `dq_icl_dt`, `dq_rai_dt`, `dq_sno_dt` [kg/kg/s],
positive when a source of the corresponding tracer.
"""
@inline function microphysics_tendencies_1m( #compute_1m_precipitation_tendencies!(
    ρ, q_tot_nonneg, q_lcl, q_icl, q_rai, q_sno, T, cmp, thp, dt, nsubs,
)
    local_tendency = BMT.bulk_microphysics_tendencies(
        BMT.LinearizedAverage(),
        BMT.Microphysics1Moment(), cmp, thp, ρ, T,
        q_tot_nonneg, q_lcl, q_icl, q_rai, q_sno, dt, nsubs,
    )
    return local_tendency
end
@inline function microphysics_tendencies_1m( #microphysics_tendencies_quadrature_1m
    scheme, sgs_quad, cmp, thp, ρ, T, q_tot_nonneg,
    q_lcl, q_icl, q_rai, q_sno, T′T′, q′q′, corr_Tq,
    λ_lagrange, α, dt, nsubs,
    # `λ` (liquid fraction) and `mu_S` (linearized SGS saturation-excess mean) are
    # invariant across the quadrature. They default to being computed here from the
    # mean state; a caller evaluating this broadcast over many quadrature points can
    # precompute them once and pass them in to avoid recomputing them per point.
    λ = TD.liquid_fraction(thp, T, max(zero(ρ), q_lcl), max(zero(ρ), q_icl)),
    mu_S = q_tot_nonneg - TD.q_vap_saturation(thp, T, ρ),
    args...,
)
    FT = typeof(ρ)
    # Clamp specific humidities to non-negative.
    q_rai_nonneg = max(FT(0), q_rai)
    q_sno_nonneg = max(FT(0), q_sno)

    evaluator = Microphysics1MEvaluator(
        scheme, cmp, thp, ρ,
        q_rai_nonneg, q_sno_nonneg,
        λ, λ_lagrange, mu_S, α,
        dt, nsubs, args,
    )
    return integrate_over_sgs(
        evaluator, sgs_quad, q_tot_nonneg, T, q′q′, T′T′, corr_Tq,
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
