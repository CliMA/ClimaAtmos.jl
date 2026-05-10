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
- `thp`: Thermodynamics parameters
- `T`: Air temperature [K]
- `q_liq`: Cloud liquid specific humidity [kg/kg]
- `q_ice`: Cloud ice specific humidity [kg/kg]
- `Φ`: Geopotential energy [J/kg]

# Returns
Energy multiplier [J/kg] computed as:
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
    Microphysics0MEvaluator

GPU-safe functor for computing 0-moment microphysics tendencies at quadrature
points. Delegates condensate diagnosis to `SaturationAdjustmentEvaluator`,
then calls `BMT.bulk_microphysics_tendencies(Microphysics0Moment(), ...)`.

Returns a `NamedTuple` with `dq_tot_dt` and `e_tot_hlpr` so that both fields
are SGS-averaged by `integrate_over_sgs`.

# Fields
- `cm_params`: 0M microphysics parameters
- `sat_eval`: `SaturationAdjustmentEvaluator` for condensate diagnosis
- `Φ`: Geopotential energy [J/kg] (constant within a grid cell)
"""
struct Microphysics0MEvaluator{CMP, SAE, FT}
    cm_params::CMP
    sat_eval::SAE
    Φ::FT
end
function Microphysics0MEvaluator(cm_params, thermo_params, ρ, T_mean, Φ)
    # Grid-mean liquid fraction, held fixed across quadrature points
    # (paper Eq. 3.31, equilibrium fallback). The 0M scheme has no prognostic
    # phase memory, so we use a temperature-based ramp at the grid mean.
    λ_mean = TD.liquid_fraction_ramp(thermo_params, T_mean)
    sat_eval = SaturationAdjustmentEvaluator(thermo_params, ρ, λ_mean)
    return Microphysics0MEvaluator(cm_params, sat_eval, Φ)
end
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
    # Compute energy helper at this quadrature point using the
    # locally-diagnosed condensate, so both fields are SGS-averaged.
    e_tot_hlpr = e_tot_0M_precipitation_sources_helper(
        eval.sat_eval.thermo_params, T_hat, sa.q_liq, sa.q_ice, eval.Φ,
    )
    return (; dq_tot_dt, e_tot_hlpr)
end

"""
    microphysics_tendencies_0m(SG_quad, cmp, thp, ρ, T, q_tot, T′T′, q′q′, corr_Tq, Φ, tst, dt)
    microphysics_tendencies_0m(cmp, thp, ρ, T, q_tot, q_liq, q_ice, Φ, tst, dt)

Computes 0-moment microphysics tendencies.

When using SGS-quadratures, the tendencies are integrated over the joint PDF of (T, q_tot).
At each quadrature point, condensate is diagnosed from saturation excess,
then 0M precipitation removal tendencies are computed.

The option without SGS-quadratures can be used in the EDMF updrafts, or to compute
grid-mean tendency without taking account of fluctuations. It computes the
0M precipitation removal tendencies based on the provided point-values of temperature
and specific humidities.

When running with explicit microphysics timestepping, the total water sink
is limited by the available water.

# Inputs
- `SG_quad`: SGSQuadrature configuration
- `cmp`, `thp` - cloud microphysics and thermodynamics parameters
- `ρ`, `T` - density [kg/m3] and temperature [K]
- `q_tot`, `q_liq`, `q_ice` - total water, liquid water and ice specific humidities [kg/kg]
- `ϕ` - geopotential
- `T′T′`: Variance of temperature ``\\langle T'^2 \\rangle``
- `q′q′`: Variance of q_tot ``\\langle q'^2 \\rangle``
- `corr_Tq`: Correlation coefficient ρ(T', q')
- `dt`: model timestep length [s]

# Returns
NamedTuple with `dq_tot_dt` and `e_tot_hlpr`.
"""
@inline function microphysics_tendencies_0m(
    SG_quad, cmp, thp, ρ, T, q_tot_nonneg, T′T′, q′q′, corr_Tq, Φ, dt,
)
    # Create GPU-safe functor (Φ is constant within a grid cell)
    # The evaluator does saturation adjustment, computes saturation vapor pressure
    # and computes the total water sink and energy helper from 0M microphysics
    evaluator = Microphysics0MEvaluator(cmp, thp, ρ, T, Φ)
    # Integrate over quadrature points; both dq_tot_dt and e_tot_hlpr
    # are averaged over the SGS distribution.
    (; dq_tot_dt, e_tot_hlpr) = integrate_over_sgs(
        evaluator, SG_quad, q_tot_nonneg, T, q′q′, T′T′, corr_Tq,
    )
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
    equilibrium_cloud_condensates_1m(tps, T, ρ, q_tot, λ, q_rai, q_sno)

Compute the equilibrium cloud liquid and cloud ice specific humidities
(`q_lcl_eq`, `q_icl_eq`) for a thermodynamic state `(T, ρ, q_tot)` using
a prescribed (grid-mean) liquid fraction `λ`. Rain and snow condensates
are removed from the equilibrium partition.

Returns `(q_lcl_eq, q_icl_eq)`.
"""
@inline function equilibrium_cloud_condensates_1m(tps, T, ρ, q_tot, λ, q_rai, q_sno)
    FT = typeof(ρ)
    q_sat = TD.q_vap_saturation(tps, T, ρ)
    excess = max(FT(0), q_tot - q_sat)
    q_lcl_eq = max(FT(0), λ * excess - q_rai)
    q_icl_eq = max(FT(0), (FT(1) - λ) * excess - q_sno)
    return q_lcl_eq, q_icl_eq
end

"""
    (eval::Microphysics1MEvaluator)(T_hat, q_tot_hat)

GPU-safe functor for computing 1-moment microphysics tendencies at SGS
quadrature points. The local thermodynamic state `(T_hat, q_tot_hat)` is
sampled from the SGS distribution; the local condensate is diagnosed from
the local saturation excess and capped at the *in-cloud* condensate
`q_lcl_in_cloud = q_lcl_mean / max(CF, CF_min)` (and analogously for ice).

# Arguments
- `T_hat`: Temperature at quadrature point [K]
- `q_tot_hat`: Total specific humidity at quadrature point [kg/kg]

# Returns
`NamedTuple` from `BMT.average_bulk_microphysics_tendencies` with fields:
- `dq_lcl_dt`: Cloud liquid tendency [kg/kg/s]
- `dq_icl_dt`: Cloud ice tendency [kg/kg/s]
- `dq_rai_dt`: Rain tendency [kg/kg/s]
- `dq_sno_dt`: Snow tendency [kg/kg/s]

# Condensate Diagnosis

At each quadrature point, the local cloud condensate is

    q_lcl_hat = min(q_lcl_in_cloud,
                    max(0, λ (q_tot_hat - q_sat(T_hat, ρ)) - q_rai))

with `λ` the prognostic-state liquid fraction (held fixed across all
quadrature points of the subdomain). At clear quadrature points
(`q_tot_hat ≤ q_sat(T_hat, ρ)`), `q_lcl_hat = 0`. At strongly supersaturated
points, `q_lcl_hat` saturates at the in-cloud value. This bounds the
Jensen's-inequality amplification of nonlinear sinks (autoconversion,
accretion) by the standard GCM in-cloud / clear partition while letting the
quadrature respond to local thermodynamic variability through the
saturation curve.

When the subdomain is clear (`q_lcl_mean = 0`), `q_lcl_in_cloud = 0` and the
local condensate is identically zero at every quadrature point;
phase-change rates remain responsive to local SGS supersaturation through
`q_v_hat - q_sat(T_hat, ρ)` inside `BMT`.
"""
struct Microphysics1MEvaluator{S, MP, TPS, FT, Args <: Tuple}
    scheme::S
    mp::MP
    tps::TPS
    ρ::FT
    # Grid-mean temperature (passed through; quadrature varies T_hat)
    T_mean::FT
    # In-cloud cloud condensates: q_mean / max(CF, CF_min)
    q_lcl_in_cloud::FT
    q_icl_in_cloud::FT
    # Grid-mean precipitation (in-cloud / clear partition not applied)
    q_rai::FT
    q_sno::FT
    # Prognostic-state liquid fraction, fixed across quadrature points
    λ::FT
    # Numerical parameters
    dt::FT
    nsubs::Int
    args::Args
end
@inline function (eval::Microphysics1MEvaluator)(T_hat, q_tot_hat)

    q_lcl_eq_hat, q_icl_eq_hat = equilibrium_cloud_condensates_1m(
        eval.tps, T_hat, eval.ρ, q_tot_hat, eval.λ, eval.q_rai, eval.q_sno,
    )

    # Cap the local equilibrium fluctuation at the in-cloud condensate so that
    # nonlinear sinks see at most the in-cloud value within the cloudy fraction.
    q_lcl_hat = min(eval.q_lcl_in_cloud, q_lcl_eq_hat)
    q_icl_hat = min(eval.q_icl_in_cloud, q_icl_eq_hat)

    return BMT.average_bulk_microphysics_tendencies(
        eval.scheme, eval.mp, eval.tps, eval.ρ, T_hat, q_tot_hat,
        q_lcl_hat, q_icl_hat, eval.q_rai, eval.q_sno, eval.dt, eval.nsubs, eval.args...,
    )
end

"""
    microphysics_tendencies_1m(ρ, q_tot, q_lcl, q_icl, q_rai, q_sno, T, cmp, thp, dt, nsubs,)
    microphysics_tendencies_1m(
        scheme, sgs_quad, cmp, thp, ρ, T, q_tot,
        q_lcl_mean, q_icl_mean, q_rai, q_sno, T′T′, q′q′, corr_Tq,
        cloud_fraction, dt, nsubs,
    )

Computes time-averaged 1-moment microphysics tendencies. When using SGS-quadratures,
the tendencies are integrated over the joint PDF of (T, q_tot), with the
condensate at each quadrature point taken from the local equilibrium excess
capped at the in-cloud condensate `q_lcl_mean / max(CF, CF_min)` (similarly
for ice). The grid-mean (no-quadrature) path takes condensate inputs as-is.
```math
\\bar{S} = \\int\\int S(T, q, q_l(T, q; q_l^{ic}), \\dots) P(T, q) \\, dT \\, dq
```
The option without SGS-quadratures can be used in the EDMF updrafts, or to compute
grid-mean tendency without taking account of fluctuations.

# Arguments
- `scheme`: Microphysics scheme type (from CloudMicrophysics.BulkMicrophysicsTendencies)
- `sgs_quad`: SGSQuadrature configuration
- `cmp`, `thp`: Microphysics and thermodynamics parameters
- `ρ`, `T`: Air density [kg/m³] and temperature [K]
- `q_tot`: Total water [kg/kg]
- `q_lcl_mean`, `q_icl_mean`: Mean cloud liquid water and cloud ice [kg/kg]
- `q_rai`, `q_sno`: Rain and snow [kg/kg]
- `T′T′`: Variance of temperature ``\\langle T'^2 \\rangle``
- `q′q′`: Variance of q_tot ``\\langle q'^2 \\rangle``
- `corr_Tq`: Correlation coefficient ρ(T', q') obtained from `correlation_Tq(params)`.
- `cloud_fraction`: Subdomain cloud fraction CF ∈ [0, 1], used to convert grid-mean
  condensates to in-cloud values via `q_lcl_in_cloud = q_lcl_mean / max(CF, CF_min)`.
- `args...`: Additional arguments (e.g., N_lcl for 2M)
- `dt`: Model timestep [s] (for tendency averaging)
- `nsubs`: Number of substeps for computing average tendencies over time

# Returns
NamedTuple with microphysics source terms:
- `dq_lcl_dt`: Cloud liquid tendency [kg/kg/s]
- `dq_icl_dt`: Cloud ice tendency [kg/kg/s]
- `dq_rai_dt`: Rain tendency [kg/kg/s]
- `dq_sno_dt`: Snow tendency [kg/kg/s]
"""
@inline function microphysics_tendencies_1m( #compute_1m_precipitation_tendencies!(
    ρ, q_tot_nonneg, q_lcl, q_icl, q_rai, q_sno, T, cmp, thp, dt, nsubs,
)
    local_tendency = BMT.average_bulk_microphysics_tendencies(
        BMT.Microphysics1Moment(), cmp, thp, ρ, T,
        q_tot_nonneg, q_lcl, q_icl, q_rai, q_sno, dt, nsubs,
    )
    return local_tendency
end
@inline function microphysics_tendencies_1m( #microphysics_tendencies_quadrature_1m
    scheme, sgs_quad, cmp, thp, ρ, T, q_tot_nonneg,
    q_lcl_mean, q_icl_mean, q_rai, q_sno, T′T′, q′q′, corr_Tq,
    cloud_fraction, dt, nsubs, args...,
)
    FT = typeof(ρ)

    # Clamp specific humidities to prevent negativity in quadratures
    q_lcl_nonneg = max(FT(0), q_lcl_mean)
    q_icl_nonneg = max(FT(0), q_icl_mean)
    q_rai_nonneg = max(FT(0), q_rai)
    q_sno_nonneg = max(FT(0), q_sno)

    # Convert grid-mean cloud condensates to in-cloud values, q_in_cloud = q_mean / CF.
    # Floor CF to keep q_in_cloud finite at small cloud fractions.
    CF_min = FT(0.01)
    cf_eff = max(cloud_fraction, CF_min)
    q_lcl_in_cloud = q_lcl_nonneg / cf_eff
    q_icl_in_cloud = q_icl_nonneg / cf_eff

    # Liquid fraction from prognostic cloud-condensate composition
    # (paper Eq. 3.31): q_c = q_l + q_i (cloud condensate only; rain and snow
    # are excluded). Held fixed across quadrature points. TD.liquid_fraction
    # returns q_l/(q_l+q_i) when condensate exists and a smooth temperature
    # ramp around T_freeze otherwise.
    λ = TD.liquid_fraction(thp, T, q_lcl_nonneg, q_icl_nonneg)

    # Create functor
    evaluator = Microphysics1MEvaluator(
        scheme, cmp, thp, ρ, T, q_lcl_in_cloud, q_icl_in_cloud,
        q_rai_nonneg, q_sno_nonneg, λ, dt, nsubs, args,
    )
    # Integrate over quadrature points using functor (GPU-safe, no closure)
    local_tendency = integrate_over_sgs(
        evaluator, sgs_quad, q_tot_nonneg, T, q′q′, T′T′, corr_Tq,
    )
    return local_tendency
end

###
### 2 Moment Microphysics
###

"""
    compute_prescribed_aerosol_properties!(
        seasalt_num, seasalt_mean_radius, sulfate_num,
        prescribed_aerosol_field, aerosol_params,
    )

Computes the number concentrations (per unit mass of air) of prescribed sea salt and sulfate aerosols, as well as
the geometric mean radius of sea salt aerosol, and writes the results in-place.

# Arguments
- `seasalt_num`: Array to be overwritten with the total number concentration of sea salt aerosol [kg⁻¹].
- `seasalt_mean_radius`: Array to be overwritten with the geometric mean radius of sea salt aerosol [m].
- `sulfate_num`: Array to be overwritten with the total number concentration of sulfate aerosol [kg⁻¹].
- `prescribed_aerosol_field`: A container holding mass mixing ratios of aerosol tracers (e.g., `:SSLT01`, `:SO4`).
- `aerosol_params`: Parameters defining aerosol properties (e.g., density, mode radius, geometric standard deviation, hygroscopicity).

# Notes
- Sea salt number concentration and mean radius are computed by aggregating contributions from all available `:SSLT0X` modes.
- If no sea salt is present, the mean radius is set to zero to avoid division by zero.
- Aerosol mass is converted to number using assumed particle radii and densities.
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

Computes the source term for cloud droplet number concentration per mass due to aerosol activation,
based on the Abdul-Razzak and Ghan (2000) parameterization.

This function estimates the number of aerosols activated into cloud droplets per mass of air per second
from a bi-modal aerosol distribution (sea salt and sulfate), given local supersaturation and vertical
velocity. The result is returned as a tendency (per second) of liquid droplet number concentration.

# Arguments
- `act_params`: Aerosol activation parameters (AerosolActivationParameters)
- `seasalt_num`: Number concentration per mass of sea salt aerosols [kg⁻¹]
- `seasalt_mean_radius`: Mean dry radius of sea salt aerosol mode [m]
- `sulfate_num`: Number concentration per mass of sulfate aerosols [kg⁻¹]
- `qₜ`: Total water specific humidity [kg/kg]
- `qₗ`: Liquid water (cloud + rain) specific humidity [kg/kg]
- `qᵢ`: Ice water (cloud ice + snow) specific humidity [kg/kg]
- `nₗ`: Liquid droplet number concentration per mass [kg⁻¹]
- `ρ`: Air density [kg/m³]
- `w`: Vertical velocity [m/s]
- `cmp`: Microphysics2MParams parameters
- `thermo_params`: Thermodynamics parameters
- `T`: Air temperature [K]
- `p`: Air pressure [Pa]
- `dt`: Model timestep [s]
- `aerosol_params`: Prescribed aerosol parameters (NamedTuple with seasalt/sulfate properties)

# Returns
- Tendency of cloud liquid droplet number concentration per mass of air due to aerosol activation [kg⁻¹/s].
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
    compute_2m_precipitation_tendencies!(mp_tendency, ρ, qₜ, qₗ, nₗ, qᵣ, nᵣ, T, dt, mp, thp)

Compute 2-moment warm rain microphysics tendencies (cloud condensation/evaporation,
autoconversion, accretion, and precipitation) in a single call.

# Arguments
- `mp_tendency`: Output NamedTuple for liquid mass, liquid number, rain mass, rain number tendencies
- `ρ`: Air density [kg/m³]
- `qₜ`: Total water specific humidity [kg/kg]
- `qₗ`: Cloud liquid specific humidity [kg/kg]
- `nₗ`: Cloud liquid number concentration [1/kg]
- `qᵣ`: Rain specific humidity [kg/kg]
- `nᵣ`: Rain number concentration [1/kg]
- `T`: Air temperature [K]
- `dt`: Model timestep [s] (for tendency limiting)
- `mp`: Microphysics parameters (`CMP.Microphysics2MParams`)
- `thp`: Thermodynamics parameters

# Output
Modifies mp_tendency in-place with limited tendencies.
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
    microphysics_tendencies_quadrature_2m(...)

SGS quadrature integration for Microphysics2Moment (warm rain only).

!!! warning "Limited SGS support"
    Only `GridMeanSGS` is currently supported for 2-moment microphysics.
    Passing any other distribution type (e.g., `GaussianSGS`) will fall back to
    grid-mean evaluation. Full quadrature integration for 2M would require a
    separate evaluator that handles number concentration perturbations.

# Arguments
- `SG_quad`: SGSQuadrature configuration (only `GridMeanSGS` supported)
- `cmp`: Microphysics 2M parameters
- `tps`: Thermodynamics parameters
- `ρ`: Air density [kg/m³]
- `T`: Temperature [K]
- `q_liq`: Cloud liquid [kg/kg]
- `n_liq`: Cloud liquid number [1/kg]
- `q_rai`: Rain [kg/kg]
- `n_rai`: Rain number [1/kg]

# Returns
NamedTuple with tendencies: `dq_lcl_dt`, `dn_lcl_dt`, `dq_rai_dt`, `dn_rai_dt`
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
