# A set of wrappers for using CloudMicrophysics.jl functions inside EDMFX loops

import Thermodynamics as TD
import CloudMicrophysics.Parameters as CMP
import CloudMicrophysics.BulkMicrophysicsTendencies as BMT
import CloudMicrophysics.AerosolModel as CMAM
import CloudMicrophysics.AerosolActivation as CMAA

# Import SGS quadrature utilities
using ..ClimaAtmos: integrate_over_sgs


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
    compute_1m_precipitation_tendencies!(Sqₗᵐ, Sqᵢᵐ, Sqᵣᵐ, Sqₛᵐ, ρ, qₜ, qₗ, qᵢ, qᵣ, qₛ, T, dt, mp, thp)

Compute 1-moment microphysics tendencies using the fused BMT API.

This function computes all microphysics tendencies (cloud condensation/evaporation,
autoconversion, accretion, and precipitation) in a single call.

# Arguments
- `Sqₗᵐ`, `Sqᵢᵐ`, `Sqᵣᵐ`, `Sqₛᵐ`: Output arrays for liquid, ice, rain, snow tendencies [1/s]
- `ρ`: Air density [kg/m³]
- `qₜ`: Total water specific humidity [kg/kg]
- `qₗ`: Cloud liquid specific humidity [kg/kg]
- `qᵢ`: Cloud ice specific humidity [kg/kg]
- `qᵣ`: Rain specific humidity [kg/kg]
- `qₛ`: Snow specific humidity [kg/kg]
- `T`: Air temperature [K]
- `dt`: Model timestep [s] (for tendency limiting)
- `mp`: Microphysics parameters (`CMP.Microphysics1MParams`)
- `thp`: Thermodynamics parameters

# Output
Modifies `Sqₗᵐ`, `Sqᵢᵐ`, `Sqᵣᵐ`, `Sqₛᵐ` in-place with limited tendencies.

# Note
Tendencies are limited to prevent unphysically large
tendencies.

`mp_tendency` is a pre-allocated scratch field of matching NamedTuple type,
used to avoid allocations from the BMT return value.
"""
function compute_1m_precipitation_tendencies!(
    Sqₗᵐ,
    Sqᵢᵐ,
    Sqᵣᵐ,
    Sqₛᵐ,
    mp_tendency,
    ρ,
    qₜ,
    qₗ,
    qᵢ,
    qᵣ,
    qₛ,
    T,
    dt,
    mp,
    thp,
)
    FT = eltype(thp)

    # Call BMT to get all tendencies at once (writes into pre-allocated mp_tendency)
    @. mp_tendency = BMT.bulk_microphysics_tendencies(
        BMT.Microphysics1Moment(),
        mp,
        thp,
        ρ,
        T,
        qₜ,
        qₗ,
        qᵢ,
        qᵣ,
        qₛ,
    )

    # Apply limiting via shared helper
    @. mp_tendency = apply_1m_tendency_limits(mp_tendency, thp, qₜ, qₗ, qᵢ, qᵣ, qₛ, dt)
    @. Sqₗᵐ = mp_tendency.dq_lcl_dt
    @. Sqᵢᵐ = mp_tendency.dq_icl_dt
    @. Sqᵣᵐ = mp_tendency.dq_rai_dt
    @. Sqₛᵐ = mp_tendency.dq_sno_dt
end

"""
    compute_2m_precipitation_tendencies!(Sqₗᵐ, Snₗᵐ, Sqᵣᵐ, Snᵣᵐ, ρ, qₜ, qₗ, nₗ, qᵣ, nᵣ, T, dt, mp, thp)

Compute 2-moment microphysics tendencies using the fused BMT API (warm rain only).

This function computes all microphysics tendencies (cloud condensation/evaporation,
autoconversion, accretion, and precipitation) in a single call.

# Arguments
- `Sqₗᵐ`, `Snₗᵐ`, `Sqᵣᵐ`, `Snᵣᵐ`: Output arrays for liquid mass, liquid number, rain mass, rain number tendencies
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
Modifies output arrays in-place with limited tendencies.

# Note
Tendencies are limited using `limit_sink` to prevent unphysical
depletion of hydrometeor categories.

`mp_tendency` is a pre-allocated scratch field of matching NamedTuple type,
used to avoid allocations from the BMT return value.
"""
function compute_2m_precipitation_tendencies!(
    Sqₗᵐ,
    Snₗᵐ,
    Sqᵣᵐ,
    Snᵣᵐ,
    mp_tendency,
    ρ,
    qₜ,
    qₗ,
    nₗ,
    qᵣ,
    nᵣ,
    T,
    dt,
    mp,
    thp,
)
    FT = eltype(thp)

    # Call BMT to get all tendencies at once (writes into pre-allocated mp_tendency)
    @. mp_tendency = BMT.bulk_microphysics_tendencies(
        BMT.Microphysics2Moment(),
        mp,
        thp,
        ρ,
        T,
        qₜ,
        qₗ,
        nₗ,
        qᵣ,
        nᵣ,
    )

    # Apply coupled limiting directly
    f_liq = @. lazy(
        coupled_sink_limit_factor(
            mp_tendency.dq_lcl_dt, mp_tendency.dn_lcl_dt, qₗ, nₗ, dt,
        ),
    )
    f_rai = @. lazy(
        coupled_sink_limit_factor(
            mp_tendency.dq_rai_dt, mp_tendency.dn_rai_dt, qᵣ, nᵣ, dt,
        ),
    )
    @. Sqₗᵐ = mp_tendency.dq_lcl_dt * f_liq
    @. Snₗᵐ = mp_tendency.dn_lcl_dt * f_liq
    @. Sqᵣᵐ = mp_tendency.dq_rai_dt * f_rai
    @. Snᵣᵐ = mp_tendency.dn_rai_dt * f_rai
end

#####
##### 2M microphysics
#####

"""
    compute_prescribed_aerosol_properties!(
        seasalt_num,
        seasalt_mean_radius,
        sulfate_num,
        prescribed_aerosol_field,
        aerosol_params,
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
    seasalt_num,
    seasalt_mean_radius,
    sulfate_num,
    prescribed_aerosol_field,
    aerosol_params,
)

    FT = eltype(aerosol_params)
    @. seasalt_num = 0
    @. seasalt_mean_radius = 0
    @. sulfate_num = 0

    # Get aerosol concentrations if available
    seasalt_names = [:SSLT01, :SSLT02, :SSLT03, :SSLT04, :SSLT05]
    sulfate_names = [:SO4]
    for aerosol_name in propertynames(prescribed_aerosol_field)
        if aerosol_name in seasalt_names
            seasalt_particle_radius = getproperty(
                aerosol_params,
                Symbol(string(aerosol_name) * "_radius"),
            )
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
        act_params,
        seasalt_num,
        seasalt_mean_radius,
        sulfate_num,
        qₜ,
        qₗ,
        qᵢ,
        nₗ,
        ρ,
        w,
        cmp,
        thermo_params,
        T,
        p,
        dt,
        aerosol_params,
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
    act_params,
    seasalt_num,
    seasalt_mean_radius,
    sulfate_num,
    qₜ,
    qₗ,
    qᵢ,
    nₗ,
    ρ,
    w,
    cmp,
    thermo_params,
    T,
    p,
    dt,
    aerosol_params,  # Tuple-wrapped at call sites for broadcast safety
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
        act_params,
        distribution,
        air_params,
        thermo_params,
        T,
        p,
        w,
        qₜ,
        qₗ,
        qᵢ,
        nₗ * ρ,
        FT(0),
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

# ============================================================================
# SGS Quadrature Integration for 0-Moment Microphysics
# ============================================================================

"""
    Microphysics0MEvaluator

GPU-safe functor for computing 0-moment microphysics tendencies at quadrature
points. Delegates condensate diagnosis to `SaturationAdjustmentEvaluator`,
then calls `BMT.bulk_microphysics_tendencies(Microphysics0Moment(), ...)`.

# Fields
- `cm_params`: 0M microphysics parameters
- `sat_eval`: `SaturationAdjustmentEvaluator` for condensate diagnosis
"""
struct Microphysics0MEvaluator{CMP, SAE}
    cm_params::CMP
    sat_eval::SAE
end

function Microphysics0MEvaluator(cm_params, thermo_params, ρ)
    sat_eval = SaturationAdjustmentEvaluator(thermo_params, ρ)
    return Microphysics0MEvaluator(cm_params, sat_eval)
end

@inline function (eval::Microphysics0MEvaluator)(T_hat, q_hat)
    # Diagnose condensate via saturation adjustment
    sa = eval.sat_eval(T_hat, q_hat)

    # Compute q_sat for the 0M tendency function
    # Clamp to non-negative: Float32 rounding in the λ/(1-λ) split can make
    # q_liq + q_ice slightly exceed q_cond, yielding a tiny negative remainder.
    q_sat = max(zero(q_hat), q_hat - sa.q_liq - sa.q_ice)

    # Compute 0M tendencies at this quadrature point
    return BMT.bulk_microphysics_tendencies(
        BMT.Microphysics0Moment(),
        eval.cm_params, eval.sat_eval.thermo_params,
        T_hat, sa.q_liq, sa.q_ice, q_sat,
    )
end

"""
    microphysics_tendencies_quadrature_0m(
        SG_quad, cm_params, thermo_params,
        ρ, T_mean, q_tot_mean, T′T′, q′q′, corr_Tq,
    )

Compute SGS-averaged 0-moment microphysics tendencies by integrating over
the joint PDF of (T, q_tot). At each quadrature point, condensate is diagnosed
from saturation excess, then 0M precipitation removal tendencies are computed.

# Returns
NamedTuple with SGS-averaged `dq_tot_dt` and `e_int_precip`.
"""
@inline function microphysics_tendencies_quadrature_0m(
    SG_quad,
    cm_params, thermo_params,
    ρ, T_mean, q_tot_mean,
    T′T′, q′q′, corr_Tq,
)
    # Create GPU-safe functor
    evaluator = Microphysics0MEvaluator(cm_params, thermo_params, ρ)

    # Integrate over quadrature points
    return integrate_over_sgs(evaluator, SG_quad, q_tot_mean, T_mean, q′q′, T′T′, corr_Tq)
end


#####
##### SGS Quadrature Integration for Microphysics
#####

"""
    microphysics_tendencies_quadrature(
        scheme,
        SG_quad,
        mp, tps,
        ρ, p_c,
        T_mean, q_tot_mean, q_lcl_mean, q_icl_mean, q_rai, q_sno,
        T′T′, q′q′, corr_Tq,
        args...
    )

Compute subgrid-scale (SGS) averaged microphysics tendencies by integrating
point-wise microphysics over the joint PDF of (T, q_tot):

```math
\\bar{S} = \\int\\int S(T, q, \\dots) P(T, q) \\, dT \\, dq
```

This enables microphysics tendencies to account for subgrid-scale fluctuations
in temperature and moisture, capturing threshold effects at cloud edges and
improving representation of phase changes.

# Arguments
- `scheme`: Microphysics scheme type (from CloudMicrophysics.BulkMicrophysicsTendencies)
- `SG_quad`: SGSQuadrature configuration
- `mp`: Microphysics parameters (scheme-specific)
- `tps`: Thermodynamics parameters
- `ρ`: Air density [kg/m³]
- `p_c`: Pressure [Pa]
- `T_mean`: Mean temperature [K]
- `q_tot_mean`: Mean total water [kg/kg]
- `q_lcl_mean`: Mean cloud liquid [kg/kg]
- `q_icl_mean`: Mean cloud ice [kg/kg]
- `q_rai`: Rain [kg/kg] (not perturbed)
- `q_sno`: Snow [kg/kg] (not perturbed)
- `T′T′`: Variance of temperature ``\\langle T'^2 \\rangle``
- `q′q′`: Variance of q_tot ``\\langle q'^2 \\rangle``
- `corr_Tq`: Correlation coefficient ρ(T', q')
- `args...`: Additional arguments (e.g., N_lcl for 2M)

# Returns
NamedTuple with averaged source terms:
- `dq_lcl_dt`: Cloud liquid tendency [kg/kg/s]
- `dq_icl_dt`: Cloud ice tendency [kg/kg/s]
- `dq_rai_dt`: Rain tendency [kg/kg/s]
- `dq_sno_dt`: Snow tendency [kg/kg/s]

# Condensate Model

At each quadrature point (T_hat, q_tot_hat):
1. Compute saturation q_sat_hat = q_vap_saturation(T_hat, ρ)
2. Diagnose condensate: q_cond = max(0, q_tot_hat - q_sat_hat)
3. Partition using liquid fraction: q_lcl_hat = λ × q_cond, q_icl_hat = (1-λ) × q_cond

This captures threshold behavior at cloud edges where saturation excess transitions.

# Note on Variances

Call sites must convert θ-based variances to T-based variances using the chain rule:
```julia
∂T_∂θ = ... # (∂T/∂θ_liq_ice) computed at grid mean state
T′T′ = (∂T_∂θ)² × θ′θ′
```
The T-q correlation coefficient is obtained from `correlation_Tq(params)`.
"""

# ============================================================================
# Kernel Fission: Pre-computation Helper
# ============================================================================

"""
    MicrophysicsPrecompute{FT}

Pre-computed saturation values for quadrature integration. By computing these
in a separate lightweight step, we reduce the size of the evaluator struct
passed through the quadrature loop, reducing register pressure.

# Fields
- `q_cond_mean::FT`: Grid-mean condensate (q_lcl + q_icl)
- `q_sat_mean::FT`: Grid-mean saturation specific humidity
- `excess_mean::FT`: Grid-mean saturation excess (q_tot - q_sat)
"""
struct MicrophysicsPrecompute{FT}
    q_cond_mean::FT
    q_sat_mean::FT
    excess_mean::FT
end

"""
    compute_microphysics_precompute(tps, ρ, T_mean, q_tot_mean, q_lcl_mean, q_icl_mean)

Pre-compute grid-mean saturation values. This lightweight computation can be
fused by the compiler or split into a separate kernel, reducing register pressure
in the main quadrature loop.

# Returns
`MicrophysicsPrecompute` struct with pre-computed saturation values.
"""
@inline function compute_microphysics_precompute(
    tps,
    ρ::FT,
    T_mean::FT,
    q_tot_mean::FT,
    q_lcl_mean::FT,
    q_icl_mean::FT,
) where {FT}
    q_cond_mean = q_lcl_mean + q_icl_mean
    q_sat_mean = TD.q_vap_saturation(tps, T_mean, ρ)
    excess_mean = q_tot_mean - q_sat_mean

    return MicrophysicsPrecompute(q_cond_mean, q_sat_mean, excess_mean)
end

# ============================================================================
# Original Evaluator (kept for reference/compatibility)
# ============================================================================

struct MicrophysicsEvaluator{S, MP, TPS, FT, Args <: Tuple}
    scheme::S
    mp::MP
    tps::TPS
    ρ::FT
    # Grid-mean state
    T_mean::FT
    q_tot_mean::FT
    q_lcl_mean::FT
    q_icl_mean::FT
    q_rai::FT
    q_sno::FT
    # Precomputed grid-mean values (avoid redundant computation in quadrature loop)
    q_cond_mean::FT
    q_sat_mean::FT      # Saturation at grid mean
    excess_mean::FT     # Saturation excess at grid mean (q_tot_mean - q_sat_mean)
    args::Args
end

"""
    (eval::MicrophysicsEvaluator)(T_hat, q_tot_hat)

Evaluate microphysics tendencies at a quadrature point.

This functor computes bulk microphysics tendencies at a perturbed thermodynamic
state `(T_hat, q_tot_hat)` for use in SGS quadrature integration. The condensate
at each quadrature point is diagnosed using a perturbation-based model.

# Arguments
- `T_hat`: Temperature at quadrature point [K]
- `q_tot_hat`: Total specific humidity at quadrature point [kg/kg]

# Returns
`NamedTuple` from `BMT.bulk_microphysics_tendencies` with fields:
- `dq_lcl_dt`: Cloud liquid tendency [kg/kg/s]
- `dq_icl_dt`: Cloud ice tendency [kg/kg/s]
- `dq_rai_dt`: Rain tendency [kg/kg/s]
- `dq_sno_dt`: Snow tendency [kg/kg/s]

# Condensate Diagnosis

At each quadrature point, condensate is diagnosed as:

    q_cond_hat = max(0, excess_hat + bias)

where `excess_hat = q_tot_hat - q_sat(T_hat, ρ)` is the local saturation excess,
and `bias` corrects for any non-equilibrium offset at the grid mean:

    bias = q_cond_mean - max(0, excess_mean)

The bias represents prognostic condensate that cannot be explained by saturation
alone (e.g., condensate persisting in a subsaturating environment). When the grid
mean is in equilibrium (`q_cond_mean ≈ max(0, excess_mean)`), the bias vanishes
and condensate at each quadrature point reduces to `max(0, excess_hat)`.

When the grid mean is subsaturated (`excess_mean < 0`) with no condensate
(`q_cond_mean = 0`), the bias is zero, so condensate forms only where the
quadrature point is actually supersaturated.

Condensate is partitioned into liquid and ice using `λ(T_hat)`.
"""
@inline function (eval::MicrophysicsEvaluator)(T_hat, q_tot_hat)
    FT = typeof(eval.ρ)

    # Compute saturation excess perturbation relative to grid mean
    # At grid mean: q_cond_mean = q_lcl_mean + q_icl_mean (given)
    # At quadrature point: q_cond_hat = q_cond_mean + Δq_cond
    # where Δq_cond is the perturbation in saturation excess

    # Compute saturation at quadrature point only (grid mean is precomputed)
    q_sat_hat = TD.q_vap_saturation(eval.tps, T_hat, eval.ρ)

    # Saturation excess at the quadrature point
    excess_hat = q_tot_hat - q_sat_hat

    # Non-equilibrium bias: difference between prognostic condensate and
    # equilibrium condensate (clamped to saturated regime only).
    # - Subsaturated mean (excess_mean < 0, q_cond_mean = 0): bias = 0
    # - Saturated equilibrium (q_cond_mean ≈ excess_mean): bias ≈ 0
    # - Non-equilibrium (q_cond_mean ≠ excess_mean): bias preserves offset
    bias = eval.q_cond_mean - max(FT(0), eval.excess_mean)
    q_cond_hat = max(FT(0), excess_hat + bias)

    # Partition using grid-mean liquid fraction
    λ = TD.liquid_fraction(eval.tps, T_hat, eval.q_lcl_mean, eval.q_icl_mean)

    # Scale condensate to preserve the partitioning ratio from grid mean
    # If q_cond_mean > 0, scale proportionally; otherwise use λ partitioning
    # Use eps-guarded division to prevent overflow when q_cond_mean is tiny
    has_grid_mean_condensate = eval.q_cond_mean > FT(0)
    scale = ifelse(
        has_grid_mean_condensate,
        q_cond_hat / max(eval.q_cond_mean, ϵ_numerics(FT)),
        FT(1),
    )

    q_lcl_hat = ifelse(
        has_grid_mean_condensate,
        eval.q_lcl_mean * scale,
        λ * q_cond_hat,
    )
    q_icl_hat = ifelse(
        has_grid_mean_condensate,
        eval.q_icl_mean * scale,
        (FT(1) - λ) * q_cond_hat,
    )

    # Ensure non-negative and clamp to physical bounds
    q_lcl_hat = max(FT(0), q_lcl_hat)
    q_icl_hat = max(FT(0), q_icl_hat)
    q_tot_hat = max(FT(0), q_tot_hat)

    # Call CloudMicrophysics point-wise tendencies
    return BMT.bulk_microphysics_tendencies(
        eval.scheme, eval.mp, eval.tps, eval.ρ, T_hat,
        q_tot_hat, q_lcl_hat, q_icl_hat, eval.q_rai, eval.q_sno,
        eval.args...,
    )
end

# ============================================================================
# Simplified Evaluator with Pre-computed Values
# ============================================================================

"""
    MicrophysicsEvaluatorSimple

Simplified evaluator that receives pre-computed saturation values, reducing
struct size and register pressure compared to `MicrophysicsEvaluator`.

By removing `q_tot_mean` and packing pre-computed values into a small struct,
we reduce the evaluator footprint and help the compiler generate more efficient
GPU code with lower register usage.
"""
struct MicrophysicsEvaluatorSimple{S, MP, TPS, FT, PC, Args <: Tuple}
    scheme::S
    mp::MP
    tps::TPS
    ρ::FT
    # Grid-mean prognostic state (only what's needed for quadrature point eval)
    T_mean::FT
    q_lcl_mean::FT
    q_icl_mean::FT
    q_rai::FT
    q_sno::FT
    # Pre-computed values (from kernel 1)
    precomp::PC
    args::Args
end

@inline function (eval::MicrophysicsEvaluatorSimple)(T_hat, q_tot_hat)
    FT = typeof(eval.ρ)

    # Compute saturation at quadrature point only
    q_sat_hat = TD.q_vap_saturation(eval.tps, T_hat, eval.ρ)
    excess_hat = q_tot_hat - q_sat_hat

    # Use pre-computed bias
    bias = eval.precomp.q_cond_mean - max(FT(0), eval.precomp.excess_mean)
    q_cond_hat = max(FT(0), excess_hat + bias)

    # Partition using grid-mean liquid fraction
    λ = TD.liquid_fraction(eval.tps, T_hat, eval.q_lcl_mean, eval.q_icl_mean)

    has_grid_mean_condensate = eval.precomp.q_cond_mean > FT(0)
    scale = ifelse(
        has_grid_mean_condensate,
        q_cond_hat / max(eval.precomp.q_cond_mean, ϵ_numerics(FT)),
        FT(1),
    )

    q_lcl_hat = ifelse(
        has_grid_mean_condensate,
        eval.q_lcl_mean * scale,
        λ * q_cond_hat,
    )
    q_icl_hat = ifelse(
        has_grid_mean_condensate,
        eval.q_icl_mean * scale,
        (FT(1) - λ) * q_cond_hat,
    )

    # Ensure non-negative
    q_lcl_hat = max(FT(0), q_lcl_hat)
    q_icl_hat = max(FT(0), q_icl_hat)
    q_tot_hat = max(FT(0), q_tot_hat)

    return BMT.bulk_microphysics_tendencies(
        eval.scheme, eval.mp, eval.tps, eval.ρ, T_hat,
        q_tot_hat, q_lcl_hat, q_icl_hat, eval.q_rai, eval.q_sno,
        eval.args...,
    )
end

# ============================================================================
# Main Quadrature Function with Kernel Fission
# ============================================================================

@inline function microphysics_tendencies_quadrature(
    scheme,
    SG_quad,
    mp, tps,
    ρ, p_c,
    T_mean, q_tot_mean, q_lcl_mean, q_icl_mean, q_rai, q_sno,
    T′T′, q′q′, corr_Tq,
    args...,  # Keep varargs for GPU compatibility
)
    FT = eltype(tps)

    # Fast path for GridMeanSGS: skip quadrature, call BMT directly at grid mean
    if SG_quad isa GridMeanSGS ||
       (SG_quad isa SGSQuadrature && SG_quad.dist isa GridMeanSGS)
        return BMT.bulk_microphysics_tendencies(
            scheme, mp, tps, ρ, T_mean,
            q_tot_mean, q_lcl_mean, q_icl_mean, q_rai, q_sno,
            args...,
        )
    end

    # Pre-compute saturation values (lightweight, can be fused or split)
    precomp = compute_microphysics_precompute(
        tps, ρ, T_mean, q_tot_mean, q_lcl_mean, q_icl_mean
    )

    # Create simplified evaluator with pre-computed values
    evaluator = MicrophysicsEvaluatorSimple(
        scheme, mp, tps, ρ,
        T_mean, q_lcl_mean, q_icl_mean, q_rai, q_sno,
        precomp,
        args,  # Tuple will be inferred
    )

    # Integrate (now with smaller evaluator struct)
    return integrate_over_sgs(evaluator, SG_quad, q_tot_mean, T_mean, q′q′, T′T′, corr_Tq)
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
    SG_quad::SGSQuadrature,
    cmp, tps,
    ρ, T,
    q_tot, q_liq, n_liq, q_rai, n_rai,
)
    # Fall back to grid-mean evaluation for non-GridMeanSGS distributions
    return BMT.bulk_microphysics_tendencies(
        BMT.Microphysics2Moment(),
        cmp, tps,
        ρ, T,
        q_tot, q_liq, n_liq, q_rai, n_rai,
    )
end

"""
    microphysics_tendencies_quadrature_2m(::GridMeanSGS, ...)

Direct GridMeanSGS dispatch for 2M: evaluates BMT at grid mean.
"""
@inline function microphysics_tendencies_quadrature_2m(
    ::GridMeanSGS,
    cmp, tps,
    ρ, T,
    q_tot, q_liq, n_liq, q_rai, n_rai,
)
    return BMT.bulk_microphysics_tendencies(
        BMT.Microphysics2Moment(),
        cmp, tps,
        ρ, T,
        q_tot, q_liq, n_liq, q_rai, n_rai,
    )
end
