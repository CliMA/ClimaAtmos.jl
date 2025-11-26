# A set of wrappers for using CloudMicrophysics.jl functions inside EDMFX loops

import Thermodynamics as TD
import CloudMicrophysics.ThermodynamicsInterface as CMTDI
import CloudMicrophysics.Microphysics0M as CM0
import CloudMicrophysics.Microphysics1M as CM1
import CloudMicrophysics.Microphysics2M as CM2
import CloudMicrophysics.MicrophysicsNonEq as CMNe
import CloudMicrophysics.AerosolModel as CMAM
import CloudMicrophysics.AerosolActivation as CMAA
import CloudMicrophysics.Parameters as CMP

# Define some aliases and functions to make the code more readable
const Tₐ = TD.air_temperature
const Pₐ = TD.air_pressure
const PP = TD.PhasePartition
const qᵥ = TD.vapor_specific_humidity

# Clip any specific humidity
function clip(q)
    FT = eltype(q)
    return max(FT(0), q)
end

import ClimaCore.MatrixFields: @name, FieldName

function terminal_velocity_func_1M(cmc, cmp, q_name)
    vₗ(ρ, q) = CMNe.terminal_velocity(cmc.liquid, cmc.Ch2022.rain, ρ, q)
    vᵢ(ρ, q) = CMNe.terminal_velocity(cmc.ice, cmc.Ch2022.small_ice, ρ, q)
    vᵣ(ρ, q) = CM1.terminal_velocity(cmp.pr, cmp.tv.rain, ρ, q)
    vₛ(ρ, q) = CM1.terminal_velocity(cmp.ps, cmp.tv.snow, ρ, q)
    q_name == @name(q_liq) && return vₗ
    q_name == @name(q_ice) && return vᵢ
    q_name == @name(q_rai) && return vᵣ
    q_name == @name(q_sno) && return vₛ
    throw(ArgumentError("Invalid q_name"))
end

function _terminal_velocity_func_2M(cm2p, cmc, cm1p, q_name)
    # TODO: sedimentation of ice is based on the 1M scheme
    # TODO: sedimentation of snow is based on the 1M scheme
    vₗ(ρ, q, n) = CM2.cloud_terminal_velocity(cm2p.sb.pdf_c, cm2p.ctv, q, ρ, n)
    vᵢ(ρ, q, _) = terminal_velocity_func_1M(cmc, cm1p, q_name)(ρ, q)
    vᵣ(ρ, q, n) = CM2.rain_terminal_velocity(cm2p.sb, cm2p.rtv, q, ρ, n)
    vₛ(ρ, q, _) = terminal_velocity_func_1M(cmc, cm1p, q_name)(ρ, q)
    q_name == @name(q_liq) && return vₗ
    q_name == @name(q_ice) && return vᵢ
    q_name == @name(q_rai) && return vᵣ
    q_name == @name(q_sno) && return vₛ
    throw(ArgumentError("Invalid q_name"))
end
terminal_velocity_mass_func_2M(cm2p, cmc, cm1p, q_name) =
    last ∘ _terminal_velocity_func_2M(cm2p, cmc, cm1p, q_name)
terminal_velocity_number_func_2M(cm2p, cmc, cm1p, q_name) =
    first ∘ _terminal_velocity_func_2M(cm2p, cmc, cm1p, q_name)

# Helper function to compute the limit of the tendency in the traingle limiter.
# The limit is defined as the available amont / n times model time step.
function limit(q, dt, n::Int)
    FT = eltype(q)
    return q / FT(dt) / n
end

function moisture_fixer(q, qᵥ, dt)
    FT = eltype(q)
    return triangle_inequality_limiter(
        -min(FT(0), q / FT(dt)),
        limit(qᵥ, FT(dt), 5),
        FT(0),
    )
end

"""
    cloud_sources(cm_params, thp, qₜ, qₗ, qᵢ, qᵣ, qₛ, ρ, Tₐ, dt)

 - cm_params - CloudMicrophysics parameters struct for cloud water or ice condensate
 - thp - Thermodynamics parameters struct
 - qₜ - total specific humidity
 - qₗ - liquid specific humidity
 - qᵢ - ice specific humidity
 - qᵣ - rain specific humidity
 - qₛ - snow specific humidity
 - ρ - air density
 - Tₐ - air temperature
 - dt - model time step

Returns the condensation/evaporation or deposition/sublimation rate for
non-equilibrium Morrison and Milbrandt 2015 cloud formation.
"""
function cloud_sources(
    cm_params::CMP.CloudLiquid{FT},
    thp,
    qₜ,
    qₗ,
    qᵢ,
    qᵣ,
    qₛ,
    ρ,
    T,
    dt,
) where {FT}

    qᵥ = qₜ - qₗ - qᵢ - qᵣ - qₛ
    qₛₗ = TD.q_vap_from_p_vap(
        thp,
        T,
        ρ,
        TD.saturation_vapor_pressure(thp, T, TD.Liquid()),
    )

    if qᵥ + qₗ > FT(0)
        S = CMNe.conv_q_vap_to_q_lcl_icl_MM2015(
            cm_params,
            thp,
            qₜ,
            qₗ,
            qᵢ,
            qᵣ,
            qₛ,
            ρ,
            T,
        )
    else
        S = FT(0)
    end

    return ifelse(
        S > FT(0),
        triangle_inequality_limiter(S, limit(qᵥ - qₛₗ, dt, 2), limit(qₗ, dt, 2)),
        -triangle_inequality_limiter(abs(S), limit(qₗ, dt, 2), limit(qᵥ - qₛₗ, dt, 2)),
    )
end
function cloud_sources(
    cm_params::CMP.CloudIce{FT},
    thp,
    qₜ,
    qₗ,
    qᵢ,
    qᵣ,
    qₛ,
    ρ,
    T,
    dt,
) where {FT}

    qᵥ = qₜ - qₗ - qᵢ - qᵣ - qₛ

    qₛᵢ = TD.q_vap_from_p_vap(
        thp,
        T,
        ρ,
        TD.saturation_vapor_pressure(thp, T, TD.Ice()),
    )

    if qᵥ + qᵢ > FT(0)
        S = CMNe.conv_q_vap_to_q_lcl_icl_MM2015(
            cm_params,
            thp,
            qₜ,
            qₗ,
            qᵢ,
            qᵣ,
            qₛ,
            ρ,
            T,
        )
    else
        S = FT(0)
    end

    # Additional condition to avoid creating ice in conditions above freezing
    # Representing the lack of INPs in warm temperatures
    if T > thp.T_freeze && S > FT(0)
        S = FT(0)
    end

    return ifelse(
        S > FT(0),
        triangle_inequality_limiter(S, limit(qᵥ - qₛᵢ, dt, 2), limit(qᵢ, dt, 2)),
        -triangle_inequality_limiter(abs(S), limit(qᵢ, dt, 2), limit(qᵥ - qₛᵢ, dt, 2)),
    )
end

"""
    q_tot_0M_precipitation_sources(thp, cmp, dt, qₜ, ts)

 - thp, cmp - structs with thermodynamic and microphysics parameters
 - dt - model time step
 - qₜ - total water specific humidity
 - ts - thermodynamic state (see Thermodynamics.jl package for details)

Returns the qₜ source term due to precipitation formation
defined as Δm_tot / (m_dry + m_tot) for the 0-moment scheme
"""
function q_tot_0M_precipitation_sources(thp, cmp::CMP.Parameters0M, dt, qₜ, ts)
    FT = eltype(qₜ)
    return -triangle_inequality_limiter(
        -CM0.remove_precipitation(cmp, PP(thp, ts)),
        qₜ / FT(dt),
    )
end

"""
    e_tot_0M_precipitation_sources_helper(thp, ts, Φ)

 - thp - set with thermodynamics parameters
 - ts - thermodynamic state (see td package for details)
 - Φ - geopotential

Returns the total energy source term multiplier from precipitation formation
for the 0-moment scheme
"""
function e_tot_0M_precipitation_sources_helper(thp, ts, Φ)

    λ = TD.liquid_fraction(thp, ts)
    Iₗ = TD.internal_energy_liquid(thp, ts)
    Iᵢ = TD.internal_energy_ice(thp, ts)

    return λ * Iₗ + (1 - λ) * Iᵢ + Φ
end

"""
    compute_precipitation_sources!(Sᵖ, Sᵖ_snow, Sqₗᵖ, Sqᵢᵖ, Sqᵣᵖ, Sqₛᵖ, ρ, qₜ, qₗ, qᵢ, qᵣ, qₛ, ts, dt, mp, thp)

 - Sᵖ, Sᵖ_snow - temporary containters to help compute precipitation source terms
 - Sqₗᵖ, Sqᵢᵖ, Sqᵣᵖ, Sqₛᵖ - cached storage for precipitation source terms
 - ρ - air density
 - qₜ, qₗ, qᵢ, qᵣ, qₛ - total water, liquid and ice, rain and snow specific humidity
 - ts - thermodynamic state (see td package for details)
 - dt - model time step
 - thp, cmp - structs with thermodynamic and microphysics parameters

Returns the q source terms due to precipitation formation from the 1-moment scheme.
The specific humidity source terms are defined as Δmᵢ / (m_dry + m_tot)
where i stands for total, rain or snow.
Also returns the total energy source term due to the microphysics processes.
"""
function compute_precipitation_sources!(
    Sᵖ,
    Sᵖ_snow,
    Sqₗᵖ,
    Sqᵢᵖ,
    Sqᵣᵖ,
    Sqₛᵖ,
    ρ,
    qₜ,
    qₗ,
    qᵢ,
    qᵣ,
    qₛ,
    ts,
    dt,
    mp,
    thp,
)
    FT = eltype(thp)
    @. Sqₗᵖ = FT(0)
    @. Sqᵢᵖ = FT(0)
    @. Sqᵣᵖ = FT(0)
    @. Sqₛᵖ = FT(0)

    #! format: off
    # rain autoconversion: q_liq -> q_rain
    @. Sᵖ = ifelse(
        mp.Ndp <= 0,
        CM1.conv_q_lcl_to_q_rai(mp.pr.acnv1M, qₗ, true),
        CM2.conv_q_lcl_to_q_rai(mp.var, qₗ, ρ, mp.Ndp),
    )
    @. Sᵖ = triangle_inequality_limiter(Sᵖ, limit(qₗ, dt, 5), limit(qᵣ, dt, 5))
    @. Sqₗᵖ -= Sᵖ
    @. Sqᵣᵖ += Sᵖ

    # snow autoconversion assuming no supersaturation: q_ice -> q_snow
    @. Sᵖ = triangle_inequality_limiter(
        CM1.conv_q_icl_to_q_sno_no_supersat(mp.ps.acnv1M, qᵢ, true),
        limit(qᵢ, dt, 5),
        limit(qₛ, dt, 5),
    )
    @. Sqᵢᵖ -= Sᵖ
    @. Sqₛᵖ += Sᵖ

    # accretion: q_liq + q_rain -> q_rain
    @. Sᵖ = triangle_inequality_limiter(
        CM1.accretion(mp.cl, mp.pr, mp.tv.rain, mp.ce, qₗ, qᵣ, ρ),
        limit(qₗ, dt, 5),
        limit(qᵣ, dt, 5),
    )
    @. Sqₗᵖ -= Sᵖ
    @. Sqᵣᵖ += Sᵖ

    # accretion: q_ice + q_snow -> q_snow
    @. Sᵖ = triangle_inequality_limiter(
        CM1.accretion(mp.ci, mp.ps, mp.tv.snow, mp.ce, qᵢ, qₛ, ρ),
        limit(qᵢ, dt, 5),
        limit(qₛ, dt, 5),
    )
    @. Sqᵢᵖ -= Sᵖ
    @. Sqₛᵖ += Sᵖ

    # accretion: q_liq + q_sno -> q_sno or q_rai
    # sink of cloud water via accretion cloud water + snow
    @. Sᵖ = triangle_inequality_limiter(
        CM1.accretion(mp.cl, mp.ps, mp.tv.snow, mp.ce, qₗ, qₛ, ρ),
        limit(qₗ, dt, 5),
    )
    # if T < T_freeze cloud droplets freeze to become snow
    # else the snow melts and both cloud water and snow become rain
    α(thp, ts) = TD.Parameters.cv_l(thp) / TD.latent_heat_fusion(thp, ts) * (Tₐ(thp, ts) - mp.ps.T_freeze)
    @. Sᵖ_snow = ifelse(
        Tₐ(thp, ts) < mp.ps.T_freeze,
        Sᵖ,
        FT(-1) * triangle_inequality_limiter(Sᵖ * α(thp, ts), limit(qₛ, dt, 5)),
    )
    @. Sqₛᵖ += Sᵖ_snow
    @. Sqₗᵖ -= Sᵖ
    @. Sqᵣᵖ += ifelse(Tₐ(thp, ts) < mp.ps.T_freeze, FT(0), Sᵖ - Sᵖ_snow)

    # accretion: q_ice + q_rai -> q_sno
    @. Sᵖ = triangle_inequality_limiter(
        CM1.accretion(mp.ci, mp.pr, mp.tv.rain, mp.ce, qᵢ, qᵣ, ρ),
        limit(qᵢ, dt, 5),
        limit(qₛ, dt, 5),
    )
    @. Sqᵢᵖ -= Sᵖ
    @. Sqₛᵖ += Sᵖ
    # sink of rain via accretion cloud ice - rain
    @. Sᵖ = triangle_inequality_limiter(
        CM1.accretion_rain_sink(mp.pr, mp.ci, mp.tv.rain, mp.ce, qᵢ, qᵣ, ρ),
        limit(qᵣ, dt, 5),
        limit(qₛ, dt, 5),
    )
    @. Sqᵣᵖ -= Sᵖ
    @. Sqₛᵖ += Sᵖ

    # accretion: q_rai + q_sno -> q_rai or q_sno
    @. Sᵖ = ifelse(
        Tₐ(thp, ts) < mp.ps.T_freeze,
        triangle_inequality_limiter(
            CM1.accretion_snow_rain(mp.ps, mp.pr, mp.tv.rain, mp.tv.snow, mp.ce, qₛ, qᵣ, ρ),
            limit(qᵣ, dt, 5),
            limit(qₛ, dt, 5),
        ),
        -triangle_inequality_limiter(
            CM1.accretion_snow_rain(mp.pr, mp.ps, mp.tv.snow, mp.tv.rain, mp.ce, qᵣ, qₛ, ρ),
            limit(qₛ, dt, 5),
            limit(qᵣ, dt, 5),
        ),
    )
    @. Sqₛᵖ += Sᵖ
    @. Sqᵣᵖ -= Sᵖ
    #! format: on
end

"""
    compute_precipitation_sinks!(Sᵖ, Sqᵣᵖ, Sqₛᵖ, ρ, qₜ, qₗ, qᵢ, qᵣ, qₛ, ts, dt, mp, thp)

 - Sᵖ - a temporary containter to help compute precipitation source terms
 - Sqᵣᵖ, Sqₛᵖ - cached storage for precipitation source terms
 - ρ - air density
 - qₜ, qₗ, qᵢ, qᵣ, qₛ - total water, cloud liquid and ice, rain and snow specific humidities
 - ts - thermodynamic state (see td package for details)
 - dt - model time step
 - thp, cmp - structs with thermodynamic and microphysics parameters

Returns the q source terms due to precipitation sinks from the 1-moment scheme.
The specific humidity source terms are defined as Δmᵢ / (m_dry + m_tot)
where i stands for total, rain or snow.
Also returns the total energy source term due to the microphysics processes.
"""
function compute_precipitation_sinks!(
    Sᵖ,
    Sqᵣᵖ,
    Sqₛᵖ,
    ρ,
    qₜ,
    qₗ,
    qᵢ,
    qᵣ,
    qₛ,
    ts,
    dt,
    mp,
    thp,
)
    FT = eltype(thp)
    sps = (mp.ps, mp.tv.snow, mp.aps, thp)
    rps = (mp.pr, mp.tv.rain, mp.aps, thp)

    #! format: off
    # evaporation: q_rai -> q_vap
    @. Sᵖ = -triangle_inequality_limiter(
        -CM1.evaporation_sublimation(rps..., qₜ, qₗ, qᵢ, qᵣ, qₛ, ρ, Tₐ(thp, ts)),
        limit(qᵣ, dt, 5),
        limit(qᵥ(thp, ts), dt, 5),
    )
    @. Sqᵣᵖ += Sᵖ

    # melting: q_sno -> q_rai
    @. Sᵖ = triangle_inequality_limiter(
        CM1.snow_melt(sps..., qₛ, ρ, Tₐ(thp, ts)),
        limit(qₛ, dt, 5),
        limit(qᵣ, dt, 5),
    )
    @. Sqᵣᵖ += Sᵖ
    @. Sqₛᵖ -= Sᵖ

    # deposition/sublimation: q_vap <-> q_sno
    @. Sᵖ = CM1.evaporation_sublimation(sps..., qₜ, qₗ, qᵢ, qᵣ, qₛ, ρ, Tₐ(thp, ts))
    @. Sᵖ = ifelse(
        Sᵖ > FT(0),
        triangle_inequality_limiter(Sᵖ, limit(qᵥ(thp, ts), dt, 5), limit(qₛ, dt, 5)),
        -triangle_inequality_limiter(FT(-1) * Sᵖ, limit(qₛ, dt, 5), limit(qᵥ(thp, ts), dt, 5)),
    )
    @. Sqₛᵖ += Sᵖ
    #! format: on
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
        ts,
        dt,
    )

Computes the source term for cloud droplet number concentration per mass due to aerosol activation,
based on the Abdul-Razzak and Ghan (2000) parameterization.

This function estimates the number of aerosols activated into cloud droplets per mass of air per second
from a bi-modal aerosol distribution (sea salt and sulfate), given local supersaturation and vertical
velocity. The result is returned as a tendency (per second) of liquid droplet number concentration.

# Arguments
- `seasalt_num`: Number concentration per mass of sea salt aerosols [kg⁻¹].
- `seasalt_mean_radius`: Mean dry radius of sea salt aerosol mode [m].
- `sulfate_num`: Number concentration per mass of sulfate aerosols [kg⁻¹].
- `qₜ`, `qₗ`, `qᵢ` - total water, liquid (cloud liquid and rain) and ice (cloud ice and snow) specific humidity
- `nₗ` - liquid (cloud liquid and rain) number concentration per mass [kg⁻¹]
- `ρ`: Air density [kg/m³].
- `w`: Vertical velocity [m/s].
- `cmp`: Microphysics parameters
- `thermo_params`: Thermodynamic parameters for computing saturation, pressure, temperature, etc.
- `ts`: Thermodynamic state (e.g., prognostic variables) used for evaluating the phase partition.
- `dt`: Time step (s) over which the activation tendency is applied.

# Returns
- Tendency of cloud liquid droplet number concentration per mass of air due to aerosol activation [m⁻³/s].
"""
function aerosol_activation_sources(
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
    ts,
    dt,
)

    FT = eltype(nₗ)
    air_params = cmp.aps
    arg_params = cmp.arg
    aerosol_params = cmp.aerosol
    T = Tₐ(thermo_params, ts)
    p = Pₐ(thermo_params, ts)
    S = CMTDI.supersaturation_over_liquid(thermo_params, qₜ, qₗ, qᵢ, ρ, T)
    n_aer = seasalt_num + sulfate_num
    if S < 0 || n_aer < eps(FT)
        return FT(0)
    end

    seasalt_mode = CMAM.Mode_κ(
        seasalt_mean_radius,
        aerosol_params.seasalt_std,
        max(0, seasalt_num) * ρ,
        (FT(1),),
        (FT(1),),
        (FT(0),),
        (aerosol_params.seasalt_kappa,),
    )
    sulfate_mode = CMAM.Mode_κ(
        aerosol_params.sulfate_radius,
        aerosol_params.sulfate_std,
        max(0, sulfate_num) * ρ,
        (FT(1),),
        (FT(1),),
        (FT(0),),
        (aerosol_params.sulfate_kappa,),
    )

    aerosol_dist = CMAM.AerosolDistribution((seasalt_mode, sulfate_mode))

    args = (
        arg_params,
        aerosol_dist,
        air_params,
        thermo_params,
        T,
        p,
        max(0, w),
        qₜ,
        qₗ,
        qᵢ,
        ρ * nₗ,
        FT(0),
    ) #assuming no ice particles because we don't track n_ice for now
    S_max = CMAA.max_supersaturation(args...)
    n_act = CMAA.total_N_activated(args...) / ρ

    return ifelse(
        S_max < S || isnan(n_act) || n_act < nₗ,
        FT(0),
        (n_act - nₗ) / FT(dt),
    )
end

"""
    compute_warm_precipitation_sources_2M!(Sᵖ, S₂ᵖ, Snₗᵖ, Snᵣᵖ, Sqₗᵖ, Sqᵣᵖ, ρ, nₗ, nᵣ, qₜ, qₗ, qᵢ, qᵣ, qₛ, ts, dt, sb, thp)

 - `Sᵖ`, `S₂ᵖ` - temporary containters to help compute precipitation source terms
 - `Snₗᵖ`, `Snᵣᵖ`, `Sqₗᵖ`, `Sqᵣᵖ` - cached storage for precipitation source terms
 - `ρ` - air density
 - `nₗ`, `nᵣ` - cloud liquid and rain number concentration per mass [1 / kg of moist air]
 - `qₜ`, `qₗ`, `qᵢ`, `qᵣ`, `qₛ` - total water, cloud liquid, cloud ice, rain and snow specific humidity
 - `ts` - thermodynamic state (see td package for details)
 - `dt` - model time step
 - `thp`, `mp` - structs with thermodynamic and microphysics parameters

Computes precipitation number and mass sources due to warm precipitation processes based on the 2-moment
[Seifert and Beheng (2006) scheme](https://clima.github.io/CloudMicrophysics.jl/dev/Microphysics2M/).
"""
function compute_warm_precipitation_sources_2M!(
    Sᵖ,
    S₂ᵖ,
    Snₗᵖ,
    Snᵣᵖ,
    Sqₗᵖ,
    Sqᵣᵖ,
    ρ,
    nₗ,
    nᵣ,
    qₜ,
    qₗ,
    qᵢ,
    qᵣ,
    qₛ,
    ts,
    dt,
    mp,
    thp,
)

    FT = eltype(thp)
    @. Snₗᵖ = FT(0)
    @. Snᵣᵖ = FT(0)
    @. Sqₗᵖ = FT(0)
    @. Sqᵣᵖ = FT(0)

    # auto-conversion (mass)
    @. Sᵖ = triangle_inequality_limiter(
        CM2.autoconversion(
            mp.sb.acnv,
            mp.sb.pdf_c,
            qₗ,
            qᵣ,
            ρ,
            ρ * nₗ,
        ).dq_rai_dt,
        limit(qₗ, dt, 5), # cap rate to at most 20% of qₗ per timestep to ensure stability
        limit(qᵣ, dt, 5),
    )
    @. Sqₗᵖ -= Sᵖ
    @. Sqᵣᵖ += Sᵖ

    # auto-conversion (number) and liquid self-collection
    @. Sᵖ = triangle_inequality_limiter(
        CM2.autoconversion(
            mp.sb.acnv,
            mp.sb.pdf_c,
            qₗ,
            qᵣ,
            ρ,
            ρ * nₗ,
        ).dN_lcl_dt / ρ,
        limit(nₗ, dt, 10),
    )
    # triangle_inequality_limiter assumes positive rates and limits.
    # Here the physical rate is negative (a sink), so we negate it before passing,
    # and negate the result again to preserve the original sign.
    @. S₂ᵖ =
        -triangle_inequality_limiter(
            -CM2.cloud_liquid_self_collection(mp.sb.acnv, mp.sb.pdf_c, qₗ, ρ, Sᵖ) / ρ,
            limit(nₗ / ρ, dt, 5),
        )
    @. Snₗᵖ += Sᵖ
    @. Snₗᵖ += S₂ᵖ
    @. Snᵣᵖ -= 0.5 * Sᵖ # each raindrop forms from two cloud particles → factor 0.5

    # rain self-collection and breakup
    @. Sᵖ =
        -triangle_inequality_limiter(
            -CM2.rain_self_collection(mp.sb.pdf_r, mp.sb.self, qᵣ, ρ, ρ * nᵣ) /
            ρ,
            limit(nᵣ, dt, 5),
        )
    @. S₂ᵖ = triangle_inequality_limiter(
        CM2.rain_breakup(mp.sb.pdf_r, mp.sb.brek, qᵣ, ρ, ρ * nᵣ, Sᵖ) / ρ,
        limit(nᵣ, dt, 5),
    )
    @. Snᵣᵖ += Sᵖ
    @. Snᵣᵖ += S₂ᵖ

    # accretion (mass)
    @. Sᵖ = triangle_inequality_limiter(
        CM2.accretion(mp.sb, qₗ, qᵣ, ρ, ρ * nₗ).dq_rai_dt,
        limit(qₗ, dt, 5),
        limit(qᵣ, dt, 5),
    )
    @. Sqₗᵖ -= Sᵖ
    @. Sqᵣᵖ += Sᵖ

    # accretion (number)
    @. Sᵖ =
        -triangle_inequality_limiter(
            -CM2.accretion(mp.sb, qₗ, qᵣ, ρ, ρ * nₗ).dN_lcl_dt / ρ,
            limit(nₗ, dt, 5),
        )
    @. Snₗᵖ += Sᵖ

    # evaporation (mass)
    @. Sᵖ =
        -triangle_inequality_limiter(
            -CM2.rain_evaporation(
                mp.sb,
                mp.aps,
                thp,
                qₜ,
                qₗ,
                qᵢ,
                qᵣ,
                qₛ,
                ρ,
                ρ * nᵣ,
                Tₐ(thp, ts),
            ).evap_rate_1,
            limit(qᵣ, dt, 5),
            limit(qᵥ(thp, ts), dt, 5),
        )
    @. Sqᵣᵖ += Sᵖ

    # evaporation (number)
    @. Sᵖ =
        -triangle_inequality_limiter(
            -CM2.rain_evaporation(
                mp.sb,
                mp.aps,
                thp,
                qₜ,
                qₗ,
                qᵢ,
                qᵣ,
                qₛ,
                ρ,
                ρ * nᵣ,
                Tₐ(thp, ts),
            ).evap_rate_0 / ρ,
            limit(nᵣ, dt, 5),
        )
    @. Snᵣᵖ += Sᵖ

    # cloud liquid number adjustment for mass limits
    # TODO: Once CCN number becomes a prognostic variable, these number adjustment tendencies
    #       should be linked to it. Any increase in droplet number (source here) would imply
    #       a corresponding sink in CCN, and vice versa.
    @. Sᵖ = CM2.number_increase_for_mass_limit(
        mp.sb.numadj,
        mp.sb.pdf_c.xc_max,
        qₗ,
        ρ,
        ρ * nₗ,
    )
    @. S₂ᵖ =
        -triangle_inequality_limiter(
            -CM2.number_decrease_for_mass_limit(
                mp.sb.numadj,
                mp.sb.pdf_c.xc_min,
                qₗ,
                ρ,
                ρ * nₗ,
            ),
            limit(nₗ, dt, 5),
        )
    @. Snₗᵖ = Sᵖ + S₂ᵖ
    # rain number adjustment for mass limits
    @. Sᵖ = CM2.number_increase_for_mass_limit(
        mp.sb.numadj,
        mp.sb.pdf_r.xr_max,
        qᵣ,
        ρ,
        ρ * nᵣ,
    )
    @. S₂ᵖ =
        -triangle_inequality_limiter(
            -CM2.number_decrease_for_mass_limit(
                mp.sb.numadj,
                mp.sb.pdf_r.xr_min,
                qᵣ,
                ρ,
                ρ * nᵣ,
            ),
            limit(nᵣ, dt, 5),
        )
    @. Snᵣᵖ += Sᵖ + S₂ᵖ

end

function compute_cold_precipitation_sources_P3!(
    ᶜScoll,         # NamedTuple-valued Field with P3 liquid-ice collision sources
    params_2mp3,    # Parameters for 2M and P3 schemes, see `get_microphysics_2m_p3_parameters`
    thermo_params,  # An instance of `Thermodynamics.Parameters.ThermodynamicsParameters`
    ᶜY_reduced,     # A reduced set of prognostic variables needed for P3 sources
    ᶜts,            # Thermodynamic state
    ᶜlogλ,          # Logarithm of the P3 distribution slope parameter
)

    (; warm, cold) = params_2mp3
    (; ρ, ρq_liq, ρn_liq, ρq_rai, ρn_rai, ρq_ice, ρn_ice, ρq_rim, ρb_rim) = ᶜY_reduced

    ᶜF_rim = @. lazy(ρq_rim / ρq_ice)
    ᶜρ_rim = @. lazy(ρq_rim / ρb_rim)

    @. ᶜScoll = CMP3.bulk_liquid_ice_collision_sources(cold.params, ᶜlogλ,
        ρq_ice, max(0, ρn_ice),
        ᶜF_rim, ᶜρ_rim,
        warm.sb.pdf_c, warm.sb.pdf_r,
        ρq_liq, ρn_liq, ρq_rai, ρn_rai,
        warm.aps, thermo_params, (cold.velocity_params,),
        ρ, Tₐ(thermo_params, ᶜts),
    )

    return nothing
end
