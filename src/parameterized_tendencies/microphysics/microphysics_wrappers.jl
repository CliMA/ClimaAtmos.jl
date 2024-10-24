# A set of wrappers for using CloudMicrophysics.jl functions inside EDMFX loops

import Thermodynamics as TD
import CloudMicrophysics.Microphysics0M as CM0
import CloudMicrophysics.Microphysics1M as CM1
import CloudMicrophysics.Microphysics2M as CM2
import CloudMicrophysics.MicrophysicsNonEq as CMNe
import CloudMicrophysics.Parameters as CMP
using CUDA
using NVTX

# define some aliases and functions to make the code more readable
const Iₗ = TD.internal_energy_liquid
const Iᵢ = TD.internal_energy_ice
const Lf = TD.latent_heat_fusion
const Tₐ = TD.air_temperature
const PP = TD.PhasePartition
const qᵥ = TD.vapor_specific_humidity
qₜ(thp, ts) = TD.PhasePartition(thp, ts).tot
qₗ(thp, ts) = TD.PhasePartition(thp, ts).liq
qᵢ(thp, ts) = TD.PhasePartition(thp, ts).ice
cᵥₗ(thp) = TD.Parameters.cv_l(thp)
cᵥᵢ(thp) = TD.Parameters.cv_i(thp)

# helper function to limit the tendency
function limit(q::FT, dt::FT, n::Int) where {FT}
    return q / dt / n
end

"""
    cloud_sources(cm_params, thp, ts, dt)

 - cm_params - CloudMicrophysics parameters struct for cloud water or ice condensate
 - thp - Thermodynamics parameters struct
 - ts - thermodynamics state
 - dt - model time step

Returns the condensation/evaporation or deposition/sublimation rate for
non-equilibrium Morrison and Milbrandt 2015 cloud formation.
"""
function cloud_sources(cm_params::CMP.CloudLiquid{FT}, thp, ts, dt) where {FT}

    q = TD.PhasePartition(thp, ts)
    ρ = TD.air_density(thp, ts)

    S = CMNe.conv_q_vap_to_q_liq_ice_MM2015(cm_params, thp, q, ρ, Tₐ(thp, ts))

    # keeping the same limiter for now
    return ifelse(
        S > FT(0),
        min(S, limit(qᵥ(thp, ts), dt, 2)),
        -min(abs(S), limit(qₗ(thp, ts), dt, 2)),
    )
end
function cloud_sources(cm_params::CMP.CloudIce{FT}, thp, ts, dt) where {FT}

    q = TD.PhasePartition(thp, ts)
    ρ = TD.air_density(thp, ts)

    S = CMNe.conv_q_vap_to_q_liq_ice_MM2015(cm_params, thp, q, ρ, Tₐ(thp, ts))

    # keeping the same limiter for now
    return ifelse(
        S > FT(0),
        min(S, limit(qᵥ(thp, ts), dt, 2)),
        -min(abs(S), limit(qᵢ(thp, ts), dt, 2)),
    )
end

"""
    q_tot_precipitation_sources(precip_model, thp, cmp, dt, qₜ, ts)

 - precip_model - a type for precipitation scheme choice
 - thp, cmp - structs with thermodynamic and microphysics parameters
 - dt - model time step
 - qₜ - total water specific humidity
 - ts - thermodynamic state (see Thermodynamics.jl package for details)

Returns the qₜ source term due to precipitation formation
defined as Δm_tot / (m_dry + m_tot)
"""
function q_tot_precipitation_sources(
    ::NoPrecipitation,
    thp,
    cmp,
    dt,
    qₜ::FT,
    ts,
) where {FT <: Real}
    return FT(0)
end
function q_tot_precipitation_sources(
    ::Microphysics0Moment,
    thp,
    cmp,
    dt,
    qₜ::FT,
    ts,
) where {FT <: Real}
    return -min(max(qₜ, 0) / dt, -CM0.remove_precipitation(cmp, PP(thp, ts)))
end

"""
    e_tot_0M_precipitation_sources_helper(thp, ts, Φ)

 - thp - set with thermodynamics parameters
 - ts - thermodynamic state (see td package for details)
 - Φ - geopotential

Returns the total energy source term multiplier from precipitation formation
for the 0-moment scheme
"""
function e_tot_0M_precipitation_sources_helper(
    thp,
    ts,
    Φ::FT,
) where {FT <: Real}

    λ::FT = TD.liquid_fraction(thp, ts)

    return λ * Iₗ(thp, ts) + (1 - λ) * Iᵢ(thp, ts) + Φ
end

"""
    compute_precipitation_sources!(Sᵖ, Sᵖ_snow, Sqₜᵖ, Sqᵣᵖ, Sqₛᵖ, Seₜᵖ, ρ, qᵣ, qₛ, ts, Φ, dt, mp, thp)

 - Sᵖ, Sᵖ_snow - temporary containters to help compute precipitation source terms
 - Sqₜᵖ, Sqᵣᵖ, Sqₛᵖ, Seₜᵖ - cached storage for precipitation source terms
 - ρ - air density
 - qᵣ, qₛ - precipitation (rain and snow) specific humidity
 - ts - thermodynamic state (see td package for details)
 - Φ - geopotential
 - dt - model time step
 - thp, cmp - structs with thermodynamic and microphysics parameters

Returns the q source terms due to precipitation formation from the 1-moment scheme.
The specific humidity source terms are defined as defined as Δmᵢ / (m_dry + m_tot)
where i stands for total, rain or snow.
Also returns the total energy source term due to the microphysics processes.
"""
NVTX.@annotate function compute_precipitation_sources!(
    Sᵖ,
    Sᵖ_snow,
    Sqₜᵖ,
    Sqᵣᵖ,
    Sqₛᵖ,
    Seₜᵖ,
    ρ,
    qᵣ,
    qₛ,
    ts,
    Φ,
    dt,
    mp,
    thp,
)
    FT = eltype(thp)
    device = ClimaComms.device(Sᵖ)
    dims = Base.size(Fields.field_values(Sᵖ))
    fvargs =
        Fields.field_values.((
            Sᵖ,
            Sᵖ_snow,
            Sqₜᵖ,
            Sqᵣᵖ,
            Sqₛᵖ,
            Seₜᵖ,
            ρ,
            qᵣ,
            qₛ,
            ts,
            Φ,
        ))
    args = (dt, mp, thp)
    pointwise_dispatch(
        device,
        dims,
        compute_precipitation_sources_kernel!,
        fvargs...,
        args...,
    )
    return nothing
end

@inline function compute_precipitation_sources_kernel!(
    Sᵖ,
    Sᵖ_snow,
    Sqₜᵖ,
    Sqᵣᵖ,
    Sqₛᵖ,
    Seₜᵖ,
    ρ,
    qᵣ,
    qₛ,
    ts,
    Φ,
    dt,
    mp,
    thp,
    idx,
)
    FT = eltype(thp)
    @inbounds begin
        # @. Sqₜᵖ = FT(0) should work after fixing
        # https://github.com/CliMA/ClimaCore.jl/issues/1786
        Sqₜᵖ[idx] = zero(Sqₜᵖ[idx])
        Sqᵣᵖ[idx] = zero(Sqᵣᵖ[idx])
        Sqₛᵖ[idx] = zero(Sqₛᵖ[idx])
        Seₜᵖ[idx] = zero(Seₜᵖ[idx])

        #! format: off
        # rain autoconversion: q_liq -> q_rain
        Sᵖ[idx] = ifelse(
            mp.Ndp <= 0,
            CM1.conv_q_liq_to_q_rai(mp.pr.acnv1M, qₗ(thp, ts[idx]), true),
            CM2.conv_q_liq_to_q_rai(mp.var, qₗ(thp, ts[idx]), ρ[idx], mp.Ndp),
        )
        Sᵖ[idx] = min(limit(qₗ(thp, ts[idx]), dt, 5), Sᵖ[idx])
        Sqₜᵖ[idx] -= Sᵖ[idx]
        Sqᵣᵖ[idx] += Sᵖ[idx]
        Seₜᵖ[idx] -= Sᵖ[idx] * (Iₗ(thp, ts[idx]) + Φ[idx])
    
        # snow autoconversion assuming no supersaturation: q_ice -> q_snow
        Sᵖ[idx] = min(
            limit(qᵢ(thp, ts[idx]), dt, 5),
            CM1.conv_q_ice_to_q_sno_no_supersat(mp.ps.acnv1M, qᵢ(thp, ts[idx]), true),
        )
        Sqₜᵖ[idx] -= Sᵖ[idx]
        Sqₛᵖ[idx] += Sᵖ[idx]
        Seₜᵖ[idx] -= Sᵖ[idx] * (Iᵢ(thp, ts[idx]) + Φ[idx])
    
        # accretion: q_liq + q_rain -> q_rain
        Sᵖ[idx] = min(
            limit(qₗ(thp, ts[idx]), dt, 5),
            CM1.accretion(mp.cl, mp.pr, mp.tv.rain, mp.ce, qₗ(thp, ts[idx]), qᵣ[idx], ρ[idx]),
        )
        Sqₜᵖ[idx] -= Sᵖ[idx]
        Sqᵣᵖ[idx] += Sᵖ[idx]
        Seₜᵖ[idx] -= Sᵖ[idx] * (Iₗ(thp, ts[idx]) + Φ[idx])
    
        # accretion: q_ice + q_snow -> q_snow
        Sᵖ[idx] = min(
            limit(qᵢ(thp, ts[idx]), dt, 5),
            CM1.accretion(mp.ci, mp.ps, mp.tv.snow, mp.ce, qᵢ(thp, ts[idx]), qₛ[idx], ρ[idx]),
        )
        Sqₜᵖ[idx] -= Sᵖ[idx]
        Sqₛᵖ[idx] += Sᵖ[idx]
        Seₜᵖ[idx] -= Sᵖ[idx] * (Iᵢ(thp, ts[idx]) + Φ[idx])
    
        # accretion: q_liq + q_sno -> q_sno or q_rai
        # sink of cloud water via accretion cloud water + snow
        Sᵖ[idx] = min(
            limit(qₗ(thp, ts[idx]), dt, 5),
            CM1.accretion(mp.cl, mp.ps, mp.tv.snow, mp.ce, qₗ(thp, ts[idx]), qₛ[idx], ρ[idx]),
        )
        # if T < T_freeze cloud droplets freeze to become snow
        # else the snow melts and both cloud water and snow become rain
        #α(thp, ts[idx]) = cᵥₗ(thp) / Lf(thp, ts[idx]) * (Tₐ(thp, ts[idx]) - mp.ps.T_freeze)
        α(thparg, tsarg) = cᵥₗ(thparg) / Lf(thparg, tsarg) * (Tₐ(thparg, tsarg) - mp.ps.T_freeze)
        Sᵖ_snow[idx] = ifelse(
            Tₐ(thp, ts[idx]) < mp.ps.T_freeze,
            Sᵖ[idx],
            FT(-1) * min(Sᵖ[idx] * α(thp, ts[idx]), limit(qₛ[idx], dt, 5)),
        )
        Sqₛᵖ[idx] += Sᵖ_snow[idx]
        Sqₜᵖ[idx] -= Sᵖ[idx]
        Sqᵣᵖ[idx] += ifelse(Tₐ(thp, ts[idx]) < mp.ps.T_freeze, FT(0), Sᵖ[idx] - Sᵖ_snow[idx])
        Seₜᵖ[idx] -= ifelse(
            Tₐ(thp, ts[idx]) < mp.ps.T_freeze,
            Sᵖ[idx] * (Iᵢ(thp, ts[idx]) + Φ[idx]),
            Sᵖ[idx] * (Iₗ(thp, ts[idx]) + Φ[idx]) - Sᵖ_snow[idx] * (Iₗ(thp, ts[idx]) - Iᵢ(thp, ts[idx])),
        )
    
        # accretion: q_ice + q_rai -> q_sno
        Sᵖ[idx] = min(
            limit(qᵢ(thp, ts[idx]), dt, 5),
            CM1.accretion(mp.ci, mp.pr, mp.tv.rain, mp.ce, qᵢ(thp, ts[idx]), qᵣ[idx], ρ[idx]),
        )
        Sqₜᵖ[idx] -= Sᵖ[idx]
        Sqₛᵖ[idx] += Sᵖ[idx]
        Seₜᵖ[idx] -= Sᵖ[idx] * (Iᵢ(thp, ts[idx]) + Φ[idx])
        # sink of rain via accretion cloud ice - rain
        Sᵖ[idx] = min(
            limit(qᵣ[idx], dt, 5),
            CM1.accretion_rain_sink(mp.pr, mp.ci, mp.tv.rain, mp.ce, qᵢ(thp, ts[idx]), qᵣ[idx], ρ[idx]),
        )
        Sqᵣᵖ[idx] -= Sᵖ[idx]
        Sqₛᵖ[idx] += Sᵖ[idx]
        Seₜᵖ[idx] += Sᵖ[idx] * Lf(thp, ts[idx])
    
        # accretion: q_rai + q_sno -> q_rai or q_sno
        Sᵖ[idx] = ifelse(
            Tₐ(thp, ts[idx]) < mp.ps.T_freeze,
            min(
                limit(qᵣ[idx], dt, 5),
                CM1.accretion_snow_rain(mp.ps, mp.pr, mp.tv.rain, mp.tv.snow, mp.ce, qₛ[idx], qᵣ[idx], ρ[idx]),
            ),
            -min(
                limit(qₛ[idx], dt, 5),
                CM1.accretion_snow_rain(mp.pr, mp.ps, mp.tv.snow, mp.tv.rain, mp.ce, qᵣ[idx], qₛ[idx], ρ[idx]),
            ),
        )
        Sqₛᵖ[idx] += Sᵖ[idx]
        Sqᵣᵖ[idx] -= Sᵖ[idx]
        Seₜᵖ[idx] += Sᵖ[idx] * Lf(thp, ts[idx])
        #! format: on
    end
end

function pointwise_dispatch(
    device::ClimaComms.CUDADevice,
    dims,
    pointwisefn!,
    args...,
)
    NI, NJ, _, NV, NH = dims
    max_threads = 768#512#256
    @assert NI * NJ ≤ max_threads
    nvthreads = Int(fld(max_threads, NI * NJ))
    nvblocks = Int(cld(NV, nvthreads))
    CUDA.@cuda always_inline = true threads = (NI, NJ, nvthreads) blocks =
        (nvblocks, NH) pointwise_cuda_kernel!(pointwisefn!, NV, args...)
    return nothing
end

function pointwise_cuda_kernel!(pointwisefn!, NV, args...)
    (i, j, tv) = threadIdx()
    (bv, bh, _) = blockIdx()
    v = tv + (bv - 1) * blockDim().z
    if v ≤ NV
        idx = CartesianIndex(i, j, 1, v, bh)
        pointwisefn!(args..., idx)
    end
    return nothing
end

"""
    compute_precipitation_heating(Seₜᵖ, ᶜwᵣ, ᶜwₛ, ᶜu, qᵣ, qₛ, ᶜts, thp)

 - Seₜᵖ - cached storage for precipitation energy source terms
 - ᶜwᵣ, ᶜwₛ - rain and snow terminal velocities
 - ᶜu - air velocity
 - qᵣ, qₛ - precipitation (rain and snow) specific humidities
 - ᶜts - thermodynamic state (see td package for details)
 - ᶜ∇T - cached temporary variable to store temperature gradient
 - thp - structs with thermodynamic and microphysics parameters

 Augments the energy source terms with heat exchange between air
 and precipitating species, following eq. 36 from Raymond 2013
 doi:10.1002/jame.20050 and assuming that precipitation has the same
 temperature as the surrounding air.
"""
function compute_precipitation_heating!(
    ᶜSeₜᵖ,
    ᶜwᵣ,
    ᶜwₛ,
    ᶜu,
    ᶜqᵣ,
    ᶜqₛ,
    ᶜts,
    ᶜ∇T,
    thp,
)
    # TODO - at some point we want to switch to assuming that precipitation
    # is at wet bulb temperature

    # compute full temperature gradient
    @. ᶜ∇T = CT123(ᶜgradᵥ(ᶠinterp(Tₐ(thp, ᶜts))))
    @. ᶜ∇T += CT123(gradₕ(Tₐ(thp, ᶜts)))
    # dot product with effective velocity of precipitation
    # (times q and specific heat)
    @. ᶜSeₜᵖ -= dot(ᶜ∇T, (ᶜu - C123(Geometry.WVector(ᶜwᵣ)))) * cᵥₗ(thp) * ᶜqᵣ
    @. ᶜSeₜᵖ -= dot(ᶜ∇T, (ᶜu - C123(Geometry.WVector(ᶜwₛ)))) * cᵥᵢ(thp) * ᶜqₛ
end
"""
    compute_precipitation_sinks!(Sᵖ, Sqₜᵖ, Sqᵣᵖ, Sqₛᵖ, Seₜᵖ, ρ, qᵣ, qₛ, ts, Φ, dt, mp, thp)

 - Sᵖ - a temporary containter to help compute precipitation source terms
 - Sqₜᵖ, Sqᵣᵖ, Sqₛᵖ, Seₜᵖ - cached storage for precipitation source terms
 - ρ - air density
 - qᵣ, qₛ - precipitation (rain and snow) specific humidities
 - ts - thermodynamic state (see td package for details)
 - Φ - geopotential
 - dt - model time step
 - thp, cmp - structs with thermodynamic and microphysics parameters

Returns the q source terms due to precipitation sinks from the 1-moment scheme.
The specific humidity source terms are defined as defined as Δmᵢ / (m_dry + m_tot)
where i stands for total, rain or snow.
Also returns the total energy source term due to the microphysics processes.
"""
function compute_precipitation_sinks!(
    Sᵖ,
    Sqₜᵖ,
    Sqᵣᵖ,
    Sqₛᵖ,
    Seₜᵖ,
    ρ,
    qᵣ,
    qₛ,
    ts,
    Φ,
    dt,
    mp,
    thp,
)
    FT = eltype(Sqₜᵖ)
    sps = (mp.ps, mp.tv.snow, mp.aps, thp)
    rps = (mp.pr, mp.tv.rain, mp.aps, thp)

    #! format: off
    # evaporation: q_rai -> q_vap
    @. Sᵖ = -min(
        limit(qᵣ, dt, 5),
        -CM1.evaporation_sublimation(rps..., PP(thp, ts), qᵣ, ρ, Tₐ(thp, ts)),
    )
    @. Sqₜᵖ -= Sᵖ
    @. Sqᵣᵖ += Sᵖ
    @. Seₜᵖ -= Sᵖ * (Iₗ(thp, ts) + Φ)

    # melting: q_sno -> q_rai
    @. Sᵖ = min(
        limit(qₛ, dt, 5),
        CM1.snow_melt(sps..., qₛ, ρ, Tₐ(thp, ts)),
    )
    @. Sqᵣᵖ += Sᵖ
    @. Sqₛᵖ -= Sᵖ
    @. Seₜᵖ -= Sᵖ * Lf(thp, ts)

    # deposition/sublimation: q_vap <-> q_sno
    @. Sᵖ = CM1.evaporation_sublimation(sps..., PP(thp, ts), qₛ, ρ, Tₐ(thp, ts))
    @. Sᵖ = ifelse(
        Sᵖ > FT(0),
        min(limit(qᵥ(thp, ts), dt, 5), Sᵖ),
        -min(limit(qₛ, dt, 5), FT(-1) * Sᵖ),
    )
    @. Sqₜᵖ -= Sᵖ
    @. Sqₛᵖ += Sᵖ
    @. Seₜᵖ -= Sᵖ * (Iᵢ(thp, ts) + Φ)
    #! format: on
end
