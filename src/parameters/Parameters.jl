module Parameters

import ClimaCore as CC
import Insolation.Parameters as IP
import Thermodynamics as TD
import CloudMicrophysics as CM

abstract type AbstractClimaAtmosParameters end
const ACAP = AbstractClimaAtmosParameters

Base.broadcastable(param_set::ACAP) = tuple(param_set)

Base.@kwdef struct ClimaAtmosParameters{FT, TP, RP, IP, MPP, SFP, TCP} <: ACAP
    thermodynamics_params::TP
    rrtmgp_params::RP
    insolation_params::IP
    microphysics_params::MPP
    surface_fluxes_params::SFP
    turbconv_params::TCP
    Omega::FT
    f_plane_coriolis_frequency::FT
    planet_radius::FT
    astro_unit::FT
    entr_tau::FT
    entr_coeff::FT
    detr_coeff::FT
    C_E::FT
    # Held Suarez
    ΔT_y_dry::FT
    ΔT_y_wet::FT
    σ_b::FT
    T_equator_dry::FT
    T_equator_wet::FT
    T_min_hs::FT
    Δθ_z::FT
    # Sponge
    alpha_rayleigh_w::FT
    alpha_rayleigh_uh::FT
    zd_viscous::FT
    zd_rayleigh::FT
    kappa_2_sponge::FT
end

Base.eltype(::ClimaAtmosParameters{FT}) where {FT} = FT

# Forward Thermodynamics parameters
for var in fieldnames(TD.Parameters.ThermodynamicsParameters)
    @eval $var(ps::ACAP) = TD.Parameters.$var(thermodynamics_params(ps))
end
# Thermodynamics derived parameters
for var in [:molmass_ratio, :R_d, :R_v, :cp_d, :cv_v, :cv_l, :cv_d]
    @eval $var(ps::ACAP) = TD.Parameters.$var(thermodynamics_params(ps))
end

# Forwarding CloudMicrophysics parameters
ρ_cloud_liq(ps::ACAP) = CM.Parameters.ρ_cloud_liq(microphysics_params(ps))

# Insolation parameters
day(ps::ACAP) = IP.day(insolation_params(ps))
tot_solar_irrad(ps::ACAP) = IP.tot_solar_irrad(insolation_params(ps))

# Define parameters as functions
for var in fieldnames(ClimaAtmosParameters)
    @eval $var(ps::ACAP) = ps.$var
end

end
