module Parameters

import ClimaCore as CC
import Insolation.Parameters as IP
import Thermodynamics as TD
import CloudMicrophysics as CM

abstract type AbstractClimaAtmosParameters end
const ACAP = AbstractClimaAtmosParameters

Base.broadcastable(param_set::ACAP) = Ref(param_set)

Base.@kwdef struct ClimaAtmosParameters{FT, TP, RP, IP, MPP, SFP, TCP} <: ACAP
    Omega::FT
    f_plane_coriolis_frequency::FT
    planet_radius::FT
    astro_unit::FT
    # TODO: remove defaults, or move these parameters to CLIMAParameters
    f::FT = 0 # coriolis parameter. TODO: remove?
    Cd::FT = 0 # drag coefficients. TODO: remove?
    ug::FT = 0
    vg::FT = 0
    thermodynamics_params::TP
    rrtmgp_params::RP
    insolation_params::IP
    microphysics_params::MPP
    surfacefluxes_params::SFP
    turbconv_params::TCP
end

rrtmgp_params(ps::ACAP) = ps.rrtmgp_params
thermodynamics_params(ps::ACAP) = ps.thermodynamics_params
surface_fluxes_params(ps::ACAP) = ps.surfacefluxes_params
microphysics_params(ps::ACAP) = ps.microphysics_params
insolation_params(ps::ACAP) = ps.insolation_params
turbconv_params(ps::ACAP) = ps.turbconv_params

# Forward Thermodynamics parameters
for var in fieldnames(TD.Parameters.ThermodynamicsParameters)
    @eval $var(ps::ACAP) = TD.Parameters.$var(thermodynamics_params(ps))
end

# Thermodynamics derived parameters
molmass_ratio(ps::ACAP) = TD.Parameters.molmass_ratio(thermodynamics_params(ps))
R_d(ps::ACAP) = TD.Parameters.R_d(thermodynamics_params(ps))
R_v(ps::ACAP) = TD.Parameters.R_v(thermodynamics_params(ps))
cp_d(ps::ACAP) = TD.Parameters.cp_d(thermodynamics_params(ps))
cv_v(ps::ACAP) = TD.Parameters.cv_v(thermodynamics_params(ps))
cv_l(ps::ACAP) = TD.Parameters.cv_l(thermodynamics_params(ps))
cv_d(ps::ACAP) = TD.Parameters.cv_d(thermodynamics_params(ps))

Omega(ps::ACAP) = ps.Omega
f_plane_coriolis_frequency(ps::ACAP) = ps.f_plane_coriolis_frequency
planet_radius(ps::ACAP) = ps.planet_radius
astro_unit(ps::ACAP) = ps.astro_unit
f(ps::ACAP) = ps.f
Cd(ps::ACAP) = ps.Cd
uh_g(ps::ACAP) = CC.Geometry.UVVector(ps.ug, ps.vg)

# Forwarding CloudMicrophysics parameters
ρ_cloud_liq(ps::ACAP) = CM.Parameters.ρ_cloud_liq(microphysics_params(ps))

# Insolation parameters
day(ps::ACAP) = IP.day(insolation_params(ps))
tot_solar_irrad(ps::ACAP) = IP.tot_solar_irrad(insolation_params(ps))

end
