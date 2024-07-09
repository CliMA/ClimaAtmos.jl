module Parameters

import ClimaCore as CC
import Insolation.Parameters as IP
import Thermodynamics as TD
import CloudMicrophysics as CM
import SurfaceFluxes as SF

abstract type AbstractClimaAtmosParameters end
const ACAP = AbstractClimaAtmosParameters

abstract type AbstractTurbulenceConvectionParameters end
const ATCP = AbstractTurbulenceConvectionParameters

abstract type AbstractSurfaceTemperatureParameters end
const ASTP = AbstractSurfaceTemperatureParameters

Base.broadcastable(param_set::ACAP) = tuple(param_set)
Base.broadcastable(param_set::ATCP) = tuple(param_set)
Base.broadcastable(param_set::ASTP) = tuple(param_set)

Base.@kwdef struct TurbulenceConvectionParameters{FT} <: ATCP
    surface_area::FT
    max_area::FT
    min_area::FT
    tke_ed_coeff::FT
    tke_diss_coeff::FT
    tke_surf_scale::FT
    diagnostic_covariance_coeff::FT
    static_stab_coeff::FT
    Prandtl_number_scale::FT
    Prandtl_number_0::FT
    Ri_crit::FT
    smin_ub::FT
    smin_rm::FT
    min_updraft_top::FT
    pressure_normalmode_buoy_coeff1::FT
    pressure_normalmode_drag_coeff::FT
    entr_tau::FT
    entr_coeff::FT
    detr_tau::FT
    detr_coeff::FT
    detr_buoy_coeff::FT
    detr_vertdiv_coeff::FT
    detr_massflux_vertdiv_coeff::FT
    min_area_limiter_scale::FT
    min_area_limiter_power::FT
    max_area_limiter_scale::FT
    max_area_limiter_power::FT
end

Base.@kwdef struct SurfaceTemperatureParameters{FT} <: ASTP
    SST_mean::FT
    SST_delta::FT
    SST_wavelength::FT
    SST_wavelength_latitude::FT
end

Base.@kwdef struct ClimaAtmosParameters{
    FT,
    TDP,
    RP,
    IP,
    MPC,
    MPP,
    WP,
    SFP,
    TCP,
    STP,
    TP,
} <: ACAP
    thermodynamics_params::TDP
    rrtmgp_params::RP
    insolation_params::IP
    microphysics_cloud_params::MPC
    microphysics_precipitation_params::MPP
    water_params::WP
    surface_fluxes_params::SFP
    turbconv_params::TCP
    surface_temp_params::STP
    topography_params::TP
    Omega::FT
    f_plane_coriolis_frequency::FT
    planet_radius::FT
    astro_unit::FT
    c_smag::FT
    C_E::FT
    C_H::FT
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
    # Radiation
    idealized_ocean_albedo::FT
    water_refractive_index::FT
    optics_lookup_temperature_min::FT
    optics_lookup_temperature_max::FT
end

Base.eltype(::ClimaAtmosParameters{FT}) where {FT} = FT

# Forward TurbulenceConvection parameters
const TCPS = TurbulenceConvectionParameters
for var in fieldnames(TCPS)
    @eval $var(ps::ACAP) = $var(turbconv_params(ps))
end
for fn in fieldnames(TCPS)
    @eval $(fn)(ps::ATCP) = ps.$(fn)
end

# Forward Thermodynamics parameters
for var in fieldnames(TD.Parameters.ThermodynamicsParameters)
    @eval $var(ps::ACAP) = TD.Parameters.$var(thermodynamics_params(ps))
end
# Thermodynamics derived parameters
for var in [:molmass_ratio, :R_d, :R_v, :e_int_v0, :cp_d, :cv_v, :cv_l, :cv_d]
    @eval $var(ps::ACAP) = TD.Parameters.$var(thermodynamics_params(ps))
end

# Forwarding CloudMicrophysics parameters
ρ_cloud_liq(ps::ACAP) = ps.water_params.ρw
ρ_cloud_ice(ps::ACAP) = ps.water_params.ρi

# Forwarding SurfaceFluxes parameters
von_karman_const(ps::ACAP) =
    SF.Parameters.von_karman_const(surface_fluxes_params(ps))

# Insolation parameters
day(ps::ACAP) = IP.day(insolation_params(ps))
tot_solar_irrad(ps::ACAP) = IP.tot_solar_irrad(insolation_params(ps))

# Define parameters as functions
for var in fieldnames(ClimaAtmosParameters)
    @eval $var(ps::ACAP) = ps.$var
end

end
