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

Base.@kwdef struct TurbulenceConvectionParameters{FT, VFT1, VFT2} <: ATCP
    surface_area::FT
    max_area::FT
    min_area::FT
    tke_ed_coeff::FT
    tke_diss_coeff::FT
    tke_surf_scale::FT
    tke_surf_flux_coeff::FT
    diagnostic_covariance_coeff::FT
    static_stab_coeff::FT
    Prandtl_number_scale::FT
    Prandtl_number_0::FT
    Pr_max::FT
    smin_ub::FT
    smin_rm::FT
    min_updraft_top::FT
    pressure_normalmode_buoy_coeff1::FT
    pressure_normalmode_drag_coeff::FT
    entr_inv_tau::FT
    entr_coeff::FT
    detr_inv_tau::FT
    detr_coeff::FT
    detr_buoy_coeff::FT
    detr_vertdiv_coeff::FT
    entr_param_vec::VFT1
    turb_entr_param_vec::VFT2
    entr_mult_limiter_coeff::FT
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
    TP,
    RP,
    IP,
    MPC,
    MP0M,
    MP1M,
    MP2M,
    SFP,
    TCP,
    STP,
    VDP,
    EFP,
} <: ACAP
    thermodynamics_params::TP
    rrtmgp_params::RP
    insolation_params::IP
    microphysics_cloud_params::MPC
    microphysics_0m_params::MP0M
    microphysics_1m_params::MP1M
    microphysics_2m_params::MP2M
    surface_fluxes_params::SFP
    turbconv_params::TCP
    surface_temp_params::STP
    vert_diff_params::VDP
    external_forcing_params::EFP
    Omega::FT
    f_plane_coriolis_frequency::FT
    planet_radius::FT
    astro_unit::FT
    c_smag::FT
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
    alpha_rayleigh_sgs_tracer::FT
    zd_viscous::FT
    zd_rayleigh::FT
    kappa_2_sponge::FT
    # Radiation
    idealized_ocean_albedo::FT
    water_refractive_index::FT
    optics_lookup_temperature_min::FT
    optics_lookup_temperature_max::FT
    # Hyperdiffusion
    α_hyperdiff_tracer::FT
    # Vertical diffusion
    α_vert_diff_tracer::FT
    # Gryanik b_m coefficient
    coeff_b_m_gryanik::FT
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
for var in [:Rv_over_Rd, :e_int_v0, :cv_v, :cv_l, :cv_d]
    @eval $var(ps::ACAP) = TD.Parameters.$var(thermodynamics_params(ps))
end

# Forwarding SurfaceFluxes parameters
von_karman_const(ps::ACAP) =
    SF.Parameters.von_karman_const(surface_fluxes_params(ps))

# ------ MOST (Monin–Obukhov) stability-function coefficients ------

# Gryanik b_m
# needed because surface_fluxes_params defaults to BusingerParams
coefficient_b_m_gryanik(ps::ACAP) = ps.coeff_b_m_gryanik

# Insolation parameters
day(ps::ACAP) = IP.day(insolation_params(ps))
tot_solar_irrad(ps::ACAP) = IP.tot_solar_irrad(insolation_params(ps))

# Forward External Forcing parameters
efp_fields = [
    :gcmdriven_momentum_relaxation_timescale,
    :gcmdriven_scalar_relaxation_timescale,
    :gcmdriven_relaxation_minimum_height,
    :gcmdriven_relaxation_maximum_height,
]

for fn_efp in efp_fields
    @eval $(fn_efp)(ps::ACAP) = external_forcing_params(ps).$(fn_efp)
end

# Define parameters as functions
for var in fieldnames(ClimaAtmosParameters)
    @eval $var(ps::ACAP) = ps.$var
end

end
