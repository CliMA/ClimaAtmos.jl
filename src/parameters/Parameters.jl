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

abstract type AbstractGravityWaveParameters end
const AGWP = AbstractGravityWaveParameters

Base.broadcastable(param_set::ACAP) = tuple(param_set)
Base.broadcastable(param_set::ATCP) = tuple(param_set)
Base.broadcastable(param_set::ASTP) = tuple(param_set)

Base.@kwdef struct TurbulenceConvectionParameters{FT, VFT1, VFT2, VTF3} <: ATCP
    max_surface_area::FT
    max_area::FT
    min_area::FT
    tke_ed_coeff::FT
    Ri_crit::FT
    tke_surf_scale::FT
    tke_surf_flux_coeff::FT
    diagnostic_covariance_coeff::FT
    Tq_correlation_coefficient::FT
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
    entr_detr_limit_inv_tau::FT
    detr_inv_tau::FT
    detr_coeff::FT
    detr_buoy_coeff::FT
    detr_buoy_inv_tau_max::FT
    detr_vertdiv_coeff::FT
    entr_param_vec::VFT1
    turb_entr_param_vec::VFT2
    entr_mult_limiter_coeff::FT
    detr_massflux_vertdiv_coeff::FT
    min_area_limiter_scale::FT
    min_area_limiter_power::FT
    max_area_limiter_scale::FT
    max_area_limiter_power::FT
    cloud_fraction_steepness_scale::FT
    cloud_fraction_param_vec::VTF3
    cloud_fraction_eps_rel::FT
    cloud_fraction_sigma_abs::FT
    cloud_fraction_floor_release_margin::FT
    cloud_fraction_floor_release_abs_margin::FT
    cloud_fraction_floor_release_sharpness::FT
    cloud_fraction_floor_residual::FT
    cloud_fraction_wellmixed_gref::FT
    interface_entr_efficiency::FT
    # Surface mass flux closure (`set_edmfx_surface_conditions!`).
    sfc_mass_flux_ustar_coeff::FT
    convective_zi::FT
    sfc_mass_flux_cap_fraction::FT
end

Base.@kwdef struct SurfaceTemperatureParameters{FT} <: ASTP
    SST_mean::FT
    SST_delta::FT
    SST_wavelength::FT
    SST_wavelength_latitude::FT
end

Base.@kwdef struct NonOrographicGravityWaveParameters{FT} <: AGWP
    source_pressure::FT
    damp_pressure::FT
    source_height::FT
    Bw::FT
    Bn::FT
    dc::FT
    cmax::FT
    c0::FT
    nk::FT
    cw::FT
    cw_tropics::FT
    cn::FT
    Bt_0::FT
    Bt_n::FT
    Bt_s::FT
    Bt_eq::FT
    ϕ0_n::FT
    ϕ0_s::FT
    dϕ_n::FT
    dϕ_s::FT
end

Base.@kwdef struct OrographicGravityWaveParameters{FT} <: AGWP
    γ::FT                    # mountain_height_width_exponent: L ∝ h^γ (equation 14, paper suggests γ ≈ 0.4)
    ϵ::FT                    # number_density_exponent: number density of orography in a grid cell, n(h) ∝ h^(-ε)
    β::FT                    # mountain_shape_parameter: L(z) = L_b(1 - z/h)^β (equation 12), β=1 for triangular mountains and β<1 for blunt mountains, β>1 for pointy mountains
    h_frac::FT               # critical_height_threshold: h_crit = h_frac * (V / N), height fraction for blocking threshold
    ρscale::FT               # density_scale_factor: density scale factor for dimensional analysis
    L0::FT                   # reference_mountain_width: L_0 = 80 km, reference horizontal scale
    a0::FT                   # linear_drag_coefficient: a_0 = 0.9, coefficient for propagating wave drag
    a1::FT                   # nonlinear_drag_coefficient: a_1 = 3.0, coefficient for nonpropagating (blocked) drag
    Fr_crit::FT              # critical_froude_number: Fr_crit = 0.7, critical Froude number h̃_c = Fr_crit
end

# Physical/tuning parameters for the Beres (2004) convective gravity-wave source.
Base.@kwdef struct BeresSourceParameters{FT} <: AGWP
    Q0_threshold::FT         # K/s, minimum heating rate to activate Beres
    scale_factor::FT         # dimensionless amplitude scaling (folds ρ₀/(Lτ), |Q_t|² weight, tuning)
    σ_x::FT                  # m, convective cell horizontal half-width
    ν_min::FT                # 1/s, min frequency (period ~120 min)
    ν_max::FT                # 1/s, max frequency (period ~10 min)
    n_ν::Int                 # quadrature points (must be 4k+1: 5, 9, 13...)
    h_heat_min::FT           # m, minimum heating depth to activate (filters shallow convection)
    n_h_avg::Int             # number of h values to average over (1 = no averaging)
    Δh_frac::FT              # fractional half-range for h averaging: h ± Δh_frac·h
    z_bot_floor::FT          # m, minimum allowed z_bot (excludes PBL turbulence in Q_conv)
    steady_dc_frac::FT       # steady DC heating weight: Q_t(0)² = steady_dc_frac·ν_min
    L_system::FT             # m, largest convective-system scale; sets k_min=2π/L for the steady source
end

Base.@kwdef struct ClimaAtmosParameters{
    FT,
    TP,
    RP,
    TG,
    IP,
    MPC,
    MP0M,
    MP1M,
    MP2M,
    MP2MP3,
    SFP,
    TCP,
    STP,
    VDP,
    EFP,
    PAP,
    NOGWP,
    OGWP,
    BSP,
} <: ACAP
    thermodynamics_params::TP
    rrtmgp_params::RP
    trace_gas_params::TG
    insolation_params::IP
    microphysics_cloud_params::MPC
    microphysics_0m_params::MP0M
    microphysics_1m_params::MP1M
    microphysics_2m_params::MP2M
    microphysics_2mp3_params::MP2MP3
    surface_fluxes_params::SFP
    turbconv_params::TCP
    surface_temp_params::STP
    vert_diff_params::VDP
    external_forcing_params::EFP
    prescribed_aerosol_params::PAP
    non_orographic_gravity_wave_params::NOGWP
    orographic_gravity_wave_params::OGWP
    beres_source_params::BSP
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
    alpha_rayleigh_tracer::FT
    zd_viscous::FT
    zd_rayleigh::FT
    kappa_2_sponge::FT
    # Radiation
    idealized_ocean_albedo::FT
    water_refractive_index::FT
    # Hyperdiffusion
    α_hyperdiff_tracer::FT
    # Vertical diffusion
    α_vert_diff_tracer::FT
    # Constant horizontal diffusion
    constant_horizontal_diffusion_D::FT
    # SGS quadrature bounds
    T_min_sgs::FT
    q_max_sgs::FT
    # Fixed terminal velocities
    fixed_cloud_liquid_terminal_velocity::FT
    fixed_cloud_ice_terminal_velocity::FT
    fixed_rain_terminal_velocity::FT
    fixed_snow_terminal_velocity::FT
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

for var in fieldnames(TD.Parameters.ThermodynamicsParameters)
    @eval $var(ps::ACAP) = TD.Parameters.$var(thermodynamics_params(ps))
end
# Thermodynamics derived parameters
for var in [:Rv_over_Rd, :kappa_d, :e_int_v0, :e_int_i0, :cv_v, :cv_l, :cv_d]
    @eval $var(ps::ACAP) = TD.Parameters.$var(thermodynamics_params(ps))
end

# Forwarding SurfaceFluxes parameters
von_karman_const(ps::ACAP) =
    SF.Parameters.von_karman_const(surface_fluxes_params(ps))

# ------ MOST (Monin–Obukhov) stability-function coefficients ------

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
