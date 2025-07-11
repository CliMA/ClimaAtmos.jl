import ClimaParams as CP
import RRTMGP.Parameters.RRTMGPParameters
import SurfaceFluxes.Parameters.SurfaceFluxesParameters
import SurfaceFluxes.UniversalFunctions as UF
import Insolation.Parameters.InsolationParameters
import Thermodynamics.Parameters.ThermodynamicsParameters
import CloudMicrophysics as CM
import StaticArrays as SA

"""
    ClimaAtmosParameters(FT::AbstractFloat)
    ClimaAtmosParameters(toml_dict)
    ClimaAtmosParameters(config::AtmosConfig)

Construct the parameter set for any ClimaAtmos configuration.
"""
ClimaAtmosParameters(config::AtmosConfig) =
    ClimaAtmosParameters(config.toml_dict)

ClimaAtmosParameters(::Type{FT}) where {FT <: AbstractFloat} =
    ClimaAtmosParameters(CP.create_toml_dict(FT))

function ClimaAtmosParameters(toml_dict::TD) where {TD <: CP.AbstractTOMLDict}
    FT = CP.float_type(toml_dict)

    turbconv_params = TurbulenceConvectionParameters(toml_dict)
    TCP = typeof(turbconv_params)

    thermodynamics_params = ThermodynamicsParameters(toml_dict)
    TP = typeof(thermodynamics_params)

    rrtmgp_params = RRTMGPParameters(toml_dict)
    RP = typeof(rrtmgp_params)

    insolation_params = InsolationParameters(toml_dict)
    IP = typeof(insolation_params)

    surface_fluxes_params =
        SurfaceFluxesParameters(toml_dict, UF.BusingerParams)
    SFP = typeof(surface_fluxes_params)

    # Fetch Gryanik b_m coefficient (since surface_fluxes_params defaults to BusingerParams)
    coeff_b_m_gryanik_val = UF.GryanikParams(FT).b_m

    surface_temp_params = SurfaceTemperatureParameters(toml_dict)
    STP = typeof(surface_temp_params)

    microphysics_cloud_params = cloud_parameters(toml_dict)
    MPC = typeof(microphysics_cloud_params)

    microphysics_0m_params = CM.Parameters.Parameters0M(toml_dict)
    microphysics_1m_params = microphys_1m_parameters(toml_dict)
    microphysics_2m_params = microphys_2m_parameters(toml_dict)
    MP0M = typeof(microphysics_0m_params)
    MP1M = typeof(microphysics_1m_params)
    MP2M = typeof(microphysics_2m_params)

    vert_diff_params = vert_diff_parameters(toml_dict)
    VDP = typeof(vert_diff_params)

    external_forcing_params = external_forcing_parameters(toml_dict)
    EFP = typeof(external_forcing_params)

    parameters =
        CP.get_parameter_values(toml_dict, atmos_name_map, "ClimaAtmos")
    return CAP.ClimaAtmosParameters{
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
    }(;
        parameters...,
        thermodynamics_params,
        rrtmgp_params,
        insolation_params,
        microphysics_cloud_params,
        microphysics_0m_params,
        microphysics_1m_params,
        microphysics_2m_params,
        surface_fluxes_params,
        turbconv_params,
        surface_temp_params,
        vert_diff_params,
        external_forcing_params,
        coeff_b_m_gryanik = coeff_b_m_gryanik_val,
    )
end

atmos_name_map = (;
    :f_plane_coriolis_frequency => :f_plane_coriolis_frequency,
    :equator_pole_temperature_gradient_wet => :ΔT_y_wet,
    :angular_velocity_planet_rotation => :Omega,
    :equator_pole_temperature_gradient_dry => :ΔT_y_dry,
    :held_suarez_T_equator_wet => :T_equator_wet,
    :zd_rayleigh => :zd_rayleigh,
    :zd_viscous => :zd_viscous,
    :planet_radius => :planet_radius,
    :potential_temp_vertical_gradient => :Δθ_z,
    :C_H => :C_H,
    :c_smag => :c_smag,
    :alpha_rayleigh_w => :alpha_rayleigh_w,
    :alpha_rayleigh_uh => :alpha_rayleigh_uh,
    :astronomical_unit => :astro_unit,
    :held_suarez_T_equator_dry => :T_equator_dry,
    :drag_layer_vertical_extent => :σ_b,
    :kappa_2_sponge => :kappa_2_sponge,
    :held_suarez_minimum_temperature => :T_min_hs,
    :ocean_surface_albedo => :idealized_ocean_albedo,
    :water_refractive_index => :water_refractive_index,
    :optics_lookup_temperature_min => :optics_lookup_temperature_min,
    :optics_lookup_temperature_max => :optics_lookup_temperature_max,
    :tracer_hyperdiffusion_factor => :α_hyperdiff_tracer,
    :tracer_vertical_diffusion_factor => :α_vert_diff_tracer,
)

cloud_parameters(::Type{FT}) where {FT <: AbstractFloat} =
    cloud_parameters(CP.create_toml_dict(FT))

cloud_parameters(toml_dict::CP.AbstractTOMLDict) = (;
    liquid = CM.Parameters.CloudLiquid(toml_dict),
    ice = CM.Parameters.CloudIce(toml_dict),
    Ch2022 = CM.Parameters.Chen2022VelType(toml_dict),
    N_cloud_liquid_droplets = CP.get_parameter_values(
        toml_dict,
        "prescribed_cloud_droplet_number_concentration",
        "ClimaAtmos",
    ).prescribed_cloud_droplet_number_concentration,
    aml = aerosol_ml_parameters(toml_dict),
)

microphys_1m_parameters(::Type{FT}) where {FT <: AbstractFloat} =
    microphys_1m_parameters(CP.create_toml_dict(FT))

microphys_1m_parameters(toml_dict::CP.AbstractTOMLDict) = (;
    cl = CM.Parameters.CloudLiquid(toml_dict),
    ci = CM.Parameters.CloudIce(toml_dict),
    pr = CM.Parameters.Rain(toml_dict),
    ps = CM.Parameters.Snow(toml_dict),
    ce = CM.Parameters.CollisionEff(toml_dict),
    tv = CM.Parameters.Blk1MVelType(toml_dict),
    aps = CM.Parameters.AirProperties(toml_dict),
    var = CM.Parameters.VarTimescaleAcnv(toml_dict),
    Ndp = CP.get_parameter_values(
        toml_dict,
        "prescribed_cloud_droplet_number_concentration",
        "ClimaAtmos",
    ).prescribed_cloud_droplet_number_concentration,
)

microphys_2m_parameters(::Type{FT}) where {FT <: AbstractFloat} =
    microphys_2m_parameters(CP.create_toml_dict(FT))

microphys_2m_parameters(toml_dict::CP.AbstractTOMLDict) = (;
    sb = CM.Parameters.SB2006(toml_dict),
    aps = CM.Parameters.AirProperties(toml_dict),
    tv = CM.Parameters.SB2006VelType(toml_dict),
)

function vert_diff_parameters(toml_dict)
    name_map = (; :C_E => :C_E, :H_diffusion => :H, :D_0_diffusion => :D₀)
    return CP.get_parameter_values(toml_dict, name_map, "ClimaAtmos")
end

function external_forcing_parameters(toml_dict)
    efp_fields = [
        "gcmdriven_momentum_relaxation_timescale",
        "gcmdriven_scalar_relaxation_timescale",
        "gcmdriven_relaxation_minimum_height",
        "gcmdriven_relaxation_maximum_height",
    ]
    return CP.get_parameter_values(toml_dict, efp_fields, "ClimaAtmos")
end

function aerosol_ml_parameters(toml_dict)
    name_map = (;
        :prescribed_cloud_droplet_number_concentration => :N₀,
        :dust_calibration_coefficient => :α_dust,
        :seasalt_calibration_coefficient => :α_seasalt,
        :ammonium_sulfate_calibration_coefficient => :α_SO4,
        :liquid_water_specific_humidity_calibration_coefficent => :α_q_liq,
        :reference_dust_aerosol_mass_concentration => :c₀_dust,
        :reference_seasalt_aerosol_mass_concentration => :c₀_seasalt,
        :reference_ammonium_sulfate_mass_concentration => :c₀_SO4,
        :reference_liquid_water_specific_humidity => :q₀_liq,
    )
    return CP.get_parameter_values(toml_dict, name_map, "ClimaAtmos")
end

to_svec(x::AbstractArray) = SA.SVector{length(x)}(x)
to_svec(x) = x
to_svec(x::NamedTuple) = map(x -> to_svec(x), x)

TurbulenceConvectionParameters(
    ::Type{FT},
    overrides = NamedTuple(),
) where {FT <: AbstractFloat} =
    TurbulenceConvectionParameters(CP.create_toml_dict(FT), overrides)

function TurbulenceConvectionParameters(
    toml_dict::CP.AbstractTOMLDict,
    overrides = NamedTuple(),
)
    name_map = (;
        :min_area_limiter_scale => :min_area_limiter_scale,
        :max_area_limiter_scale => :max_area_limiter_scale,
        :mixing_length_tke_surf_scale => :tke_surf_scale,
        :mixing_length_tke_surf_flux_coeff => :tke_surf_flux_coeff,
        :mixing_length_diss_coeff => :tke_diss_coeff,
        :diagnostic_covariance_coeff => :diagnostic_covariance_coeff,
        :detr_buoy_coeff => :detr_buoy_coeff,
        :EDMF_max_area => :max_area,
        :mixing_length_smin_rm => :smin_rm,
        :entr_coeff => :entr_coeff,
        :detr_coeff => :detr_coeff,
        :EDMF_surface_area => :surface_area,
        :entr_param_vec => :entr_param_vec,
        :turb_entr_param_vec => :turb_entr_param_vec,
        :entr_mult_limiter_coeff => :entr_mult_limiter_coeff,
        :minimum_updraft_top => :min_updraft_top,
        :mixing_length_eddy_viscosity_coefficient => :tke_ed_coeff,
        :mixing_length_smin_ub => :smin_ub,
        :EDMF_min_area => :min_area,
        :detr_vertdiv_coeff => :detr_vertdiv_coeff,
        :detr_massflux_vertdiv_coeff => :detr_massflux_vertdiv_coeff,
        :max_area_limiter_power => :max_area_limiter_power,
        :min_area_limiter_power => :min_area_limiter_power,
        :pressure_normalmode_drag_coeff => :pressure_normalmode_drag_coeff,
        :mixing_length_Prandtl_number_scale => :Prandtl_number_scale,
        :mixing_length_Prandtl_number_0 => :Prandtl_number_0,
        :mixing_length_Prandtl_maximum => :Pr_max,
        :mixing_length_static_stab_coeff => :static_stab_coeff,
        :pressure_normalmode_buoy_coeff1 =>
            :pressure_normalmode_buoy_coeff1,
        :detr_inv_tau => :detr_tau,
        :entr_inv_tau => :entr_tau,
    )
    parameters = CP.get_parameter_values(toml_dict, name_map, "ClimaAtmos")
    parameters = merge(parameters, overrides)
    parameters = to_svec(parameters)
    VFT1 = typeof(parameters.entr_param_vec)
    VFT2 = typeof(parameters.turb_entr_param_vec)
    FT = CP.float_type(toml_dict)
    CAP.TurbulenceConvectionParameters{FT, VFT1, VFT2}(; parameters...)
end

SurfaceTemperatureParameters(
    ::Type{FT},
    overrides = NamedTuple(),
) where {FT <: AbstractFloat} =
    SurfaceTemperatureParameters(CP.create_toml_dict(FT), overrides)

function SurfaceTemperatureParameters(
    toml_dict::CP.AbstractTOMLDict,
    overrides = NamedTuple(),
)
    name_map = (;
        :SST_mean => :SST_mean,
        :SST_delta => :SST_delta,
        :SST_wavelength => :SST_wavelength,
        :SST_wavelength_latitude => :SST_wavelength_latitude,
    )
    parameters = CP.get_parameter_values(toml_dict, name_map, "ClimaAtmos")
    parameters = merge(parameters, overrides)
    FT = CP.float_type(toml_dict)
    CAP.SurfaceTemperatureParameters{FT}(; parameters...)
end
