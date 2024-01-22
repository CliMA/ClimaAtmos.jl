import CLIMAParameters as CP
import RRTMGP.Parameters.RRTMGPParameters
import SurfaceFluxes.Parameters.SurfaceFluxesParameters
import SurfaceFluxes.UniversalFunctions as UF
import Insolation.Parameters.InsolationParameters
import Thermodynamics.Parameters.ThermodynamicsParameters
import CloudMicrophysics as CM

function TurbulenceConvectionParameters(toml_dict::CP.AbstractTOMLDict)
    name_map = (;
        :min_area_limiter_scale => :min_area_limiter_scale,
        :max_area_limiter_scale => :max_area_limiter_scale,
        :mixing_length_tke_surf_scale => :tke_surf_scale,
        :mixing_length_diss_coeff => :tke_diss_coeff,
        :detr_buoy_coeff => :detr_buoy_coeff,
        :EDMF_max_area => :max_area,
        :mixing_length_smin_rm => :smin_rm,
        :entr_coeff => :entr_coeff,
        :mixing_length_Ri_crit => :Ri_crit,
        :detr_coeff => :detr_coeff,
        :EDMF_surface_area => :surface_area,
        :minimum_updraft_top => :min_updraft_top,
        :mixing_length_eddy_viscosity_coefficient => :tke_ed_coeff,
        :mixing_length_smin_ub => :smin_ub,
        :EDMF_min_area => :min_area,
        :detr_vertdiv_coeff => :detr_vertdiv_coeff,
        :max_area_limiter_power => :max_area_limiter_power,
        :min_area_limiter_power => :min_area_limiter_power,
        :pressure_normalmode_drag_coeff => :pressure_normalmode_drag_coeff,
        :mixing_length_Prandtl_number_scale => :Prandtl_number_scale,
        :mixing_length_Prandtl_number_0 => :Prandtl_number_0,
        :mixing_length_static_stab_coeff => :static_stab_coeff,
        :pressure_normalmode_buoy_coeff1 =>
            :pressure_normalmode_buoy_coeff1,
        :detr_inv_tau => :detr_tau,
        :entr_inv_tau => :entr_tau,
    )
    parameters = CP.get_parameter_values(toml_dict, name_map, "ClimaAtmos")
    FT = CP.float_type(toml_dict)
    CAP.TurbulenceConvectionParameters{FT}(; parameters...)
end

function create_parameter_set(config::AtmosConfig)
    (; toml_dict, parsed_args) = config
    FT = CP.float_type(toml_dict)

    turbconv_params = TurbulenceConvectionParameters(toml_dict)
    TCP = typeof(turbconv_params)

    thermodynamics_params = ThermodynamicsParameters(toml_dict)
    TP = typeof(thermodynamics_params)

    rrtmgp_params = RRTMGPParameters(toml_dict)
    RP = typeof(rrtmgp_params)

    insolation_params = InsolationParameters(toml_dict)
    IP = typeof(insolation_params)

    water_params = CM.Parameters.WaterProperties(FT, toml_dict)
    WP = typeof(water_params)

    surface_fluxes_params =
        SF.Parameters.SurfaceFluxesParameters(toml_dict, UF.BusingerParams)
    SFP = typeof(surface_fluxes_params)

    # Microphysics scheme parameters (from CloudMicrophysics.jl)
    # TODO - repeating the logic from solver/model_getters.jl...
    if parsed_args["override_τ_precip"]
        toml_dict["precipitation_timescale"]["value"] =
            FT(CA.time_to_seconds(parsed_args["dt"]))
    end
    precip_model = parsed_args["precip_model"]
    microphysics_params =
        if precip_model == nothing || precip_model == "nothing"
            nothing
        elseif precip_model == "0M"
            CM.Parameters.Parameters0M(FT, toml_dict)
        elseif precip_model == "1M"
            (;
                cl = CM.Parameters.CloudLiquid(FT, toml_dict),
                ci = CM.Parameters.CloudIce(FT, toml_dict),
                pr = CM.Parameters.Rain(FT, toml_dict),
                ps = CM.Parameters.Snow(FT, toml_dict),
                ce = CM.Parameters.CollisionEff(FT, toml_dict),
                tv = CM.Parameters.Blk1MVelType(FT, toml_dict),
                aps = CM.Parameters.AirProperties(FT, toml_dict),
            )
        else
            error("Invalid precip_model $(precip_model)")
        end
    MPP = typeof(microphysics_params)

    name_map = (;
        :f_plane_coriolis_frequency => :f_plane_coriolis_frequency,
        :equator_pole_temperature_gradient_wet => :ΔT_y_wet,
        :angular_velocity_planet_rotation => :Omega,
        :equator_pole_temperature_gradient_dry => :ΔT_y_dry,
        :held_suarez_T_equator_wet => :T_equator_wet,
        :zd_rayleigh => :zd_rayleigh,
        :zd_viscous => :zd_viscous,
        :planet_radius => :planet_radius,
        :potential_temp_vertical_gradient => :Δθ_z,
        :C_E => :C_E,
        :C_H => :C_H,
        :c_smag => :c_smag,
        :alpha_rayleigh_w => :alpha_rayleigh_w,
        :alpha_rayleigh_uh => :alpha_rayleigh_uh,
        :astronomical_unit => :astro_unit,
        :held_suarez_T_equator_dry => :T_equator_dry,
        :drag_layer_vertical_extent => :σ_b,
        :kappa_2_sponge => :kappa_2_sponge,
        :held_suarez_minimum_temperature => :T_min_hs,
    )
    parameters = CP.get_parameter_values(toml_dict, name_map, "ClimaAtmos")

    return CAP.ClimaAtmosParameters{FT, TP, RP, IP, MPP, WP, SFP, TCP}(;
        parameters...,
        thermodynamics_params,
        rrtmgp_params,
        insolation_params,
        microphysics_params,
        water_params,
        surface_fluxes_params,
        turbconv_params,
    )
end
