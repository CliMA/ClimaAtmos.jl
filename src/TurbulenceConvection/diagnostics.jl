#####
##### Diagnostics
#####

#=
    io_dictionary_aux()

These functions return a dictionary whose
 - `keys` are the nc variable names
 - `values` are NamedTuples corresponding to
    - `dims` (`("z")`  or `("z", "t")`) and
    - `group` (`"reference"` or, e.g., `"profiles"`)
=#

#! format: off
# TODO: use a better name, this exports fields from the prognostic state, and the aux state.
function io_dictionary_aux()
    DT = NamedTuple{(:dims, :group, :field), Tuple{Tuple{String, String}, String, Any}}
    io_dict = Dict{String, DT}(
        "updraft_area" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_bulk(state).area),
        "updraft_ql" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_bulk(state).q_liq),
        "updraft_qi" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_bulk(state).q_ice),
        "updraft_RH" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_bulk(state).RH),
        "updraft_qt" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_bulk(state).q_tot),
        "updraft_w" => (; dims = ("zf", "t"), group = "profiles", field = state -> face_aux_bulk(state).w),
        "updraft_temperature" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_bulk(state).T),
        "updraft_thetal" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_bulk(state).θ_liq_ice),
        "updraft_buoyancy" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_bulk(state).buoy),
        "H_third_m" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_grid_mean(state).H_third_m),
        "W_third_m" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_grid_mean(state).W_third_m),
        "QT_third_m" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_grid_mean(state).QT_third_m),
        "cloud_fraction" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_grid_mean(state).cloud_fraction), # was this "cloud_fraction_mean"?
        "buoyancy_mean" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_grid_mean(state).buoy),
        "temperature_mean" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_grid_mean(state).T),
        "RH_mean" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_grid_mean(state).RH),
        "s_mean" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_grid_mean(state).s),
        "ql_mean" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_grid_mean(state).q_liq),
        "qi_mean" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_grid_mean(state).q_ice),
        "tke_mean" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_grid_mean(state).tke),
        "Hvar_mean" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_grid_mean(state).Hvar),
        "QTvar_mean" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_grid_mean(state).QTvar),
        "HQTcov_mean" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_grid_mean(state).HQTcov),
        "u_mean" => (; dims = ("zc", "t"), group = "profiles", field = state -> physical_grid_mean_u(state)),
        "v_mean" => (; dims = ("zc", "t"), group = "profiles", field = state -> physical_grid_mean_v(state)),
        "w_mean" => (; dims = ("zf", "t"), group = "profiles", field = state -> face_prog_grid_mean(state).w),
        "qt_mean" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_grid_mean(state).q_tot),
        "thetal_mean" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_grid_mean(state).θ_liq_ice),
        "eddy_viscosity" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_turbconv(state).KM),
        "eddy_diffusivity" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_turbconv(state).KH),
        "env_tke" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_environment(state).tke),
        "env_buoyancy" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_environment(state).buoy),
        "env_Hvar" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_environment(state).Hvar),
        "env_QTvar" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_environment(state).QTvar),
        "env_HQTcov" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_environment(state).HQTcov),

        "tke_buoy" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_environment_2m(state).tke.buoy),
        "tke_pressure" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_environment_2m(state).tke.press),
        "tke_dissipation" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_environment_2m(state).tke.dissipation),
        "tke_entr_gain" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_environment_2m(state).tke.entr_gain),
        "tke_detr_loss" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_environment_2m(state).tke.detr_loss),
        "tke_shear" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_environment_2m(state).tke.shear),
        "tke_interdomain" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_environment_2m(state).tke.interdomain),

        "Hvar_dissipation" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_environment_2m(state).Hvar.dissipation),
        "Hvar_entr_gain" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_environment_2m(state).Hvar.entr_gain),
        "Hvar_detr_loss" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_environment_2m(state).Hvar.detr_loss),
        "Hvar_interdomain" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_environment_2m(state).Hvar.interdomain),
        "Hvar_rain" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_environment_2m(state).Hvar.rain_src),
        "Hvar_shear" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_environment_2m(state).Hvar.shear),

        "QTvar_dissipation" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_environment_2m(state).QTvar.dissipation),
        "QTvar_entr_gain" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_environment_2m(state).QTvar.entr_gain),
        "QTvar_detr_loss" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_environment_2m(state).QTvar.detr_loss),
        "QTvar_shear" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_environment_2m(state).QTvar.shear),
        "QTvar_rain" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_environment_2m(state).QTvar.rain_src),
        "QTvar_interdomain" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_environment_2m(state).QTvar.interdomain),

        "HQTcov_rain" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_environment_2m(state).HQTcov.rain_src),
        "HQTcov_dissipation" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_environment_2m(state).HQTcov.dissipation),
        "HQTcov_entr_gain" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_environment_2m(state).HQTcov.entr_gain),
        "HQTcov_detr_loss" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_environment_2m(state).HQTcov.detr_loss),
        "HQTcov_shear" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_environment_2m(state).HQTcov.shear),
        "HQTcov_interdomain" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_environment_2m(state).HQTcov.interdomain),

        "env_w" => (; dims = ("zf", "t"), group = "profiles", field = state -> face_aux_environment(state).w),
        "env_qt" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_environment(state).q_tot),
        "env_ql" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_environment(state).q_liq),
        "env_qi" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_environment(state).q_ice),
        "env_area" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_environment(state).area),
        "env_temperature" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_environment(state).T),
        "env_RH" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_environment(state).RH),
        "env_thetal" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_environment(state).θ_liq_ice),
        "env_cloud_fraction" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_environment(state).cloud_fraction),
        "massflux_s" => (; dims = ("zf", "t"), group = "profiles", field = state -> face_aux_grid_mean(state).massflux_s),
        "diffusive_flux_s" => (; dims = ("zf", "t"), group = "profiles", field = state -> face_aux_grid_mean(state).diffusive_flux_s),
        "total_flux_s" => (; dims = ("zf", "t"), group = "profiles", field = state -> face_aux_grid_mean(state).massflux_s .+ face_aux_grid_mean(state).diffusive_flux_s),

        "qr_mean" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_prog_precipitation(state).q_rai),
        "qs_mean" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_prog_precipitation(state).q_sno),

        "mixing_length" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_turbconv(state).mixing_length),

        "updraft_cloud_fraction" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_bulk(state).cloud_fraction),

        "updraft_qt_precip" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_bulk(state).qt_tendency_precip_formation),
        "updraft_e_tot_precip" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_bulk(state).e_tot_tendency_precip_formation),

        "massflux_tendency_h" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_turbconv(state).massflux_tendency_h),
        "massflux_tendency_qt" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_turbconv(state).massflux_tendency_qt),
        "diffusive_tendency_h" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_turbconv(state).diffusive_tendency_h),
        "diffusive_tendency_qt" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_turbconv(state).diffusive_tendency_qt),

        "total_flux_h" => (; dims = ("zf", "t"), group = "profiles", field = state -> face_aux_turbconv(state).diffusive_flux_h .+ face_aux_turbconv(state).massflux_h),
        "total_flux_qt" => (; dims = ("zf", "t"), group = "profiles", field = state -> face_aux_turbconv(state).diffusive_flux_qt .+ face_aux_turbconv(state).massflux_qt),

        "ed_length_scheme" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_turbconv(state).mls),
        "mixing_length_ratio" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_turbconv(state).ml_ratio),
        "entdet_balance_length" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_turbconv(state).l_entdet),

        "rad_dTdt" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_grid_mean(state).dTdt_rad),
        "rad_flux" => (; dims = ("zf", "t"), group = "profiles", field = state -> face_aux_grid_mean(state).f_rad),
    )
    return io_dict
end

function io_dictionary_aux_calibrate()
    DT = NamedTuple{(:dims, :group, :field), Tuple{Tuple{String, String}, String, Any}}
    io_dict = Dict{String, DT}(
        "u_mean" => (; dims = ("zc", "t"), group = "profiles", field = state -> physical_grid_mean_u(state)),
        "v_mean" => (; dims = ("zc", "t"), group = "profiles", field = state -> physical_grid_mean_v(state)),
        "s_mean" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_grid_mean(state).s),
        "qt_mean" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_grid_mean(state).q_tot),
        "ql_mean" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_grid_mean(state).q_liq),
        "total_flux_h" => (; dims = ("zf", "t"), group = "profiles", field = state -> face_aux_turbconv(state).diffusive_flux_h .+ face_aux_turbconv(state).massflux_h),
        "total_flux_qt" => (; dims = ("zf", "t"), group = "profiles", field = state -> face_aux_turbconv(state).diffusive_flux_qt .+ face_aux_turbconv(state).massflux_qt),
        "total_flux_s" => (; dims = ("zf", "t"), group = "profiles", field = state -> face_aux_grid_mean(state).massflux_s .+ face_aux_grid_mean(state).diffusive_flux_s),
        "thetal_mean" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_grid_mean(state).θ_liq_ice),
        "tke_mean" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_grid_mean(state).tke),
        "qr_mean" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_prog_precipitation(state).q_rai),
        "qi_mean" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_grid_mean(state).q_ice),
        "qs_mean" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_prog_precipitation(state).q_sno),
        "cloud_fraction" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_grid_mean(state).cloud_fraction), # was this "cloud_fraction_mean"?
        "RH_mean" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_grid_mean(state).RH),
        "temperature_mean" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_aux_grid_mean(state).T),
    )
    return io_dict
end
#! format: on
