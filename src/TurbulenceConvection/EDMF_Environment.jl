function microphysics(
    grid::Grid,
    state::State,
    edmf::EDMFModel,
    precip_model::AbstractPrecipitationModel,
    Δt::Real,
    param_set::APS,
)
    FT = float_type(state)
    thermo_params = TCP.thermodynamics_params(param_set)
    aux_en = center_aux_environment(state)
    prog_gm = center_prog_grid_mean(state)
    ts_env = center_aux_environment(state).ts
    aux_en_sat = aux_en.sat
    aux_en_unsat = aux_en.unsat

    @inbounds for k in real_center_indices(grid)
        # condensation
        ts = ts_env[k]
        # autoconversion and accretion
        mph = precipitation_formation(
            param_set,
            precip_model,
            prog_gm[k],
            aux_en.area[k],
            FT(grid.zc[k].z),
            Δt,
            ts,
        )

        # update_sat_unsat
        if TD.has_condensate(thermo_params, ts)
            aux_en.cloud_fraction[k] = 1
            aux_en_sat.θ_dry[k] = TD.dry_pottemp(thermo_params, ts)
            aux_en_sat.θ_liq_ice[k] = TD.liquid_ice_pottemp(thermo_params, ts)
            aux_en_sat.T[k] = TD.air_temperature(thermo_params, ts)
            aux_en_sat.q_tot[k] = TD.total_specific_humidity(thermo_params, ts)
            aux_en_sat.q_vap[k] = TD.vapor_specific_humidity(thermo_params, ts)
        else
            aux_en.cloud_fraction[k] = 0
            aux_en_unsat.θ_dry[k] = TD.dry_pottemp(thermo_params, ts)
            aux_en_unsat.θ_virt[k] = TD.virtual_pottemp(thermo_params, ts)
            aux_en_unsat.q_tot[k] =
                TD.total_specific_humidity(thermo_params, ts)
        end

        # update_env_precip_tendencies
        # TODO: move ..._tendency_precip_formation to diagnostics
        aux_en.e_tot_tendency_precip_formation[k] =
            mph.e_tot_tendency * aux_en.area[k]
        aux_en.qt_tendency_precip_formation[k] =
            mph.qt_tendency * aux_en.area[k]
        aux_en.qr_tendency_precip_formation[k] =
            mph.qr_tendency * aux_en.area[k]
        aux_en.qs_tendency_precip_formation[k] =
            mph.qs_tendency * aux_en.area[k]
    end
    return nothing
end
