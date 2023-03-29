"""
Computes tendencies to q_tot, q_rai, q_sno, θ_liq_ice and e_tot
due to precipitation formation in the updrafts.
"""
function updraft_precipitation_formation_tendencies(
    grid::Grid,
    state::State,
    edmf::EDMFModel,
    precip_model::AbstractPrecipitationModel,
    Δt::Real,
    param_set::APS,
)
    thermo_params = TCP.thermodynamics_params(param_set)
    FT = float_type(state)
    N_up = n_updrafts(edmf)
    prog_gm = center_prog_grid_mean(state)
    aux_up = center_aux_updrafts(state)
    aux_bulk = center_aux_bulk(state)
    p_c = center_aux_grid_mean_p(state)
    ρ_c = prog_gm.ρ

    @inbounds for i in 1:N_up
        @inbounds for k in real_center_indices(grid)
            T_up = aux_up[i].T[k]
            q_tot_up = aux_up[i].q_tot[k]
            ts_up = TD.PhaseEquil_pTq(thermo_params, p_c[k], T_up, q_tot_up)

            # autoconversion and accretion
            mph = precipitation_formation(
                param_set,
                precip_model,
                prog_gm[k],
                aux_up[i].area[k],
                FT(grid.zc[k].z),
                Δt,
                ts_up,
            )
            aux_up[i].θ_liq_ice_tendency_precip_formation[k] =
                mph.θ_liq_ice_tendency * aux_up[i].area[k]
            aux_up[i].e_tot_tendency_precip_formation[k] =
                mph.e_tot_tendency * aux_up[i].area[k]
            aux_up[i].qt_tendency_precip_formation[k] =
                mph.qt_tendency * aux_up[i].area[k]
            aux_up[i].qr_tendency_precip_formation[k] =
                mph.qr_tendency * aux_up[i].area[k]
            aux_up[i].qs_tendency_precip_formation[k] =
                mph.qs_tendency * aux_up[i].area[k]
        end
    end
    # TODO - to be deleted once we sum all tendencies elsewhere
    @inbounds for k in real_center_indices(grid)
        aux_bulk.e_tot_tendency_precip_formation[k] = 0
        aux_bulk.qt_tendency_precip_formation[k] = 0
        aux_bulk.qr_tendency_precip_formation[k] = 0
        aux_bulk.qs_tendency_precip_formation[k] = 0
        @inbounds for i in 1:N_up
            aux_bulk.e_tot_tendency_precip_formation[k] +=
                aux_up[i].e_tot_tendency_precip_formation[k]
            aux_bulk.qt_tendency_precip_formation[k] +=
                aux_up[i].qt_tendency_precip_formation[k]
            aux_bulk.qr_tendency_precip_formation[k] +=
                aux_up[i].qr_tendency_precip_formation[k]
            aux_bulk.qs_tendency_precip_formation[k] +=
                aux_up[i].qs_tendency_precip_formation[k]
        end
    end
    return nothing
end

"""
Computes tendencies to q_tot, q_rai, q_sno and e_tot
due to precipitation formation in the environment.
"""
function environment_precipitation_formation_tendencies(
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
