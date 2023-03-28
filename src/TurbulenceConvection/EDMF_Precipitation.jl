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

"""
Computes the tendencies to the grid mean e_tot, q_tot, q_rain and q_snow
due to rain evaporation, snow deposition and sublimation and snow melt.

TODO - should be moved outside of the SGS model.
"""
function grid_mean_precipitation_sink_tendencies(
    ::AbstractPrecipitationModel,
    grid::Grid,
    state::State,
    param_set::APS,
    Δt::Real,
)
    FT = float_type(state)
    aux_tc = center_aux_turbconv(state)
    @. aux_tc.qt_tendency_precip_sinks = FT(0)
    @. aux_tc.e_tot_tendency_precip_sinks = FT(0)

    nothing
end
function grid_mean_precipitation_sink_tendencies(
    ::Microphysics1Moment,
    grid::Grid,
    state::State,
    param_set::APS,
    Δt::Real,
)
    thermo_params = TCP.thermodynamics_params(param_set)
    microphys_params = TCP.microphysics_params(param_set)
    aux_gm = center_aux_grid_mean(state)
    aux_tc = center_aux_turbconv(state)
    prog_gm = center_prog_grid_mean(state)
    ρ_c = prog_gm.ρ
    ts_gm = center_aux_grid_mean_ts(state)

    FT = float_type(state)

    @inbounds for k in real_center_indices(grid)
        qr = max(FT(0), prog_gm.ρq_rai[k] / ρ_c[k])
        qs = max(FT(0), prog_gm.ρq_sno[k] / ρ_c[k])
        ρ = ρ_c[k]
        T_gm = aux_gm.T[k]
        # When we fuse loops, this should hopefully disappear
        ts = ts_gm[k]
        q = TD.PhasePartition(thermo_params, ts)
        qv = TD.vapor_specific_humidity(thermo_params, ts)
        L_f = TD.latent_heat_fusion(thermo_params, ts)

        I_l = TD.internal_energy_liquid(thermo_params, ts)
        I_i = TD.internal_energy_ice(thermo_params, ts)
        Φ = geopotential(thermo_params, grid.zc.z[k])

        α_evp = TCP.microph_scaling(param_set)
        α_dep_sub = TCP.microph_scaling_dep_sub(param_set)
        α_melt = TCP.microph_scaling_melt(param_set)

        # TODO - move limiters elsewhere
        # TODO - when using adaptive timestepping we are limiting the source terms
        #        with the previous timestep dt
        S_qr_evap =
            -min(
                qr / Δt,
                -α_evp * CM1.evaporation_sublimation(
                    microphys_params,
                    rain_type,
                    q,
                    qr,
                    ρ,
                    T_gm,
                ),
            )
        S_qs_melt =
            -min(qs / Δt, α_melt * CM1.snow_melt(microphys_params, qs, ρ, T_gm))
        tmp =
            α_dep_sub * CM1.evaporation_sublimation(
                microphys_params,
                snow_type,
                q,
                qs,
                ρ,
                T_gm,
            )
        if tmp > 0
            S_qs_sub_dep = min(qv / Δt, tmp)
        else
            S_qs_sub_dep = -min(qs / Δt, -tmp)
        end

        aux_tc.e_tot_tendency_precip_sinks[k] =
            -S_qr_evap * (I_l + Φ) - S_qs_sub_dep * (I_i + Φ) + S_qs_melt * L_f
        aux_tc.qt_tendency_precip_sinks[k] = -S_qr_evap - S_qs_sub_dep
        aux_tc.qr_tendency_precip_sinks[k] = S_qr_evap - S_qs_melt
        aux_tc.qs_tendency_precip_sinks[k] = S_qs_sub_dep + S_qs_melt
    end
    return nothing
end
