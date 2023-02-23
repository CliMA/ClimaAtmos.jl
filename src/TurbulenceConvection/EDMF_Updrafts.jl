"""
Computes tendencies to qt and θ_liq_ice due to precipitation formation
"""
function compute_precipitation_formation_tendencies(
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
    aux_gm = center_aux_grid_mean(state)
    aux_up = center_aux_updrafts(state)
    aux_bulk = center_aux_bulk(state)
    tendencies_gm = center_tendencies_grid_mean(state)
    p_c = center_aux_grid_mean_p(state)
    ρ_c = prog_gm.ρ

    precip_fraction = compute_precip_fraction(edmf.precip_fraction_model, state)

    @inbounds for i in 1:N_up
        @inbounds for k in real_center_indices(grid)
            T_up = aux_up[i].T[k]
            q_tot_up = aux_up[i].q_tot[k]
            ts_up = TD.PhaseEquil_pTq(thermo_params, p_c[k], T_up, q_tot_up)

            # autoconversion and accretion
            mph = precipitation_formation(
                param_set,
                precip_model,
                prog_gm.ρq_rai[k] / ρ_c[k],
                prog_gm.ρq_sno[k] / ρ_c[k],
                aux_up[i].area[k],
                ρ_c[k],
                FT(grid.zc[k].z),
                Δt,
                ts_up,
                precip_fraction,
            )
            aux_up[i].qt_tendency_precip_formation[k] =
                mph.qt_tendency * aux_up[i].area[k]
            aux_up[i].θ_liq_ice_tendency_precip_formation[k] =
                mph.θ_liq_ice_tendency * aux_up[i].area[k]
            aux_up[i].e_tot_tendency_precip_formation[k] =
                mph.e_tot_tendency * aux_up[i].area[k]
            tendencies_gm.ρq_rai[k] +=
                ρ_c[k] * mph.qr_tendency * aux_up[i].area[k]
            tendencies_gm.ρq_sno[k] +=
                ρ_c[k] * mph.qs_tendency * aux_up[i].area[k]
        end
    end
    # TODO - to be deleted once we sum all tendencies elsewhere
    @inbounds for k in real_center_indices(grid)
        aux_bulk.e_tot_tendency_precip_formation[k] = 0
        aux_bulk.qt_tendency_precip_formation[k] = 0
        @inbounds for i in 1:N_up
            aux_bulk.e_tot_tendency_precip_formation[k] +=
                aux_up[i].e_tot_tendency_precip_formation[k]
            aux_bulk.qt_tendency_precip_formation[k] +=
                aux_up[i].qt_tendency_precip_formation[k]
        end
    end
    return nothing
end
