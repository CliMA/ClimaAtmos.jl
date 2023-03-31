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
    aux_up = center_aux_updrafts(state)
    aux_bulk = center_aux_bulk(state)
    p_c = center_aux_grid_mean_p(state)
    ρ_c = prog_gm.ρ

    @inbounds for i in 1:N_up
        @inbounds for k in real_center_indices(grid)
            ts_up = aux_up[i].ts[k]

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
    @. aux_bulk.e_tot_tendency_precip_formation = 0
    @. aux_bulk.qt_tendency_precip_formation = 0
    @. aux_bulk.qr_tendency_precip_formation = 0
    @. aux_bulk.qs_tendency_precip_formation = 0
    @inbounds for i in 1:N_up
        @. aux_bulk.e_tot_tendency_precip_formation +=
            aux_up[i].e_tot_tendency_precip_formation
        @. aux_bulk.qt_tendency_precip_formation +=
            aux_up[i].qt_tendency_precip_formation
        @. aux_bulk.qr_tendency_precip_formation +=
            aux_up[i].qr_tendency_precip_formation
        @. aux_bulk.qs_tendency_precip_formation +=
            aux_up[i].qs_tendency_precip_formation
    end
    return nothing
end
