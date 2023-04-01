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
    N_up = n_updrafts(edmf)
    prog_gm = center_prog_grid_mean(state)
    aux_up = center_aux_updrafts(state)
    aux_bulk = center_aux_bulk(state)
    mph = center_aux_turbconv(state).mph
    zc = Fields.coordinate_field(axes(prog_gm.ρ)).z

    @inbounds for i in 1:N_up

        # autoconversion and accretion
        @. mph = precipitation_formation(
            param_set,
            precip_model,
            prog_gm,
            aux_up[i].area,
            zc,
            Δt,
            aux_up[i].ts,
        )
        @. aux_up[i].θ_liq_ice_tendency_precip_formation =
            mph.θ_liq_ice_tendency * aux_up[i].area
        @. aux_up[i].e_tot_tendency_precip_formation =
            mph.e_tot_tendency * aux_up[i].area
        @. aux_up[i].qt_tendency_precip_formation =
            mph.qt_tendency * aux_up[i].area
        @. aux_up[i].qr_tendency_precip_formation =
            mph.qr_tendency * aux_up[i].area
        @. aux_up[i].qs_tendency_precip_formation =
            mph.qs_tendency * aux_up[i].area

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
