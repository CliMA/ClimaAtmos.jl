function microphysics(
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
    ts = center_aux_environment(state).ts
    aux_en_sat = aux_en.sat
    aux_en_unsat = aux_en.unsat
    mph = center_aux_turbconv(state).mph
    zc = Fields.coordinate_field(axes(aux_en_sat)).z

    # autoconversion and accretion
    @. mph = precipitation_formation(
        param_set,
        precip_model,
        prog_gm,
        aux_en.area,
        zc,
        Δt,
        ts,
    )

    # update_sat_unsat
    @. aux_en.cloud_fraction =
        ifelse(TD.has_condensate(thermo_params, ts), 1, 0)

    # TODO: shouldn't we always populate aux_en_sat and aux_en_unsat?
    # Otherwise we may be using values from other timesteps / stages
    conditional_assign!(aux_en_sat.θ_dry, ts, TD.dry_pottemp, thermo_params)
    conditional_assign!(
        aux_en_sat.θ_liq_ice,
        ts,
        TD.liquid_ice_pottemp,
        thermo_params,
    )
    conditional_assign!(aux_en_sat.T, ts, TD.air_temperature, thermo_params)
    conditional_assign!(
        aux_en_sat.q_tot,
        ts,
        TD.total_specific_humidity,
        thermo_params,
    )
    conditional_assign!(
        aux_en_sat.q_vap,
        ts,
        TD.vapor_specific_humidity,
        thermo_params,
    )

    conditional_assign!(
        aux_en_unsat.θ_dry,
        ts,
        TD.dry_pottemp,
        thermo_params,
        x -> !x,
    )
    conditional_assign!(
        aux_en_unsat.θ_virt,
        ts,
        TD.virtual_pottemp,
        thermo_params,
        x -> !x,
    )
    conditional_assign!(
        aux_en_unsat.q_tot,
        ts,
        TD.total_specific_humidity,
        thermo_params,
        x -> !x,
    )

    # update_env_precip_tendencies
    # TODO: move ..._tendency_precip_formation to diagnostics
    @. aux_en.e_tot_tendency_precip_formation = mph.e_tot_tendency * aux_en.area
    @. aux_en.qt_tendency_precip_formation = mph.qt_tendency * aux_en.area
    @. aux_en.qr_tendency_precip_formation = mph.qr_tendency * aux_en.area
    @. aux_en.qs_tendency_precip_formation = mph.qs_tendency * aux_en.area
    return nothing
end

function conditional_assign!(v, ts, f::Function, thermo_params, cond = x -> x)
    @. v = ifelse(
        cond(TD.has_condensate(thermo_params, ts)),
        f(thermo_params, ts),
        v,
    )
    nothing
end
