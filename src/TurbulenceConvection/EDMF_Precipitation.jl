"""
Computes the tendencies to θ_liq_ice, q_tot, q_rain and q_snow
due to rain evaporation, snow deposition and sublimation and snow melt
"""
function compute_precipitation_sink_tendencies(
    ::AbstractPrecipitationModel,
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

function precipitation_sinks(prog_gm, aux_gm, ts_gm, param_set, Δt)
    thermo_params = TCP.thermodynamics_params(param_set)
    microphys_params = TCP.microphysics_params(param_set)
    ρ_c = prog_gm.ρ
    qr = max(0, prog_gm.ρq_rai / ρ_c)
    qs = max(0, prog_gm.ρq_sno / ρ_c)
    ρ = ρ_c
    T_gm = aux_gm.T
    # When we fuse loops, this should hopefully disappear
    ts = ts_gm
    q = TD.PhasePartition(thermo_params, ts)
    qv = TD.vapor_specific_humidity(thermo_params, ts)

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
        α_dep_sub *
        CM1.evaporation_sublimation(microphys_params, snow_type, q, qs, ρ, T_gm)
    if tmp > 0
        S_qs_sub_dep = min(qv / Δt, tmp)
    else
        S_qs_sub_dep = -min(qs / Δt, -tmp)
    end
    return PrecipSinks(S_qr_evap, S_qs_melt, S_qs_sub_dep)
end
function compute_precipitation_sink_tendencies(
    ::Microphysics1Moment,
    state::State,
    param_set::APS,
    Δt::Real,
)
    thermo_params = TCP.thermodynamics_params(param_set)
    aux_gm = center_aux_grid_mean(state)
    aux_tc = center_aux_turbconv(state)
    prog_gm = center_prog_grid_mean(state)
    ts_gm = center_aux_grid_mean_ts(state)
    zc = Fields.coordinate_field(axes(prog_gm.ρ)).z
    ps = center_aux_turbconv(state).precip_sinks

    @. ps = precipitation_sinks(prog_gm, aux_gm, ts_gm, param_set, Δt)
    # When we fuse loops, this should hopefully disappear
    # cache or inline these
    L_f(ts) = TD.latent_heat_fusion(thermo_params, ts)
    I_l(ts) = TD.internal_energy_liquid(thermo_params, ts)
    I_i(ts) = TD.internal_energy_ice(thermo_params, ts)
    Φ(z) = geopotential(thermo_params, z)

    @. aux_tc.qr_tendency_evap = ps.S_qr_evap
    @. aux_tc.qs_tendency_melt = ps.S_qs_melt
    @. aux_tc.qs_tendency_dep_sub = ps.S_qs_sub_dep

    @. aux_tc.e_tot_tendency_precip_sinks =
        -ps.S_qr_evap * (I_l(ts_gm) + Φ(zc)) -
        ps.S_qs_sub_dep * (I_i(ts_gm) + Φ(zc)) + ps.S_qs_melt * L_f(ts_gm)
    @. aux_tc.qt_tendency_precip_sinks = -ps.S_qr_evap - ps.S_qs_sub_dep
    @. aux_tc.qr_tendency_precip_sinks = ps.S_qr_evap - ps.S_qs_melt
    @. aux_tc.qs_tendency_precip_sinks = ps.S_qs_sub_dep + ps.S_qs_melt

    return nothing
end
