"""
    compute_precip_fraction

Computes diagnostic precipitation fraction
"""
compute_precip_fraction(
    precip_fraction_model::PrescribedPrecipFraction,
    ::State,
) = precip_fraction_model.prescribed_precip_frac_value

function compute_precip_fraction(
    precip_fraction_model::DiagnosticPrecipFraction,
    state::State,
)
    aux_gm = center_aux_grid_mean(state)
    maxcf = maximum(aux_gm.cloud_fraction)
    return max(maxcf, precip_fraction_model.precip_fraction_limiter)
end

"""
Computes the tendencies to θ_liq_ice, q_tot, q_rain and q_snow
due to rain evaporation, snow deposition and sublimation and snow melt
"""
function compute_precipitation_sink_tendencies(
    ::AbstractPrecipitationModel,
    ::AbstractPrecipFractionModel,
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

function compute_precipitation_sink_tendencies(
    ::Microphysics1Moment,
    precip_fraction_model::AbstractPrecipFractionModel,
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
    prog_pr = center_prog_precipitation(state)
    ρ_c = prog_gm.ρ
    tendencies_pr = center_tendencies_precipitation(state)
    ts_gm = center_aux_grid_mean_ts(state)

    precip_fraction = compute_precip_fraction(precip_fraction_model, state)

    FT = float_type(state)

    @inbounds for k in real_center_indices(grid)
        qr = max(FT(0), prog_pr.q_rai[k]) / precip_fraction
        qs = max(FT(0), prog_pr.q_sno[k]) / precip_fraction
        ρ = ρ_c[k]
        q_tot_gm = aux_gm.q_tot[k]
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
            ) * precip_fraction
        S_qs_melt =
            -min(
                qs / Δt,
                α_melt * CM1.snow_melt(microphys_params, qs, ρ, T_gm),
            ) * precip_fraction
        tmp =
            α_dep_sub *
            CM1.evaporation_sublimation(
                microphys_params,
                snow_type,
                q,
                qs,
                ρ,
                T_gm,
            ) *
            precip_fraction
        if tmp > 0
            S_qs_sub_dep = min(qv / Δt, tmp)
        else
            S_qs_sub_dep = -min(qs / Δt, -tmp)
        end

        aux_tc.qr_tendency_evap[k] = S_qr_evap
        aux_tc.qs_tendency_melt[k] = S_qs_melt
        aux_tc.qs_tendency_dep_sub[k] = S_qs_sub_dep

        tendencies_pr.q_rai[k] += S_qr_evap - S_qs_melt
        tendencies_pr.q_sno[k] += S_qs_sub_dep + S_qs_melt

        aux_tc.qt_tendency_precip_sinks[k] = -S_qr_evap - S_qs_sub_dep
        aux_tc.e_tot_tendency_precip_sinks[k] =
            -S_qr_evap * (I_l + Φ) - S_qs_sub_dep * (I_i + Φ) + S_qs_melt * L_f
    end
    return nothing
end
