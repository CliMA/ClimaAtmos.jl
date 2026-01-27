"""
    TODO: Make sure this is consistent with the function in Thermodynamics.jl.
"""
# Once PR is merged in ClimaParams, import from there.
# https://github.com/CliMA/ClimaParams.jl/pull/253
const s_ref = 7

function theta_v(thermo_params, T, p, q_tot, q_liq, q_ice)
    R_d = TD.TP.R_d(thermo_params)
    R_m = TD.gas_constant_air(thermo_params, TD.PhasePartition(q_tot, q_liq, q_ice))
    Π = TD.exner_given_pressure(thermo_params, p)
    return T * R_m / (Π * R_d)
end

function air_temperature_reference(thermo_params, p)
    T_min = TD.TP.T_min_ref(thermo_params)
    T_sfc = TD.TP.T_surf_ref(thermo_params)

    Π = TD.exner_given_pressure(thermo_params, p)

    return T_min + (T_sfc - T_min) * Π^s_ref
end

function theta_vr(thermo_params, p)
    T_r = air_temperature_reference(thermo_params, p)
    Π = TD.exner_given_pressure(thermo_params, p)
    return T_r / Π
end

function phi_r(thermo_params, p)
    cp_d = TD.TP.cp_d(thermo_params)
    T_min = TD.TP.T_min_ref(thermo_params)
    T_sfc = TD.TP.T_surf_ref(thermo_params)

    Π = TD.exner_given_pressure(thermo_params, p)

    return -cp_d * (T_min * log(Π) + (T_sfc - T_min) / s_ref * (Π^s_ref - 1))
end

function h_dr(thermo_params, p, Φ)
    T_0 = TD.TP.T_0(thermo_params)
    cp_d = TD.TP.cp_d(thermo_params)

    T_r = air_temperature_reference(thermo_params, p)
    return cp_d * (T_r - T_0) + Φ
end
