"""
    TODO: Make sure this is consistent with the function in Thermodynamics.jl.
"""
# Once PR is merged in ClimaParams, import from there.
# https://github.com/CliMA/ClimaParams.jl/pull/253
const s_ref = 7

function exner_function(thermo_params, ts)
    R_d = TD.TP.R_d(thermo_params)
    cp_d = TD.TP.cp_d(thermo_params)
    p_0 = TD.TP.p_ref_theta(thermo_params)
    p = TD.air_pressure(thermo_params, ts)
    return (p / p_0)^(R_d / cp_d)
end

function theta_v(thermo_params, ts)
    R_d = TD.TP.R_d(thermo_params)
    T = TD.air_temperature(thermo_params, ts)
    R_m = TD.gas_constant_air(thermo_params, ts)
    Π = exner_function(thermo_params, ts)
    return T * R_m / (Π * R_d)
end

function air_temperature_reference(thermo_params, ts)
    T_min = TD.TP.T_min_ref(thermo_params)
    T_sfc = TD.TP.T_surf_ref(thermo_params)

    Π = exner_function(thermo_params, ts)

    return T_min + (T_sfc - T_min) * Π^(s_ref)
end

function theta_vr(thermo_params, ts)
    T_r = air_temperature_reference(thermo_params, ts)
    Π = exner_function(thermo_params, ts)
    return T_r / Π
end

function phi_r(thermo_params, ts)
    cp_d = TD.TP.cp_d(thermo_params)
    T_min = TD.TP.T_min_ref(thermo_params)
    T_sfc = TD.TP.T_surf_ref(thermo_params)

    Π = exner_function(thermo_params, ts)

    return -cp_d * (T_min * log(Π) + (T_sfc - T_min) / (s_ref) * (Π^(s_ref) - 1))
end

function h_dr(thermo_params, ts, Φ)
    T_0 = TD.TP.T_0(thermo_params)
    cp_d = TD.TP.cp_d(thermo_params)

    T_r = air_temperature_reference(thermo_params, ts)
    return cp_d * (T_r - T_0) + Φ
end
